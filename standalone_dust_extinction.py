import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from synphot import SpectralElement, Observation
import stsynphot as stsyn
import astropy.units as u
from scipy import constants as con
from dustmaps.sfd import SFDQuery
from matplotlib import cm as cm
from astroquery.ipac.irsa import Irsa
import pacs_weight_functions as pacsweights
from astropy.coordinates import FK4, FK5
from astropy.wcs.utils import skycoord_to_pixel
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from model_envelope import Envelope
import logging

import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=10)

Irsa.ROW_LIMIT = 4000

sfd = SFDQuery()  # Not sure why I have to do this, but dustmaps requires it

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

#######
# This block sets parameter values, initializes bandpasses
# and defines utility functions.
#######
output_filename = 'output/apt_sample_serpens_field_add_all_k2v.tbl'

CK_modelfile = 'params/ckmodels.txt'
dense_extcurve_file = 'params/chapman09_extinction.asc'
diffuse_extcurve_file = 'params/kext_albedo_WD_MW_3.1_60_D03.all'

r_v_dense = 5.0
r_v_foreground = 3.1
AU = con.au
# HC in units of Angstrom * erg
HC = con.h * con.c * 1.e17

# From Pontoppidan et al. 2004, use fig10 to find N_H/A_J relation
# => 1.e23 cm^-2/30 A_J => 3.33e21 cm^-2/A_J  => 3.33e25 m^-2/A_J
n_h_per_A_J = 3.33e25  # m^-2/A_J

serpens_main_center = SkyCoord('18h29m54s', '1d15m20.1s', frame='icrs')
distance_serpens_pc = 436.  # pc
off_field_angle = 1.9  # deg

fit_stellar_types = True
default_type = 'K2V'  # Used as CK model if fit_stellar_types is False

plot_stuff = True  # Plot some diagnostic plots to figures/


def load_bandpass_dict():
    """Load in bandpasses from file or pysynphot,
    store in dictionary. Also provide order for
    bandpasses to be listed in arrays and outputs."""
    bpdict = dict()

    bpdict["V"] = SpectralElement.from_filter('johnson_v')
    bpdict["J"] = SpectralElement.from_file('params/bp_2mass_j_scaled.tbl')
    bpdict["H"] = SpectralElement.from_file('params/bp_2mass_h_scaled.tbl')
    bpdict["K"] = SpectralElement.from_file('params/bp_2mass_k_scaled.tbl')
    bpdict["IRAC1"] = SpectralElement.from_file('params/irac_3p6um_filter.tbl')
    bporder = ["V", "J", "H", "K", "IRAC1"]

    return bpdict, bporder


def load_zeropoint_dict():
    """Load zero-magnitude fluxes for relevant filters.
    All values in units of erg cm-2 s-1 A-1"""
    zeropoint_dict = dict()

    # Define Johnson V zero point from http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    zeropoint_dict["V"] = 3.631e-09

    # Define 0 Mag Fluxes for J H K, taken from https://www.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html
    # Converted from webpage to erg cm-2 s-1 A-1 by multiplying 10^3
    zeropoint_dict["J"] = 3.129e-10  # erg cm-2 s-1 A-1
    zeropoint_dict["H"] = 1.133e-10  # erg cm-2 s-1 A-1
    zeropoint_dict["K"] = 4.283e-11  # erg cm-2 s-1 A-1

    # 0 Mag Fluxes for IRAC 3.6 um from https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/17/
    # Converted from webpage to erg cm^-2 A^-1 by multiplying 2.998e-5/lam^2(in Angstrom)
    zeropoint_dict["IRAC1"] = 6.498e-12  # erg cm^-2 A^-1

    return zeropoint_dict


def flux_to_mag(spwv, spflx, bp, zeromagflx=None):
    """Observe a spectrum with a bandpass,
    provide result in magnitudes if zero-mag
    flux provided, otherwise provide flux
    """
    bp_spwv = np.interp(spwv, bp.waveset.value, bp(bp.waveset))
    jobs = spflx * bp_spwv
    jredeff = np.trapz(jobs * HC / spwv, x=spwv)
    jresponse = np.trapz(bp(bp.waveset) * HC / bp.waveset.value, x=bp.waveset.value)
    jflx = (jredeff / jresponse).value
    if zeromagflx is not None:
        return -2.5 * np.log10(jflx / zeromagflx)
    else:
        return jflx


def filter_extinction(wav, ext, bandpass_dict):
    """Apply filter dict directly to extinction curve.
    Produce an array that allows for conversion from
    magnitudes of extinction in one band to another,
    with output response normalized to response of
    Johnson V band.

    Parameters
    ----------
    wav, ext : array[float]
        Extinction curve in units of micron and cm**2/g
    bandpass_dict : dict
        Dictionary with bandpass names as keys, with
        pysynphot Bandpass() objects as values

    Returns
    -------
    ext_response : dict
        Dictionary containing bandpass names as keys,
        with the extincted filter throughput as values
    """
    # Assumes wav/ext are from extinction curve [in um], convert to angstrom:
    wav = wav * 1.e4
    ext_response = dict()
    for key in bandpass_dict:
        bp_interp = np.interp(wav, bandpass_dict[key].waveset.value,
                              bandpass_dict[key](bandpass_dict[key].waveset))
        jobs = ext * bp_interp
        jredeff = np.trapz(jobs * HC / wav, x=wav)
        jresponse = np.trapz(bandpass_dict[key](bandpass_dict[key].waveset) *
                             HC / bandpass_dict[key].waveset.value,
                             x=bandpass_dict[key].waveset.value)
        ext_response[key] = jredeff / jresponse
    return {i: (ext_response[i] / ext_response["V"]).value for i in ext_response.keys()}


def translate_in_galactic(coord, position_angle, separation):
    """Takes in SkyCoord object, translates location in galactic frame
    position_angle: 0 deg = positive change in latitude,
                   90 deg = positive change in longitude, etc.
    (Assume return wanted in fk5)
    """
    galcoord = coord.galactic
    offset = galcoord.directional_offset_by(position_angle, separation)
    return offset.fk5


def foreground_extinction(coord):
    """Queries SFD dust map, returns E(B-V)
    Multiplies by header-defined r_v_foreground
    to return value in A_V
    """
    return sfd(coord) * r_v_foreground


def query_source_table(coord, offset=1.9, cone_search_radius=6., local_override=None):
    """Query IRSA database in a cone search around a point
    specified by the coord given, then offset in galactic longitude
    to avoid local extinction

    Parameters
    ----------
    coord : astropy.SkyCoord
        Coordinate center of the region to query.
    offset : float
        Offset to apply to coord in galactic longitude, in degrees.
    cone_search_radius : float
        Radius of the cone search around the (possibly offset) coord.
        Units of arcmin applied internally.
    local_override : str, optional
        If desired, can provide a filename of an IPAC table instead
        of querying.

    Returns
    -------
    table
        astropy.table.table.Table
    """
    if local_override is not None:
        sourcetable = ascii.read(local_override)
    else:
        gal_coord = coord.galactic
        gal_coord_values = [gal_coord.l.to_value(), gal_coord.b.to_value()]
        offset_location = SkyCoord(gal_coord_values[0] + offset, gal_coord_values[1], unit='deg', frame='galactic')

        log.info(f"For Serpens Main, SFD Ext is {sfd(serpens_main_center):.2f}")
        log.info(f"For {offset:.1f} deg off Serpens Main, SFD Ext is {sfd(offset_location):.2f}")

        sourcetable = Irsa.query_region(offset_location.fk5, catalog="fp_psc", spatial="Cone",
                                        radius=cone_search_radius * u.arcmin)
    return sourcetable


def load_ck_models(modelfile):
    """Load Castelli-Kurucz stellar models as dictionary
    of pysynphot sources, with spectral type as dict keys.
    """
    with open(modelfile, 'r') as fle:
        cklist = [line.split() for line in fle][1:]

    modeldict = {}
    for line in cklist:
        tmp = stsyn.grid_to_spec('ck04models', float(line[1]), 0.0, float(line[2]))
        if tmp.integrate().value > 0.:
            modeldict[line[0]] = tmp

    return modeldict


def gen_2MASS_colordict(sourcedict, bpdict, zpdict, extinctions):
    """
    Generate dictionary of j-h, h-k, j-k colors for dictionary
    of sources. Takes dicts of J, H, K filters in bpdict, zero-
    point fluxes (in flam) via zpdict, extinction magnitudes via
    extinctions
    """
    colordict = {}
    for source in sourcedict.keys():
        mags = dict()
        for key in bpdict.keys():
            obs = Observation(sourcedict[source], bpdict[key])
            flx = obs.effstim(flux_unit='flam')
            mags[key] = -2.5 * np.log10(flx.value / zpdict[key]) + extinctions[key]
        colordict[source] = [mags["J"] - mags["H"], mags["H"] - mags["K"], mags["J"] - mags["K"]]
    return colordict


def load_ext_curves(diffuse_file, dense_file):
    """Load in extinction curve data.
    NB: Filenames have been brought to the top of file for
    clarity on file requirements to run the code, but this
    function assumes the shape of these file contents and so
    filenames are not interchangeable - if changing the input
    extinction curve is desired, this function will need to
    be modified.

    Parameters
    ----------
    diffuse_file : str
        Filename of the Weingartner-Draine extinction curve.
    dense_file : str
        Filename of the Chapman extinction curve.

    Returns
    -------
    diffuse_wav, diffuse_ext, dense_wav, dense_ext : array[float]
        Arrays of wavelength [in micron] and extinction [in cm**2/g]
        for each curve
    """
    with open(dense_file, 'r') as f:
        f.readline()
        chap_arr = np.loadtxt(f)

    dense_lam = chap_arr[:, 0][::-1]
    dense_ext = (chap_arr[:, 1] + chap_arr[:, 2])[::-1]

    with open(diffuse_file, 'r') as g:
        [g.readline() for _ in range(80)]
        wdmw_arr = np.loadtxt(g, usecols=range(6))

    wd_lam = wdmw_arr[:, 0][::-1]
    wd_ext = (1. + wdmw_arr[:, 1][::-1]) * wdmw_arr[:, 4][::-1]

    return wd_lam, wd_ext, dense_lam, dense_ext


def observe_model(model, coord, kmag_observed, bandpass_key):
    """First find diffuse extinction at location of source by converting Av
    from dustmap into Ak. Remove extinction from observed magnitude,
    then renormalize model to that value. Finally observe renormalized
    source in irac 3p6, then re-extinct. (Need to re-extinct?)
    """
    foreground_Av = foreground_extinction(coord)
    diffuse_extinction_mags = {key: foreground_Av * filter_extinctions_wd[key] for key in filter_extinctions_wd.keys()}
    base_k = kmag_observed - diffuse_extinction_mags["K"]
    renormed = model.normalize(base_k * u.ABmag, band=bp_dict["K"])
    base_bp = flux_to_mag(renormed.waveset.value, renormed(renormed.waveset).value, bp_dict[bandpass_key], zp_dict[bandpass_key])
    return base_bp


def find_stellar_fit(jmag, hmag, kmag, coord, colordict, do_fit=fit_stellar_types, default_ck=default_type):
    foreground_Av = foreground_extinction(coord)
    diffuse_extinction_mags = {key: foreground_Av * filter_extinctions_wd[key] for key in filter_extinctions_wd.keys()}
    base_j = jmag - diffuse_extinction_mags["J"]
    base_h = hmag - diffuse_extinction_mags["H"]
    base_k = kmag - diffuse_extinction_mags["K"]
    j_h = base_j - base_h
    h_k = base_h - base_k
    j_k = base_j - base_k
    bestresidual = 1.e10
    bestkey = default_ck
    if do_fit:
        for key in colordict.keys():
            modjh, modhk, modjk = colordict[key]
            residual = ((modjh - j_h) ** 2 +
                        (modhk - h_k) ** 2 +
                        (modjk - j_k) ** 2)
            if residual < bestresidual:
                bestresidual = residual
                bestkey = key
    return bestkey, base_j, base_h, base_k


def fit_sourcetable(source_table, colordict_models, fit_stellar_types, default_type):
    fitted_models = dict()

    bestfit_key_list = [''] * len(source_table)

    for i in range(len(source_table)):

        coords = SkyCoord(source_table[i]['ra'], source_table[i]['dec'], unit='deg', frame='fk5')

        fitted_model_name, model_j, model_h, model_k = find_stellar_fit(
            source_table[i]['j_m'], source_table[i]['h_m'],
            source_table[i]['k_m'], coords, colordict_models,
            fit_stellar_types, default_type)

        fitted_models[i] = {"J": model_j, "H": model_h, "K": model_k}

        for key in bp_dict.keys():
            if key not in ["J", "H", "K"]:
                fitted_models[i][key] = observe_model(model_dict[fitted_model_name],
                                                      coords, source_table[i]['k_m'], key)
        bestfit_key_list[i] = fitted_model_name

    return fitted_models, bestfit_key_list


def move_source_field(sourcetable, offset=1.9):
    """Perform the coordinate translation back to Serpens."""
    # Store SkyCoord objects for initial location and Serpens locations
    offsetcoords = [None] * len(sourcetable)
    serpenscoords = [None] * len(sourcetable)

    # Move sources to Serpens, store SkyCoord objects in coords lists for ease of use
    for i in range(len(source_table)):
        offsetcoords[i] = SkyCoord(sourcetable[i]['ra'], sourcetable[i]['dec'], unit='deg', frame='fk5')
        serpenscoords[i] = translate_in_galactic(offsetcoords[i], 90. * u.deg, -1. * offset * u.deg)
        sourcetable[i]['ra'] = serpenscoords[i].ra.to_value()
        sourcetable[i]['dec'] = serpenscoords[i].dec.to_value()
    return offsetcoords, serpenscoords, sourcetable


def load_smm_envelopes():
    # These numbers sourced from Davis et al. 1999, using SCUBA 450um fluxes
    smm4_450 = 10.8

    # Power law scaling to smooth effect of SMM1 large size and flux when compared to median envelope
    scalefactor = 0.8

    smmscale = (np.array([35.7, 3.4, 7.1, 10.8, 3.9, 1.9, 1.3, 9.1, 2.7, 2.7]) / smm4_450) ** scalefactor
    smmlist = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    smm_scaledict = {}
    for i in range(len(smmscale)):
        smm_scaledict[smmlist[i]] = smmscale[i]

    envelope_dict = dict()
    envelope_dict[1] = SkyCoord('18h29m49.9s', '1d15m18.6s', frame=FK5)
    envelope_dict[2] = SkyCoord('18h30m0.3s', '1d12m57.3s', frame=FK5)
    envelope_dict[3] = SkyCoord('18h29m59.3s', '1d13m58.3s', frame=FK5)
    envelope_dict[4] = SkyCoord('18h29m56.6s', '1d13m10.1s', frame=FK5)
    envelope_dict[5] = SkyCoord('18h29m51.1s', '1d16m35.7s', frame=FK5)
    envelope_dict[6] = SkyCoord('18h27m25.3s', '1d11m57s', frame=FK4, equinox='J1950')
    envelope_dict[8] = SkyCoord('18h27m29.5s', '1d13m02s', frame=FK4, equinox='J1950')
    envelope_dict[9] = SkyCoord('18h29m48.1s', '1d16m41.5s', frame=FK5)
    envelope_dict[10] = SkyCoord('18h27m19.8s', '1d13m43s', frame=FK4, equinox='J1950')
    envelope_dict[11] = SkyCoord('18h30m00.2s', '1d11m42.3s', frame=FK5)

    return smm_scaledict, envelope_dict


def set_source_weights_pacs(sourcecoords):
    pacs_img, pacs_wcs, fit = pacsweights.load_pacs_image(pacsweights.pacs_file_for_interpolation)
    weights = np.zeros(len(serpens_table))
    for w in range(len(weights)):
        weights[w] = pacsweights.find_weight_log(sourcecoords[w], pacs_wcs, fit, 0.04, 0.25)

    return weights, pacs_img, pacs_wcs, fit


def calc_extinctions(sourcecoords, envelope_model, envelope_coords, diff_ext_filters, dense_ext_filters):
    """Given list of coordinates, for each find the diffuse extinction
    due to foreground dustmap. Then determine if coordinate lies within
    modeled envelope extents - if so, compute additional dense extinction.

    Parameters
    ----------
    sourcecoords : arr[ncoords]
        Array of SkyCoord objects

    Returns
    -------
    diff_ext, dense_ext : arr[ncoords, nbands]
        Extinction, in magnitudes of each band for diffuse and
        dense contributions
    count_envelopes : arr[ncoords]
        Number of envelopes encountered along the line of sight to
        this coordinate
    """
    diffuse_extinctions = np.zeros((len(sourcecoords), nbands))
    dense_extinctions = np.zeros((len(sourcecoords), nbands))
    count_envelopes = np.zeros(len(sourcecoords), dtype='int')

    for i in range(len(sourcecoords)):
        foreground_ext = foreground_extinction(sourcecoords[i])
        diffuse_extinctions[i] = foreground_ext * np.array([diff_ext_filters[bp] for bp in bp_order])
        extinctions = np.zeros(nbands)
        envcount = 0
        for key in envelope_coords.keys():
            # Find projected distance between each source `i` and envelope `key`, in AU
            envelope_distance_au = envelope_coords[key].separation(serpens_coords[i]).arcsec * distance_serpens_pc

            # Using conversion factor and computed envelope surface density at this separation, scaled by
            # the SMM flux to the 'fitted' SMM4 envelope, compute extinction in J band.
            a_j = envelope_model.surface_density(envelope_distance_au * AU) / n_h_per_A_J * smm_scaledflux_dict[key]

            # Scale extinction in other bands by filter_extinctions

            extinctions = extinctions + np.array(
                [dense_ext_filters[bp] / dense_ext_filters["J"] * a_j for bp in bp_order]).flatten()
            # Count the envelope if contribution is non-zero
            if a_j > 0.1:
                envcount += 1
        count_envelopes[i] = envcount
        dense_extinctions[i] = extinctions

    # Combine Diffuse and Envelope Extinction to provide final additional extinction added to each source
    return diffuse_extinctions, dense_extinctions, count_envelopes


def print_extinction(ext_arr):
    str_return = ''
    for x in ext_arr:
        str_return = str_return + f"{x:6.2f} "
    return str_return


def write_output_table(flename):
    final_obs_mags = np.zeros((len(fitted_model_dict), nbands))
    with open(flename, 'w') as f:
        for i in range(len(source_table)):
            model_mags_arr = np.array([fitted_model_dict[i][bp] for bp in bp_order]).flatten()
            ra = source_table[i]['ra']
            dec = source_table[i]['dec']
            outstr = f"{i:4d} {ra:.5f} {dec:.5f} {envelope_tally[i]:2d} {int(round(weights[i],0)):3d} "
            outstr = outstr + print_extinction(model_mags_arr + diffuse_ext_table[i] + dense_ext_table[i])
            outstr = outstr + print_extinction(diffuse_ext_table[i])
            outstr = outstr + print_extinction(dense_ext_table[i]) + "\n"
            f.write(outstr)
            final_obs_mags[i] = model_mags_arr + diffuse_ext_table[i] + dense_ext_table[i]
    return final_obs_mags


# Load bandpasses and zero-magnitude fluxes
bp_dict, bp_order = load_bandpass_dict()
zp_dict = load_zeropoint_dict()

log.info(f"Bandpasses, zeropoints, order: {bp_dict} {zp_dict} {bp_order}")

# Number of bands to output - now V, J, H, K, and IRAC 3.6
nbands = len(bp_order)

# Load/query source table in field of interest
source_table = query_source_table(serpens_main_center, offset=off_field_angle)

log.info(f"Source table has {len(source_table)} sources.")

# Load Castelli-Kurucz stellar models, then generate colors for each stellar model in dictionary
model_dict = load_ck_models(CK_modelfile)
colordict_models_base = gen_2MASS_colordict(model_dict, bp_dict, zp_dict, {key: 0. for key in bp_dict.keys()})

# Load extinction curves - one for diffuse ISM, one for dense ISM
diffuse_wav, diffuse_ext, dense_wav, dense_ext = load_ext_curves(diffuse_extcurve_file, dense_extcurve_file)

# Create dicts for conversion between A_X for the bandpasses in dictionary
filter_extinctions_chap = filter_extinction(dense_wav, dense_ext, bp_dict)
filter_extinctions_wd = filter_extinction(diffuse_wav, diffuse_ext, bp_dict)

log.info(f"Diffuse extinctions: "
         f"{[(i, j) for i, j in zip(filter_extinctions_wd.keys(),filter_extinctions_wd.values())]}")
log.info(f"Dense extinctions: "
         f"{[(i, j) for i, j in zip(filter_extinctions_chap.keys(), filter_extinctions_chap.values())]}")

# Model a envelope to approximate SMM4. This will be scaled later to approximate envelope
# sizes and densities for all envelopes in the Serpens field.
smm4_envelope = Envelope(r_turnover=400.*AU, rho_crit=2.5e13, rho_min=1.e11, alpha=1.6)

# Find the best-fit CK model for each source in the
# source table, then use that model to estimate the [V, 3.6] mags.
# Store the un-extincted model star bandpass magnitudes in a dict.
fitted_model_dict, bestfit_keys = fit_sourcetable(source_table, colordict_models_base, fit_stellar_types, default_type)

log.info(f"fitted_model_dict entry: {fitted_model_dict[1]}")

# Optional output to check on the quality of stellar type fitting
if fit_stellar_types:
    with open('bestfit_keys.txt', 'w') as f:
        f.write("ID   CK Type\n")
        for i, item in enumerate(bestfit_keys):
            f.write("{:4d}   {:5}\n".format(i, item))

# Move the source list from initial, 'offset' position to the Serpens field of interest
offset_coords, serpens_coords, serpens_table = move_source_field(source_table, offset=off_field_angle)

log.info(f"Serpens_table head: {serpens_table[:10]}")

# Load the location and power-law-scaled flux factors for SMM envelopes in Serpens field
smm_scaledflux_dict, envelope_coord_dict = load_smm_envelopes()

# Find the extinction due to foreground dust map as well as modeled envelopes
diffuse_ext_table, dense_ext_table, envelope_tally = calc_extinctions(serpens_coords, smm4_envelope,
                                                                      envelope_coord_dict, filter_extinctions_wd,
                                                                      filter_extinctions_chap)

# Set weights for use in MSA target selection within APT.
weights, pacs_img, pacs_wcs, pacs_fit = set_source_weights_pacs(serpens_coords)

# Construct the table for input to APT.
final_mags_obs = write_output_table(output_filename)


if plot_stuff:
    plt.figure()
    plt.plot(np.log10(diffuse_wav), np.log10(diffuse_ext), label='Diffuse')
    plt.plot(np.log10(dense_wav), np.log10(dense_ext), label='Dense')
    plt.title("Quality check on extinction curves.")
    plt.legend()
    plt.savefig("figures/extinction_curves.pdf", format='pdf')
    plt.clf()

    # This block plots the histogram of source weights, currently colored by number of envelopes each
    # source interacts with.
    plt.figure(figsize=(12, 10))
    plt.hist(weights[envelope_tally == 0], bins=np.arange(0, 102, 2), color='red', label='0')
    plt.hist(weights[envelope_tally == 1], bins=np.arange(0, 102, 2), color='green', alpha=0.5, label='1')
    plt.hist(weights[envelope_tally >= 2], bins=np.arange(0, 102, 2), color='blue', alpha=0.5, label='2+')
    plt.yscale('log')
    plt.title("Histogram of weights, separated by number of envelopes in LoS")
    plt.ylim(0.9, None)
    plt.grid()
    plt.legend()
    plt.savefig('figures/weight_histogram.pdf', format='pdf')
    plt.clf()

    # This block plots the histogram of extinctions, currently colored by presence of  envelopes in a
    # source line of sight.
    extbins = np.arange(6, 101)
    plt.figure(figsize=(12, 10))
    plt.hist(final_mags_obs[:, 0][weights > 2], bins=extbins, color='green', alpha=0.5,
             label=f"Num Weight > 2 : {len(dense_ext_table[:, 4][weights > 2])}")
    plt.hist(final_mags_obs[:, 0][weights < 2], bins=extbins, color='black', alpha=0.4,
             label='Unselected Field')
    plt.grid()
    plt.xlabel('A_V')
    plt.ylabel('Number of Sources')
    plt.title('Total V Band Extinction')
    plt.legend()
    plt.savefig('extinction_histogram.png', format='png')
    plt.clf()

    # This block plots the extinction curves for each envelope
    plt.figure(figsize=(10, 10))
    env_r = np.logspace(2, np.log10(smm4_envelope.r_max/AU) - 0.1, 100) * AU
    env_AJ = smm4_envelope.surface_density(env_r) / n_h_per_A_J
    env_AK = env_AJ * filter_extinctions_chap["K"] / filter_extinctions_chap["J"]
    log.info(f"Range of r, AJ, AK: {np.min(env_r/AU)} - {np.max(env_r/AU)};\n"
             f"{np.min(env_AJ)} - {np.max(env_AJ)};\n {np.min(env_AK)} - {np.max(env_AK)}\n")

    def best_fit_radius_index(curve, fitvalue):
        mincurve = (curve - fitvalue) ** 2.
        return np.argmin(mincurve)

    r_5AK = dict()
    for key in smm_scaledflux_dict.keys():
        if key == 1:
            r_5AK[key] = env_r[best_fit_radius_index(env_AK * smm_scaledflux_dict[key], 10.)]
        else:
            r_5AK[key] = env_r[best_fit_radius_index(env_AK * smm_scaledflux_dict[key], 5.)]
        plt.plot(env_r / AU, env_AK * smm_scaledflux_dict[key], label=str(key))

    plt.legend(loc='best')
    plt.xlim([1000, 14000])
    plt.ylim([0, 30])
    plt.grid()
    plt.title('K Mag Extinction of Each Envelope')
    plt.savefig('plotted_envelope_extinction.png', format='png')
    plt.clf()

    # This block plots the sources and model envelopes overlaid on the PACS 160um image.
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection=pacs_wcs)
    jy_sr_factor = 360. ** 2. / (4. * np.pi ** 2) / (8.88889e-4 ** 2)
    logfactor = np.log10(jy_sr_factor)
    pacsimg = ax.imshow(np.log10(pacs_img.data * jy_sr_factor),
                        vmin=-1.2+logfactor,
                        vmax=0.4+logfactor, origin='lower',
                        cmap=cm.cividis)

    ra_center = serpens_main_center.ra.to_value() + (1. / 100.)
    dec_center = serpens_main_center.dec.to_value() - (1. / 80.)
    fov_radius_arcmin = 3.33
    minx, miny = pacs_wcs.wcs_world2pix(ra_center - (fov_radius_arcmin / 60.),
                                        dec_center - (fov_radius_arcmin / 60.), 1)
    maxx, maxy = pacs_wcs.wcs_world2pix(ra_center + (fov_radius_arcmin / 60.),
                                        dec_center + (fov_radius_arcmin / 60.), 1)
    env_sources = envelope_tally > 0

    # For PACS160, pixel size is 3.2 arcsec. For envelope radius of r_max in m,
    # radius in arcsec is r_max/m_per_au/dist(pc)
    patchradius = smm4_envelope.r_max / AU / distance_serpens_pc / 3.2
    fixedradius = 10000. / distance_serpens_pc / 3.2

    patchset = []
    for key in envelope_coord_dict.keys():
        # patchset.append(patches.Circle(skycoord_to_pixel(envelope_coord_dict[key],pacs_wcs,0),
        # radius=r_5AK[key]/AU/distance_serpens_pc/3.2, edgecolor='white', facecolor='green', alpha=0.3))

        # Instead of using r_5AK for plotting, use uniform radius of 1e4 AU instead
        patchset.append(patches.Circle(skycoord_to_pixel(envelope_coord_dict[key], pacs_wcs, 0),
                                       radius=fixedradius, edgecolor='white', facecolor='green', alpha=0.3))
    envelope_patches = PatchCollection(patchset, match_original=True)
    ax.add_collection(envelope_patches)

    pixel_locations = np.array([skycoord_to_pixel(serpens_coords[i], pacs_wcs, 1) for i in range(len(source_table))])
    scalebar = AnchoredSizeBar(ax.transData, (10000. / distance_serpens_pc / 3.2), '10000 AU', 'upper left',
                               pad=2, sep=6, borderpad=3.5, color='white', frameon=False,
                               size_vertical=0.3, fontproperties=fontprops,  # fontsize=12
                               )
    ax.add_artist(scalebar)

    with open('params/targets_north.txt', 'r') as f:
        lines = [line for line in f]

    pixel_locations_north = []
    for i, line in enumerate(lines[2:]):
        loc = line.split('24]')[-1]
        ra, dec = loc.split('+')
        pixel_locations_north.append(skycoord_to_pixel(SkyCoord(ra.strip(), '+'+dec.strip(),
                                                                unit=(u.hourangle, u.deg)),
                                                       pacs_wcs, 1))

    pixel_locations_north = np.array(pixel_locations_north)

    with open('params/targets_south.txt', 'r') as f:
        lines = [line for line in f]

    pixel_locations_south = []
    for i, line in enumerate(lines[2:]):
        loc = line.split('24]')[-1]
        ra, dec = loc.split('+')
        pixel_locations_south.append(skycoord_to_pixel(SkyCoord(ra.strip(), '+'+dec.strip(),
                                                                unit=(u.hourangle, u.deg)),
                                                       pacs_wcs, 1))

    pixel_locations_south = np.array(pixel_locations_south)

    pts_s = ax.scatter(pixel_locations_south[:, 0], pixel_locations_south[:, 1],
                       c='red', s=15, marker="o", edgecolor='gray')
    pts_n = ax.scatter(pixel_locations_north[:, 0], pixel_locations_north[:, 1],
                       c='white', s=15, marker="o", edgecolor='gray')

    cb1 = plt.colorbar(pacsimg, ax=ax, label=r'log$_{10}($F$_{160 \mu m }$[Jy/sr])')

    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    ax.invert_xaxis()
    plt.savefig("figures/field_north_south.pdf", format='pdf', dpi=200, bbox_inches='tight')
    plt.clf()
