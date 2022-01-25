import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from scipy.interpolate import interp2d
from scipy import constants as con
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pacs_file_for_interpolation = 'hpacs_25HPPJSMAPR_1831_p0041_00_v1.0_1471632179809.fits'


def load_pacs_image(pacs_filename):
    pacs = fits.open(pacs_filename)[1]
    pacs_wcs = WCS(pacs.header)

    xvec = np.arange(len(pacs.data[0, :]))
    yvec = np.arange(len(pacs.data[:, 0]))
    f = interp2d(x=xvec, y=yvec, z=pacs.data)
    return pacs, pacs_wcs, f


def find_weight_log(coords, wcs, func, minval, maxval):
    interp_fit = func
    pixel_location = skycoord_to_pixel(coords, wcs)
    pix1 = pixel_location[0]
    pix2 = pixel_location[1]
    interp_value = interp_fit(pix1, pix2)[0]
    if interp_value > maxval:
        interp_value = maxval
    if interp_value < minval:
        interp_value = minval
    weight = 10. ** (2. * (np.log10(interp_value) - np.log10(minval)) /
                     (np.log10(maxval) - np.log10(minval)))
    return weight


def test_map_coords_wcs():
    img,pacs_wcs,fit = load_pacs_image(pacs_file_for_interpolation)

    deg_per_arcsec = con.arcsec / con.degree
    serpens = SkyCoord('18h29m54s', '1d15m20.1s', frame='icrs')
    ra_origin = serpens.ra.to_value()
    dec_origin = serpens.dec.to_value()
    x_r = np.linspace(-180 * deg_per_arcsec, 180 * deg_per_arcsec, 121)
    y_r = np.linspace(-180 * deg_per_arcsec, 180 * deg_per_arcsec, 121)
    ra_list = []
    dec_list = []
    weight_list =[]
    for i in range(len(x_r)):
        for j in range(len(y_r)):
            ra = ra_origin+x_r[i]
            dec = dec_origin+y_r[j]
            coord = SkyCoord(ra, dec, frame='icrs', unit='deg')
            weight = find_weight_log(coord, pacs_wcs, fit, 0.025, 1)
            ra_list.append(ra)
            dec_list.append(dec)
            weight_list.append([weight])
    plt.subplot(projection=pacs_wcs)
    plt.scatter(np.reshape(np.array(ra_list), -1),
                np.reshape(np.array(dec_list), -1),
                c=np.log10(np.reshape(np.array(weight_list), -1)))
    plt.gca().invert_xaxis()
    plt.colorbar()
    plt.show()
