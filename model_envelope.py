import numpy as np
from scipy import constants as con

AU = con.au


class Envelope:
    def __init__(self, r_turnover, rho_crit, rho_min, alpha, r_arr=None):
        self.r_turnover = r_turnover
        self.rho_crit = rho_crit
        self.rho_min = rho_min
        self.alpha = alpha
        if r_arr is None:
            self.r_arr = np.logspace(2, 4.5, 300) * AU
        else:
            self.r_arr = r_arr
        self.r_max = self.find_r_max()
        self.mass = self.envelope_mass()
        self.density_arr = self.density_profile()
        self.surface_density_arr = self.surface_density()

    def density_profile(self, r=None):
        """Calculate density profile"""
        if r is not None:
            arr = r
        else:
            arr = self.r_arr
        dens_array = ((self.rho_crit * (self.r_turnover ** self.alpha)) /
                      (arr ** self.alpha + self.r_turnover ** self.alpha))
        dens_array = np.where(dens_array > self.rho_min, dens_array, 0.)
        return dens_array

    def find_r_max(self):
        return self.r_turnover * ((self.rho_crit / self.rho_min) - 1.) ** (1. / self.alpha)

    def surface_density(self, r=None):
        if r is None:
            outs = self.r_arr.copy()
        else:
            outs = np.asarray(r)
        outs = np.atleast_1d(np.where(outs < self.r_max, outs, 0.))
        for i in range(len(outs)):
            if outs[i] > 0:
                ss = np.linspace(0, np.sqrt(self.r_max ** 2. - outs[i] ** 2.), 10000)
                yy = 2. * self.rho_crit * np.array(
                    [(1. / (1. + ((s ** 2. + outs[i] ** 2.) / self.r_turnover ** 2.) ** (self.alpha / 2.)))
                     for s in ss])
                outs[i] = (np.trapz(yy, x=ss))
        return outs

    def envelope_mass(self):
        xint = np.logspace(np.log10(AU), np.log10(self.r_max), 1000)
        mass_density_arr = self.density_profile(xint)
        yy = (4. * np.pi * xint ** 2.) * mass_density_arr * 2.33 * con.m_p
        return np.trapz(yy, x=xint)

