import os

import corner
import matplotlib.pyplot as plt
import numpy as np

from corner import corner as crn
from figaro.mixture import DPGMM
from figaro.utils import get_priors
from pathlib import Path

from .extract_data import EquatorialData


class CornerPlot:
    __keys = ['ra', 'dec', 'l', 'b', 'plx',
              'pmra', 'pmdec', 'v_rad']
    __mags = ['g_mag', 'bp_mag', 'rp_mag',
              'bp_rp', 'bp_g', 'g_rp']

    def __init__(self, *,
                 density, dataset,
                 expected=None,
                 mag='g_mag', c_index='bp_rp'):

        if expected is not None:
            for k in self.__keys:
                setattr(self, k, expected[k])

            if mag in self.__mags and mag in expected.keys():
                self.mag = expected[mag]
            else:
                self.mag = None

            if c_index in self.__mags and c_index in expected.keys():
                self.c_index = expected[c_index]
            else:
                self.c_index = None

            self.__has_expected = True
        else:
            self.__has_expected = False

        self.density = density
        self.dataset = dataset

        mag_name = mag[:-4].upper()
        self.mag_name = f'${mag_name}\,[mag]$'

        if c_index == 'bp_rp':
            self.c_index_name = r'$G_{BP} - G_{RP}\,[mag]$'
        elif c_index == 'bp_g':
            self.c_index_name = r'$G_{BP} - G\,[mag]$'
        elif c_index == 'g_rp':
            self.c_index_name = r'$G - G_{RP}\,[mag]$'

    def __call__(self, sampling_size, *, plot_title=None):
        density_samples = self.density.rvs(sampling_size)

        c = crn(density_samples,
                bins=int(np.sqrt(sampling_size)),
                color='dodgerblue',
                title=plot_title,
                labels=[r'$l\,[deg]$',
                        r'$b\,[deg]$',
                        r'$plx\,[mas]$',
                        r'$\mu_{l*}\,[mas\,yr^{-1}]$',
                        r'$\mu_b\,[mas\,yr^{-1}]$',
                        r'$v_{rad}\,[km\,s^{-1}]$',
                        f'{self.mag_name}',
                        f'{self.c_index_name}'],
                hist_kwargs={'density': True,
                             'label': 'DPGMM'})

        c.suptitle(plot_title, size=20)

        if self.__has_expected:
            ra_rad = np.deg2rad(self.ra)
            dec_rad = np.deg2rad(self.dec)
            l_rad = np.deg2rad(self.l)
            b_rad = np.deg2rad(self.b)

            p_icrs = np.array([-np.sin(ra_rad),
                               np.cos(ra_rad),
                               0])
            q_icrs = np.array([-np.cos(ra_rad) * np.sin(dec_rad),
                               -np.sin(ra_rad) * np.sin(dec_rad),
                               np.cos(dec_rad)])
            p_gal = np.array([-np.sin(l_rad),
                              np.cos(l_rad),
                              0])
            q_gal = np.array([-np.cos(l_rad) * np.sin(b_rad),
                              -np.sin(l_rad) * np.sin(b_rad),
                              np.cos(b_rad)])

            mu_icrs = p_icrs * self.pmra + q_icrs * self.pmdec
            mu_gal = self.dataset.A_G_inv.dot(mu_icrs)

            pml_star = np.dot(p_gal, mu_gal)
            pmb = np.dot(q_gal, mu_gal)

            data_lines = np.array([
                self.l, self.b, self.plx,
                pml_star, pmb, self.v_rad,
                self.mag, self.c_index
            ])

            corner.overplot_lines(c, data_lines,
                                  color="C1",
                                  linewidth=0.5,
                                  label='Expected')
            corner.overplot_points(c, data_lines[None],
                                   marker=".",
                                   color="C1")

        leg = plt.legend(loc=0, frameon=False, fontsize=15,
                         bbox_to_anchor=(1 - 0.05, 3))

        return c


def dpgmm(
        path, outpath=None,
        *, convert=True, std=None, epsilon=1e-3,
        mag='g_mag', c_index='bp_rp'):
    dataset = EquatorialData(path=path, convert=convert)

    if convert:
        if outpath is None:
            outpath = Path(str(path).strip('.csv') + '.txt')
        dataset.save_dataset(outpath=outpath)

    l = dataset('l')
    b = dataset('b')
    plx = dataset('plx')
    pml = dataset('pml_star')
    pmb = dataset('pmb')
    v_rad = dataset('v_rad')
    mag_ds = dataset(mag)
    c_index_ds = dataset(c_index)

    bounds = [[l.min() - epsilon, l.max() + epsilon],
              [b.min() - epsilon, b.max() + epsilon],
              [plx.min() - epsilon, plx.max() + epsilon],
              [pml.min() - epsilon, pml.max() + epsilon],
              [pmb.min() - epsilon, pmb.max() + epsilon],
              [v_rad.min() - epsilon, v_rad.max() + epsilon],
              [mag_ds.min() - epsilon, mag_ds.max() + epsilon],
              [c_index_ds.min() - epsilon, c_index_ds.max() + epsilon]]

    samples = dataset.as_array(mag=mag, c_index=c_index)

    if std is None:
        prior = get_priors(bounds, samples)
    else:
        prior = get_priors(bounds, std=std)

    mix = DPGMM(bounds=bounds,
                prior_pars=prior)

    density = mix.density_from_samples(samples)

    return dataset, bounds, prior, mix, density
