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
              'pmra', 'pmdec', 'v_rad',
              'g_mag', 'bp_rp']

    def __init__(self, *,
                 density, dataset,
                 expected=None):

        if expected is not None:
            for k in self.__keys:
                setattr(self, k, expected[k])
            self.__has_expected = True
        else:
            self.__has_expected = False

        self.density = density
        self.dataset = dataset

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
                        r'$G\,[mag]$',
                        r'$G_{BP} - G_{RP}\,[mag]$'],
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
                self.g_mag, self.bp_rp
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


def dpgmm(path, outpath=None, *, convert=True, std=None, epsilon=1e-3):
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
    g_mag = dataset('g_mag')
    bp_rp = dataset('bp_rp')

    bounds = [[l.min() - epsilon, l.max() + epsilon],
              [b.min() - epsilon, b.max() + epsilon],
              [plx.min() - epsilon, plx.max() + epsilon],
              [pml.min() - epsilon, pml.max() + epsilon],
              [pmb.min() - epsilon, pmb.max() + epsilon],
              [v_rad.min() - epsilon, v_rad.max() + epsilon],
              [g_mag.min() - epsilon, g_mag.max() + epsilon],
              [bp_rp.min() - epsilon, bp_rp.max() + epsilon]]

    samples = dataset.as_array()

    if std is None:
        prior = get_priors(bounds, samples)
    else:
        prior = get_priors(bounds, std=std)

    mix = DPGMM(bounds=bounds,
                prior_pars=prior)

    density = mix.density_from_samples(samples)

    return dataset, bounds, prior, mix, density
