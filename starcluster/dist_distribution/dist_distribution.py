import corner
import numpy as np
import os

from cpnest import CPNest, model
from pathlib import Path
from scipy.stats import skew

from .const import R_GAL
from .const import PAR_MIN, PAR_MAX
from .const import PAR_ERR_MIN, PAR_ERR_MAX
from .const import GAL_LONG_MIN, GAL_LONG_MAX
from .parallax_limit import parallax_limit


class Distribution(model.Model):
    def __init__(self):
        self.names = ['mean_par', 'sigma_par', 'l', 'r']
        self.bounds = [[PAR_MIN, PAR_MAX],  # mas
                       [PAR_ERR_MIN, PAR_ERR_MAX],  # mas
                       [GAL_LONG_MIN, GAL_LONG_MAX],  # rad, galactic longitude
                       [0, R_GAL]]  # kpc

    def log_likelihood(self, param):
        r = param['r']
        m = param['mean_par']
        s = param['rel_sigma_par']

        log_L = -2 * np.log(r) - 1/(2 * s**2) * (1/r - m)**2

        return log_L

    def log_prior(self, param):
        log_p = super(Distribution, self).log_prior(param)

        r = param['r']
        l = param['l']
        L = parallax_limit(l)

        if r > L:
            log_p -= np.inf

        if np.isfinite(log_p):
            log_p = 0

        return log_p


class DistDistribution:
    def __init__(self, nbins=100):
        self.__nbins = nbins

        try:
            from .r_distr import NBINS
            if NBINS != self.nbins:
                raise ImportError

        except ImportError:
            self.__interpolate(self.nbins)

        finally:
            from .r_distr import skewness

    def __interpolate(self):
        joint_distribution = Distribution()

        job = CPNest(joint_distribution, verbose=0, nlive=1000,
                     maxmcmc=1500, nnest=4,
                     nensemble=4,
                     seed=1234)

        job.run()

        post = job.posterior_samples.ravel()

        hist, edges = np.histogramdd(sample=post,
                                     bins=self.nbins)



        # FIXME: Can do three integrals: one for E[r], one for E[r^2] and one
        #  for E[r^3], to be interpolated with mu and sigma of the parallax.
        #  This allows for an interpolation of the skewness coefficient
        #  mu-sigma dependent. Need to rework the above integral to make them
        #  three.

    @property
    def nbins(self):
        return self.__nbins

