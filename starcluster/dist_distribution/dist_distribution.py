import corner
import numpy as np
import os

from cpnest import CPNest, model
from pathlib import Path

from .const import R_GAL
from .parallax_limit import parallax_limit


class Distribution(model.Model):
    def __init__(self):
        self.names = ['r', 'mean_par', 'rel_sigma_par', 'l']
        self.bounds = [[0, R_GAL],
                       [1e-6, 1],
                       [1e-10, 0.5],
                       [0, 2*np.pi]]

    def log_likelihood(self, param):
        r = param['r']
        m = param['mean_par']
        s = m * param['rel_sigma_par']

        log_L = -2 * np.log(r)
        log_L -= 1/(2*s**2) * (1/r - m)**2

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
    def __init__(self, nsteps=100):
        self.__nsteps = nsteps

        try:
            from .r_distr import NSTEPS
            if NSTEPS != self.nsteps:
                raise ImportError

        except ImportError:
            self.__interpolate(self.nsteps)

        finally:
            from .r_distr import skew_eval

    def __interpolate(self, nsteps):
        joint_distribution = Distribution()

        job = CPNest(joint_distribution, verbose=0, nlive=1000,
                     maxmcmc=1500, nnest=4,
                     nensemble=4,
                     seed=1234)

        job.run()

        post = job.posterior_samples.ravel()

        # FIXME: Can do three integrals: one for E[r], one for E[r^2] and one
        #  for E[r^3], to be interpolated with mu and sigma of the parallax.
        #  This allows for an interpolation of the skewness coefficient
        #  mu-sigma dependent. Need to rework the above integral to make them
        #  three.

    @property
    def nsteps(self):
        return self.__nsteps

