import numpy as np

from cpnest import CPNest, model
from scipy.stats import skew

from .const import R_GAL
from .const import PAR_MIN, PAR_MAX
from .const import PAR_ERR_MIN, PAR_ERR_MAX
from .const import GAL_LONG_MIN, GAL_LONG_MAX
from .const import SAMPLES_PATH
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


class SkewDistribution:
    def __init__(self, nbins=1000):
        self.__nbins = nbins
        self.__interpolant = None

        try:
            from .r_distr import NBINS
            if NBINS != self.nbins:
                raise FileNotFoundError

        except FileNotFoundError:
            self.__interpolate(self.nbins)

        finally:
            from .r_distr import skewness
            self.__interpolant = skewness

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

        skew_samples = [[[skew(hist[m, s, l], bias=False)
                          for l in range(self.nbins)]
                         for s in range(self.nbins)]
                        for m in range(self.nbins)]

        m_array = np.array((edges[0][1:] + edges[0][:-1]) / 2)
        s_array = np.array((edges[1][1:] + edges[1][:-1]) / 2)
        l_array = np.array((edges[2][1:] + edges[2][:-1]) / 2)

        with open(SAMPLES_PATH, 'w+') as file:
            file.write("# mean, sigma, galactic longitude, skewness")
            for m in range(self.nbins):
                for s in range(self.nbins):
                    for l in range(self.nbins):
                        file.write(f'\n{m_array[m]},{s_array[s]},{l_array[l]}'
                                   f',{skew_samples[m,s,l]}')

    def __call__(self, m, s, l):
        return self.__interpolant((m, s, l))

    @property
    def nbins(self):
        return self.__nbins

