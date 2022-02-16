import numpy as np
import os

from pathlib import Path

from .parallax_limit import parallax_limit


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
        long = np.linspace(0, 2*np.pi, num=nsteps, endpoint=False)
        lim_dist = parallax_limit(long)



    @property
    def nsteps(self):
        return self.__nsteps

