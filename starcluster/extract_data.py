import numpy as np
import os

from pathlib import Path


class Data:
    def __init__(self, path, *, cartesian=False):
        self.path = path
        self.cartesian = cartesian

        self.__names = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    def read(self, save=True, **kwargs):
        if self.cartesian:
            return self.__open_cartesian(**kwargs)
        else:
            self.__open_gaia(save=save)

    def __open_cartesian(self, **kwargs):
        if 'names' in kwargs.keys():
            kwargs['names'] = self.__names

        data = np.genfromtxt(self.path, **kwargs)

        return data

    def __open_gaia(self, save=True):
        # 10.1051/0004-6361/201832964
        pass
