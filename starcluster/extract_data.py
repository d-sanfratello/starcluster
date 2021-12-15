import numpy as np
import os

from pathlib import Path


class Data:
    def __init__(self, path, *, cartesian=False):
        self.path = path
        self.cartesian = cartesian

        self.__names = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    def read(self, **kwargs):
        if self.cartesian:
            return self.__open_cartesian(**kwargs)
        else:
            self.__open_gaia()

    def __open_cartesian(self, **kwargs):
        if 'names' in kwargs.keys():
            kwargs['names'] = self.__names

        data = np.genfromtxt(self.path, **kwargs)

        return data

    def __open_gaia(self):
        pass
