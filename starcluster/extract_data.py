import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u
from pathlib import Path


class Data:
    def __init__(self, path, *, cartesian=False):
        self.path = path
        self.cartesian = cartesian

        self.__names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        self.__dtype = np.dtype([('x', float),
                                 ('y', float),
                                 ('z', float),
                                 ('vx', float),
                                 ('vy', float),
                                 ('vz', float)])

    def read(self, outpath=None, **kwargs):
        if self.cartesian:
            return self.__open_cartesian(**kwargs)
        else:
            self.__open_gaia(outpath=outpath, **kwargs)

    def __open_cartesian(self, **kwargs):
        if 'names' in kwargs.keys():
            kwargs['names'] = self.__names

        data = np.genfromtxt(self.path, **kwargs)

        return data

    def __open_gaia(self, outpath=None, **kwargs):
        # 10.1051/0004-6361/201832964 - parallax
        # 10.1051/0004-6361/201832727 - astrometric solution
        data = np.genfromtxt(self.path)

        if 'ruwe' in kwargs.keys():
            ruwe_lim = kwargs['ruwe']
        else:
            ruwe_lim = np.inf

        data_good = data[data['ruwe'] <= ruwe_lim]  # GAIA-C3-TN-LU-LL-124-01
        data = self.__eq_to_cartesian(data_good)

        if outpath is None:
            outpath = os.getcwd()
        outpath = Path(outpath)

        np.savetxt(outpath.joinpath('gaia_galactic.txt'),
                   data,
                   header='x\ty\tz\tvx\tvy\tvz')

    def __eq_to_cartesian(self, data):
        coords = SkyCoord(frame='icrs',
                          epoc='J2015.5',
                          ra=data['ra']*u.deg,
                          dec=data['dec']*u.deg,
                          pm_ra_cosdec=data['pmra']*u.mas/u.year,
                          pm_dec=data['pmdec']*u.mas/u.year,
                          parallax=data['parallax']*u.mas,
                          radial_velocity=data['radial_velocity']*u.km/u.s)

        coords_galactic = coords.transform_to('galactic')

        data_cart = np.array([], dtype=self.__dtype)
        data_cart['x', 'y', 'z'] = coords_galactic.cartesian.xyz
        data_cart['vx', 'vy', 'vz'] = coords_galactic.velociity.d_xyz

        return data_cart
