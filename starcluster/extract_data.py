import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from pathlib import Path


class Data:
    def __init__(self, path, *, is_cartesian=False):
        """
        Class to open an existing file containing astrometric data.

        If `is_cartesian` is True, this class will read a txt file containing
        data in galactic cartesian coordinates. If `is_cartesian` is False,
        it reads data from a gaia csv file and uses astropy to convert gaia
        data to galactic cartesian coordinates, gaia data must contain the
        `ruwe` column, to select good quality data, as expressed in
        GAIA-C3-TN-LU-LL-124-01 document.

        Data converted from a gaia csv file is then saved into a new file in
        galactic cartesian coordinates.

        Parameters
        ----------
        path:
            'str' or 'Path-like'. The path to the file containing data. For
            differences between datasets in galactic cartesian components and as
            a gaia dataset, read below.
        is_cartesian:
            'bool'. If True, the path given in `path` contains data already
            converted into galactic cartesian components. See `Data.read` for
            further information.
        """
        self.path = Path(path)
        self.is_cartesian = is_cartesian

        self.__names = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        self.__dtype = np.dtype([('x', float),
                                 ('y', float),
                                 ('z', float),
                                 ('vx', float),
                                 ('vy', float),
                                 ('vz', float)])

    def read(self, outpath=None, **kwargs):
        """
        Method of the `Data` class to read the file whose path was defined in
        the `Data.path` attribute.

        If 'Data.is_cartesian` is True, this method reads the file and
        returns a structured array. The file must be formatted with six
        columns containing the x, y and z coordinates and vx, vy and vz
        velocities. Each row corresponds to a star. The structured array has
        labels 'x', 'y', 'z', 'vx', 'vy', 'vz'.

        If `Data.is_cartesian` is False, this method reads the file as a gaia
        dataset, it creates an `astropy.coordinates.SkyCoord` object with
        right ascension, declination, parallax, proper motion and distances
        for each source. Frame of reference is set as 'ICRS' and epoch J2015.5.
        Gaia data must contain the 'ruwe' column, as it is used to choose
        good quality data, as explained in the GAIA-C3-TN-LU-LL-124-01 document.

        Parameters
        ----------
        outpath:
            None, 'str' or 'Path-like'. The path of the output file to save
            cartesian coordinates data if a gaia dataset is read. If None,
            it saves the data in the current working directory in a file
            named 'gaia_galactic.txt'. (Optional)
        kwargs:
            It `Data.is_cartesian` is True, it is any keyword argument
            np.genfromtxt accepts. In this case, however, `names` keyword
            argument is overwritten to look for the six coordinates 'x', 'y',
            'z', 'vx', 'vy', 'vz'.
            If `Data.is_cartesian` is False it is `ruwe` and is the RUWE
            limit to accept good data. See GAIA-C3-TN-LU-LL-124-01 document
            for further information. If no `ruwe` is passed in kwargs,
            the limit is set to np.inf, accepting all data.

        Returns
        -------
        Numpy structured array:
            If `Data.is_cartesian` is True, it returns a numpy structured
            array with fields 'x', 'y', 'z', 'vx', 'vy' and 'vz' containing
            the galactic cartesian coordinates for each star.
        """
        if self.is_cartesian:
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
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True)

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
        parallax = data['parallax'] * u.mas
        distance = parallax.to(u.pc, equivalencies=u.parallax())

        epochs = Time(data['ref_epoch'], format='jyear')

        coords = SkyCoord(frame='icrs',
                          equinox=epochs,
                          ra=data['ra']*u.deg,
                          dec=data['dec']*u.deg,
                          pm_ra_cosdec=data['pmra']*u.mas/u.year,
                          pm_dec=data['pmdec']*u.mas/u.year,
                          distance=distance,
                          radial_velocity=data['radial_velocity']*u.km/u.s)

        coords_galactic = coords.transform_to('galactic')

        data_cart = np.zeros(len(data), dtype=self.__dtype)
        data_cart['x'] = coords_galactic.cartesian.x.value
        data_cart['y'] = coords_galactic.cartesian.y.value
        data_cart['z'] = coords_galactic.cartesian.z.value
        data_cart['vx'] = coords_galactic.velocity.d_x.value
        data_cart['vy'] = coords_galactic.velocity.d_y.value
        data_cart['vz'] = coords_galactic.velocity.d_z.value

        return data_cart
