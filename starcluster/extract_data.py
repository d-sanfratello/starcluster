import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from pathlib import Path

# FIXME: do sampler
from starcluster.sampler import Sampler


class Data:
    astrometry_base_cols = ['source_id', 'ra', 'dec', 'parallax',
                            'pmra', 'pmdec',
                            'ruwe', 'ref_epoch',
                            'parallax_error', 'parallax_over_error',
                            'dr2_radial_velocity']
    astrometry_full_cols = ['source_id', 'ra', 'dec', 'parallax',
                            'pmra', 'pmdec', 'radial_velocity',
                            'ruwe', 'ref_epoch',
                            'ra_error', 'dec_error',
                            'parallax_error', 'parallax_over_error',
                            'pmra_error', 'pmdec_error', 'ra_dec_corr',
                            'ra_parallax_corr', 'ra_pmra_corr',
                            'ra_pmdec_corr', 'dec_parallax_corr',
                            'dec_pmra_corr', 'dec_pmdec_corr',
                            'parallax_pmra_corr', 'parallax_pmdec_corr',
                            'pmra_pmdec_corr', 'dr2_radial_velocity',
                            'dr2_radial_velocity_error']

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

        self.__names = ['source_id', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        self.__dtype = np.dtype([('source_id', np.int64),
                                 ('x', float),
                                 ('y', float),
                                 ('z', float),
                                 ('vx', float),
                                 ('vy', float),
                                 ('vz', float)])

    def read(self, outpath=None, *, ruwe=None):
        """
        Method of the `Data` class to read the file whose path was defined in
        the `Data.path` attribute.

        If 'Data.is_cartesian` is True, this method reads the file and
        returns a structured array. The file must be formatted with six
        columns containing the x, y and z coordinates and vx, vy and vz
        velocities. Each row corresponds to a star. The structured array has
        labels 'source_id', 'x', 'y', 'z', 'vx', 'vy', 'vz'.

        If `Data.is_cartesian` is False, this method reads the file as a gaia
        dataset, it creates an `astropy.coordinates.SkyCoord` object with
        right ascension, declination, parallax, proper motion and distances
        for each source. Frame of reference is set as 'ICRS' and epoch given
        from data. Gaia data must contain the 'ruwe' column, as it is used
        to choose good quality data, as explained in the GAIA-C3-TN-LU-LL-124-01
        document. It must, also, contain the column 'ref_epoch', as it is
        used to convert to galactic coordinates. Any row containing missing
        data is deleted before conversion into galactic cartesian
        coordinates. Finally, it must contain the `source_id` column as it is
        used to identify stars.

        Parameters
        ----------
        outpath:
            None, 'str' or 'Path-like'. The path of the output file to save
            cartesian coordinates data if a gaia dataset is read. If None,
            it saves the data in the current working directory in a file
            named 'gaia_galactic.txt'. (Optional)
        ruwe:
            If `Data.is_cartesian` is False it is the RUWE limit to accept
            good data. See GAIA-C3-TN-LU-LL-124-01 document for further
            information. If `ruwe` is None, the limit is set to np.inf,
            accepting all data.

        Returns
        -------
        Numpy structured array:
            If `Data.is_cartesian` is True, it returns a numpy structured
            array with fields 'source_id', 'x', 'y', 'z', 'vx', 'vy' and 'vz'
            containing
            the galactic cartesian coordinates for each star.
        """
        if self.is_cartesian:
            return self.__open_cartesian()
        else:
            self.__open_gaia(outpath=outpath, ruwe=ruwe)

    def __open_cartesian(self):
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        return data

    def __open_gaia(self, outpath=None, *, ruwe=None):
        # 10.1051/0004-6361/201832964 - parallax
        # 10.1051/0004-6361/201832727 - astrometric solution
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        # Selecting data based on missing parameters
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        # Selecting data based on RUWE (GAIA-C3-TN-LU-LL-124-01)
        if ruwe is None:
            ruwe = np.inf
        data_good = data[data['ruwe'] <= ruwe]

        data = self.__eq_to_cartesian(data_good)

        if outpath is None:
            outpath = os.getcwd()
            outpath = Path(outpath).joinpath('gaia_galactic.txt')
        else:
            outpath = Path(outpath)

        np.savetxt(outpath,
                   data,
                   header='source_id,x,y,z,vx,vy,vz',
                   delimiter=',')

    def __eq_to_cartesian(self, data):
        # FIXME: Complete with MCMC integration for galactic coordinates
        #  and conversion into cartesian galactic coordinates. See notes
        #  at pages a (for covariance matrix shape), e.2 (for (RA,
        #  DEC) as functions of (l, b), f (for posterior distribution
        #  over galactic spherical coordinates) and g (for (pmRA, pmDEC)
        #  as functions of (l, b, pml, pmb).
        source_id = []
        x = []
        y = []
        z = []
        vx = []
        vy = []
        vz = []

        for s in range(data['source_id'].shape):
            ra, dec, par, pmra, pmdec = data['ra'][s], \
                                        data['dec'][s], \
                                        data['parallax'][s], \
                                        data['pmra'][s], \
                                        data['pmdec'][s]

            s_ra = data['ra_error'][s] / np.cos(np.deg2rad(dec))
            s_dec = data['dec_error'][s]
            s_par = data['parallax_error'][s]
            s_pmra = data['prma_error'][s]
            s_pmdec = data['pmdec_error'][s]
            cov_ra_dec = data['ra_dec_corr'][s] * s_ra * s_dec
            cov_ra_par = data['ra_parallax_corr'][s] * s_ra * s_par
            cov_ra_pmra = data['ra_pmra_corr'][s] * s_ra * s_pmra
            cov_ra_pmdec = data['ra_pmdec_coor'][s] * s_ra * s_pmdec
            cov_dec_par = data['dec_parallax_corr'][s] * s_dec * s_par
            cov_dec_pmra = data['dec_pmra_corr'][s] * s_dec * s_pmra
            cov_dec_pmdec = data['dec_pmdec_corr'][s] * s_dec * s_pmdec
            cov_par_pmra = data['parallax_pmra_corr'][s] * s_par * s_pmra
            cov_par_pmdec = data['parallax_pmdec_corr'][s] * s_par * s_pmdec
            cov_pmra_pmdec = data['pmra_pmdec_corr'][s] * s_pmra * s_pmdec

            sigma = [[s_ra**2, cov_ra_dec, cov_ra_par, cov_ra_pmra, cov_ra_pmdec],
                     [cov_ra_dec, s_dec**2, cov_dec_par, cov_dec_pmra, cov_dec_pmdec],
                     [cov_ra_par, cov_dec_par, s_par**2, cov_par_pmra, cov_par_pmdec],
                     [cov_ra_pmra, cov_dec_pmra, cov_par_pmra, s_pmra**2, cov_pmra_pmdec],
                     [cov_ra_pmdec, cov_dec_pmdec, cov_par_pmdec, cov_pmra_pmdec, s_pmdec**2]]

            sigma = np.array(sigma)

            vrad = data['dr2_radial_velocity'][s]
            s_vrad = data['dr2_radial_velocity_error'][s]

            sampler = Sampler(ra, dec, pmra, pmdec, par, sigma,
                              vrad, s_vrad)
            galactic = sampler.run()

            cartesian = self.__gal_to_cartesian(galactic)

            source_id.append(data['source_id'][s])
            x.append(cartesian['x'])
            y.append(cartesian['y'])
            z.append(cartesian['z'])
            vx.append(cartesian['vx'])
            vy.append(cartesian['vy'])
            vz.append(cartesian['vz'])

        data_cart = np.zeros(len(data), dtype=self.__dtype)
        data_cart['source_id'] = source_id
        data_cart['x'] = x
        data_cart['y'] = y
        data_cart['z'] = z
        data_cart['vx'] = vx
        data_cart['vy'] = vy
        data_cart['vz'] = vz

        return data_cart

    def __gal_to_cartesian(self, gal):
        x = np.cos(gal['b']) * np.cos(gal['l']) * gal['r']
        y = np.cos(gal['b']) * np.sin(gal['l']) * gal['r']
        z = np.sin(gal['b']) * gal['r']

        # FIXME: Evaluate cartesian velocities (see how you did for for
        #  equatorial cartesian coordinates)

    def __dec(self, l, b):
        b = np.deg2rad(b)
        l = np.deg2rad(l)

        sin_dec = np.sin(self.dec_G) * np.sin(b)
        sin_dec += (np.cos(self.dec_G) * np.cos(b) * np.cos(self.theta - l))

        return np.arcsin(sin_dec)

    def __ra(self, l, b):
        b = np.deg2rad(b)
        l = np.deg2rad(l)

        f1 = np.cos(b) * np.sin(self.theta - l)
        f2 = np.cos(self.dec_G * np.sin(b))
        f2 -= (np.sin(self.dec_G) * np.cos(b) * np.cos(self.theta - l))

        return self.ra_G + np.arctan2(f1, f2)

    def __pmdec(self, l, b, pml, pmb):
        b = np.deg2rad(b)
        l = np.deg2rad(l)

        pmb_term = np.sin(self.dec_G) * np.cos(b)
        pmb_term -= np.cos(self.dec_G) * np.sin(b) * np.cos(self.theta - l)
        pmb_term *= pmb

        pml_term = np.cos(self.dec_G) * np.sin(self.theta - l)
        pml_term *= pml

        return (pml_term + pmb_term) / (1 - self.__F(l, b)**2)

    def __pmra(self, l, b, pml, pmb):
        b = np.deg2rad(b)
        l = np.deg2rad(l)

        pmb_term = -np.sin(b)*np.sin(self.theta - l) * np.__H(l, b)
        pmb_term -= self.__G(l, b) * (np.cos(self.dec_G) * np.cos(b) +
                                      np.sin(self.dec_G) * np.sin(b) *
                                      np.cos(self.theta - l))
        pmb_term *= pmb

        pml_term = self.__H(l, b) * (-np.cos(b) * np.cos(self.theta - l))
        pml_term -= self.__G(l, b) * (-np.sin(self.dec_G) * np.cos(b) *
                                      np.sin(self.theta - l))
        pml_term *= pml

        ra = (pml_term + pmb_term) / (self.__G(l, b)**2 + self.__H(l, b)**2)
        return ra * np.cos(self.__dec(l, b))

    def __F(self, l, b):
        f = np.sin(self.dec_G) * np.sin(b)
        f += np.cos(self.dec_G) * np.cos(b) * np.cos(self.theta - l)

        return f

    def __G(self, l, b):
        return np.cos(b) * np.sin(self.theta - self.l)

    def __H(self, l, b):
        h = np.cos(self.dec_G) * np.sin(b)
        h -= np.sin(self.dec_G) * np.cos(b) * np.cos(self.theta - l)

        return h

    def __rad2as(self, rad):
        deg = rad * 180 / np.pi
        arcsec = deg * 3600

        return arcsec

    @property
    def dec_G(self):
        return np.deg2rad(27.12825)

    @property
    def ra_G(self):
        return np.deg2rad(192.85948)

    @property
    def theta(self):
        return np.deg2rad(123.932)
