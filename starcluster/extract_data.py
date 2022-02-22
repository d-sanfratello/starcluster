import numpy as np
import os

from pathlib import Path

from .const import PC2KM, YR2S


class Data:
    astrometry_cols = ['source_id', 'ra', 'dec', 'parallax',
                       'pmra', 'pmdec', 'dr2_radial_velocity',
                       'ruwe', 'parallax_over_error',
                       'ref_epoch']
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

        If `is_cartesian` is `True`, this class will read a txt file containing
        data in galactic cartesian coordinates. If `is_cartesian` is `False`,
        it reads data from a Gaia csv file and uses the method from
        Hobbs et al., 2021, Ch.4 to convert Gaia data to galactic cartesian
        coordinates. Data from Gaia data must contain the `ruwe` column, to
        select good quality data, and the `parallax_over_error` column.

        Data converted from a gaia csv file is then saved into a new file in
        galactic cartesian coordinates.

        Parameters
        ----------
        path:
            'str' or 'Path-like'. The path to the file containing data. For
            differences between datasets in galactic cartesian components and as
            a gaia dataset, read below.
        is_cartesian:
            'bool'. If `True`, the path given in `path` contains data already
            converted into galactic cartesian components. See `Data.read` for
            further information.
        """
        self.path = Path(path)
        self.is_cartesian = is_cartesian

        self.__names = ['source_id',
                        'x', 'y', 'z', 'vx', 'vy', 'vz',
                        'ref_epoch']
        self.dtype = np.dtype([('source_id', np.int64),
                               ('x', float),
                               ('y', float),
                               ('z', float),
                               ('vx', float),
                               ('vy', float),
                               ('vz', float),
                               ('ref_epoch', float)])
        self.eq_dtype = np.dtype([('source_id', np.int64),
                                  ('ra', float),
                                  ('dec', float),
                                  ('parallax', float),
                                  ('pmra', float),
                                  ('pmdec', float),
                                  ('dr2_radial_velocity', float),
                                  ('ruwe', float),
                                  ('ref_epoch', float),
                                  ('parallax_over_error', float)])

        self.__A_G_inv = np.array([
            [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
            [0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
            [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669]])

    def read(self, outpath=None, *, ruwe=None, parallax_over_error=None):
        """
        Method of the `Data` class to read the file whose path was defined in
        the `Data.path` attribute.

        If 'Data.is_cartesian` is `True`, this method reads the file and
        returns a numpy structured array. The file must be formatted with eight
        columns containing the x, y and z coordinates and vx, vy and vz
        velocities, the remaining two columns are 'source_id' and 'ref_epoch'.
        Each row corresponds to a star. The structured array has labels
        'source_id', 'x', 'y', 'z', 'vx', 'vy', 'vz' and 'ref_epoch'.

        Spatial coordinates are expressed in kpc, velocities in km/s.

        If `Data.is_cartesian` is `False`, this method reads the file as a Gaia
        dataset, converting the equatorial coordinates of RA, DEC, parallax,
        proper motion and radial velocity into galactic cartesian
        coordinates, following Hobbs et al., 2021, Ch.4. If `ruwe` and
        `parallax_over_error` parameters are not `None` (independently),
        stars in the Gaia dataset are filtered, keeping only stars with ruwe
        <= `ruwe` and parallax_over_error >= `parallax_over_error`.

        Hence, Gaia csv dasatet must contain both the `ruwe` and the
        `parallax_over_error` columns. It must, also, contain the `ref_epoch`
        column. Any row containing missing data (imported as `Nan`s) is deleted
        before conversion into galactic cartesian coordinates. Finally, it must
        contain the `source_id` column as it is used to identify stars.

        Parameters
        ----------
        outpath:
            None, 'str' or 'Path-like'. The path of the output file to save
            cartesian coordinates data if a gaia dataset is read. If `None`,
            it saves the data in the current working directory in a file
            named 'gaia_galactic.txt'. (Optional if `is_cartesian` in class
            initialization was `False`, otherwise this is ignored)
        ruwe:
            If `Data.is_cartesian` is `False` it is the RUWE limit to accept
            good data. See GAIA-C3-TN-LU-LL-124-01 document for further
            information. If `ruwe` is `None`, the limit is set to np.inf,
            accepting all data. (Optional, if `is_cartesian` in class
            initialization was `True` this is ignored)
        parallax_over_error:
            The value of the parallax divided by its error. Its the inverse
            of the relative error and is used to select data which has an
            almost symmetrical probability distribution over distance,
            so that 1 / parallax is a good approximation for the mode of the
            posterior of the distance. If `parallax_over_error` is `None`,
            the limit is set to 0, accepting alla data. (Optional, if
            `is_cartesian` in class initialization was `True` this is ignored)

        Returns
        -------
        numpy structured array:
            If `Data.is_cartesian` is `True`, it returns a numpy structured
            array with fields 'source_id', 'x', 'y', 'z', 'vx', 'vy',
            'vz' and 'ref_epoch', containing the galactic cartesian coordinates
            for each star.
        """
        if self.is_cartesian:
            return self.__open_cartesian()
        else:
            self.__open_gaia(outpath=outpath,
                             ruwe=ruwe, parallax_over_error=parallax_over_error)

    def __open_cartesian(self):
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        new_data = np.zeros(data['source_id'].shape, dtype=self.dtype)
        for name in new_data.dtype.names:
            if name != 'source_id':
                new_data[name] = data[name]
            else:
                new_data['source_id'] = data['source_id'].astype(np.int64)

        return new_data

    def __open_gaia(self, outpath=None, *, ruwe=None, parallax_over_error=None):
        # 10.1051/0004-6361/201832964 - parallax
        # 10.1051/0004-6361/201832727 - astrometric solution
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        new_data = np.zeros(data['source_id'].shape, dtype=self.eq_dtype)
        for name in new_data.dtype.names:
            if name != 'source_id':
                new_data[name] = data[name]
            else:
                new_data['source_id'] = data['source_id'].astype(np.int64)

        data = new_data

        # Selecting data based on missing parameters
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        # Selecting data based on RUWE (GAIA-C3-TN-LU-LL-124-01)
        if ruwe is None:
            ruwe = np.inf
        data_good = data[data['ruwe'] <= ruwe]

        if parallax_over_error is None:
            parallax_over_error = 0.
        data_good = data_good[data_good['parallax_over_error'] >=
                              parallax_over_error]

        data = self.__eq_to_galcartesian(data_good)

        if outpath is None:
            outpath = os.getcwd()
            outpath = Path(outpath).joinpath('gaia_galactic.txt')
        else:
            outpath = Path(outpath)

        np.savetxt(outpath,
                   data,
                   header='source_id,x,y,z,vx,vy,vz,ref_epoch',
                   delimiter=',')

    def __eq_to_galcartesian(self, data):
        source_id = []
        x = []
        y = []
        z = []
        vx = []
        vy = []
        vz = []
        ref_epoch = []

        for s in range(data['source_id'].shape[0]):
            equatorial = data[:][s]
            galactic_cartesian = self.__eq_to_galactic(equatorial)

            source_id.append(data['source_id'][s])
            x.append(galactic_cartesian['x'])
            y.append(galactic_cartesian['y'])
            z.append(galactic_cartesian['z'])
            vx.append(galactic_cartesian['vx'])
            vy.append(galactic_cartesian['vy'])
            vz.append(galactic_cartesian['vz'])
            ref_epoch.append(data['ref_epoch'][s])

        data_cart = np.zeros(len(data), dtype=self.dtype)
        data_cart['source_id'] = source_id
        data_cart['x'] = x
        data_cart['y'] = y
        data_cart['z'] = z
        data_cart['vx'] = vx
        data_cart['vy'] = vy
        data_cart['vz'] = vz
        data_cart['ref_epoch'] = ref_epoch

        return data_cart

    def __eq_to_galactic(self, eq):
        ra = np.deg2rad(eq['ra'])
        dec = np.deg2rad(eq['dec'])
        pmra = self.__masyr_to_kms(eq['pmra'], eq['parallax'])
        pmdec = self.__masyr_to_kms(eq['pmdec'], eq['parallax'])

        pos_icrs = np.array([np.cos(dec)*np.cos(ra)/eq['parallax'],
                             np.cos(dec)*np.sin(ra)/eq['parallax'],
                             np.sin(dec) / eq['parallax']])
        pos_gal = self.A_G_inv.dot(pos_icrs)

        p_icrs = np.array([-np.sin(ra),
                           np.cos(ra),
                           0])
        q_icrs = np.array([-np.cos(ra) * np.sin(dec),
                           -np.sin(ra) * np.sin(dec),
                           np.cos(dec)])
        r_icrs = np.cross(p_icrs, q_icrs)

        mu_icrs = p_icrs * pmra + q_icrs * pmdec + \
            r_icrs * eq['dr2_radial_velocity']
        mu_gal = self.A_G_inv.dot(mu_icrs)

        cartesian_data = np.array([(eq['source_id'],
                                    pos_gal[0], pos_gal[1], pos_gal[2],
                                    mu_gal[0], mu_gal[1], mu_gal[2],
                                    eq['ref_epoch'])], dtype=self.dtype)

        return cartesian_data

    @staticmethod
    def __masyr_to_kms(pm, parallax):
        pm_asyr = pm * 1e-3
        pm_degyr = pm_asyr / 3600
        pm_radyr = np.deg2rad(pm_degyr)
        pm_rads = pm_radyr / YR2S

        parallax_pc = 1 / (parallax * 1e-3)
        parallax_km = parallax_pc * PC2KM

        pm_kms = pm_rads * parallax_km

        return pm_kms

    @property
    def A_G_inv(self):
        return self.__A_G_inv
