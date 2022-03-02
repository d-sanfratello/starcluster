import numpy as np
import os

from pathlib import Path

from .const import PC2KM, YR2S, PC2KPC


class Data:
    astrometry_cols = ['source_id', 'ra', 'dec', 'parallax',
                       'pmra', 'pmdec', 'dr2_radial_velocity',
                       'ref_epoch']
    dist_cols = ['id',
                 'r_med_geo', 'r_lo_geo', 'r_hi_geo',
                 'r_med_photogeo', 'r_lo_photogeo', 'r_ho_photogeo',
                 'flag']

    def __init__(self, path, *, dist_path=None, is_cartesian=False):
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
        dist_path:
            'str' or 'Path-like'. The path to the file containing distance
            data. Default `None`.
        is_cartesian:
            'bool'. If `True`, the path given in `path` contains data already
            converted into galactic cartesian components. See `Data.read` for
            further information. Default `False`
        """
        self.path = Path(path)
        self.dist_path = None
        if dist_path is not None:
            self.dist_path = Path(dist_path)
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
                                  ('ref_epoch', float)])
        self.dist_dtype = np.dtype([('id', np.int64),
                                    ('r_med_geo', float),
                                    ('r_lo_geo', float),
                                    ('r_ho_geo', float),
                                    ('r_med_photogeo', float),
                                    ('r_lo_photogeo', float),
                                    ('r_hi_photogeo', float),
                                    ('flag', int)])

        # The matrix to convert from equatorial cartesian coordinates to
        # galactic cartesian components. See Hobbs et al., 2021, Ch.4
        self.__A_G_inv = np.array([
            [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
            [0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
            [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669]])

    def read(self, outpath=None):
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
        dataset, converting the equatorial coordinates of RA, DEC, proper motion
        and radial velocity into galactic cartesian coordinates,
        following Hobbs et al., 2021, Ch.4. Distance data for each source is
        derived from the catalogue by Bailer-Jones et al. (2021)
        (2021AJ....161..147B).

        Given that in the work by Bayler-Jones et al. (2021) their
        "photogeometric" distance is usually more precise, this measure is
        used, when available, while the "geometric" distance is used otherwise.

        Gaia csv dasatet must contain the `ref_epoch` column. Any row containing
        missing data (imported as `Nan`s) is deleted before conversion into
        galactic cartesian coordinates. Finally, it must contain the `source_id`
        column as it is used to identify stars.

        Parameters
        ----------
        outpath:
            `None`, 'str' or 'Path-like'. The path of the output file to save
            cartesian coordinates data if a gaia dataset is read. If `None`,
            it saves the data in the current working directory in a file
            named 'gaia_galactic.txt'. (Optional if `is_cartesian` in class
            initialization was `False`, otherwise this is ignored)

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
            self.__open_gaia(outpath=outpath)

    def __open_cartesian(self):
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        # genfromtxt appears to ignore the first element of dtype being
        # int64. This is taken care of here.
        new_data = np.zeros(data['source_id'].shape, dtype=self.dtype)
        for name in new_data.dtype.names:
            if name != 'source_id':
                new_data[name] = data[name]
            else:
                new_data['source_id'] = data['source_id'].astype(np.int64)

        return new_data

    def __open_gaia(self, outpath=None):
        ## FIXME: check these papers and update references
        # 10.1051/0004-6361/201832964 - parallax
        # 10.1051/0004-6361/201832727 - astrometric solution
        if self.dist_path is None:
            raise AttributeError("Distance file not found")

        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        # genfromtxt appears to ignore the first element of dtype being
        # int64. This is taken care of here.
        new_data = np.zeros(data['source_id'].shape, dtype=self.eq_dtype)
        for name in new_data.dtype.names:
            if name != 'source_id':
                new_data[name] = data[name]
            else:
                new_data['source_id'] = data['source_id'].astype(np.int64)

        data = new_data

        dist_data = np.genfromtxt(self.dist_path,
                                  delimiter=',',
                                  names=True,
                                  filling_values=np.nan)

        # genfromtxt appears to ignore the element of dtype being
        # int64. This is taken care of here and for the `Flag` column, too
        new_dist_data = np.zeros(dist_data['source_id'].shape,
                                 dtype=self.dist_dtype)
        for name in new_dist_data.dtype.names:
            if name != 'id' or name != 'flag':
                new_dist_data[name] = dist_data[name]
            elif name == 'id':
                new_dist_data['id'] = dist_data['id'].astype(np.int64)
            else:
                new_dist_data['flag'] = dist_data['flag'].astype(int)

        dist_data = new_dist_data

        # Selecting only Gaia EDR3 data which has distance estimated in the
        # other catalogue.
        data = np.array([data[idx] for idx in range(data['source_id'].shape[0])
                         if data['source_id'][idx] in dist_data['id']],
                        dtype=self.eq_dtype)

        # Data containing NaNs are discarded
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        # Data dist containing NaNs are discarded
        for col in self.dist_cols:
            idx = np.where(~np.isnan(dist_data[col]))
            dist_data = dist_data[idx]

        data = self.__eq_to_galcartesian(data, dist_data)

        if outpath is None:
            outpath = os.getcwd()
            outpath = Path(outpath).joinpath('gaia_galactic.txt')
        else:
            outpath = Path(outpath)

        np.savetxt(outpath,
                   data,
                   header='source_id,x,y,z,vx,vy,vz,ref_epoch',
                   delimiter=',')

    def __eq_to_galcartesian(self, data, dist_data):
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
            distances = dist_data[:][s]
            galactic_cartesian = self.__eq_to_galactic(equatorial, distances)

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

    def __eq_to_galactic(self, eq, dist):
        ra = np.deg2rad(eq['ra'])
        dec = np.deg2rad(eq['dec'])

        distances = self.__select_dist(dist)  # pc

        pmra = self.__masyr_to_kms(eq['pmra'], distances)  # pm_ra_cosdec
        pmdec = self.__masyr_to_kms(eq['pmdec'], distances)

        # position of star in ICRS cartesian coordinates.
        pos_icrs = np.array([np.cos(dec) * np.cos(ra) * distances * PC2KPC,
                             np.cos(dec) * np.sin(ra) * distances * PC2KPC,
                             np.sin(dec) * distances * PC2KPC])

        # conversion to galactic coordinates.
        pos_gal = self.A_G_inv.dot(pos_icrs)

        # unit vector for proper motion component in RA
        # (expressed in km/s, see above)
        p_icrs = np.array([-np.sin(ra),
                           np.cos(ra),
                           0])

        # unit vector for proper motion component in DEC
        # (expressed in km/s, see above)
        q_icrs = np.array([-np.cos(ra) * np.sin(dec),
                           -np.sin(ra) * np.sin(dec),
                           np.cos(dec)])

        # unit vector for radial velocity component as cross product between
        # p_icrs and q_icrs
        r_icrs = np.cross(p_icrs, q_icrs)

        # total proper motion in ICRS system and then converted to galactic.
        mu_icrs = p_icrs * pmra + q_icrs * pmdec + \
            r_icrs * eq['dr2_radial_velocity']
        mu_gal = self.A_G_inv.dot(mu_icrs)

        cartesian_data = np.array([(eq['source_id'],
                                    pos_gal[0], pos_gal[1], pos_gal[2],
                                    mu_gal[0], mu_gal[1], mu_gal[2],
                                    eq['ref_epoch'])], dtype=self.dtype)

        return cartesian_data

    @staticmethod
    def __select_dist(dist):
        data = np.where(np.isnan(dist['r_med_photogeo']),
                        dist['r_med_geo'],
                        dist['r_med_photogeo'])

        return data  # pc

    @staticmethod
    def __masyr_to_kms(pm, dist):
        pm_asyr = pm * 1e-3  # arcsec/yr
        pm_degyr = pm_asyr / 3600  # deg/yr
        pm_radyr = np.deg2rad(pm_degyr)  # rad/yr
        pm_rads = pm_radyr / YR2S  # rad/s

        distance_km = dist * PC2KM  # km

        pm_kms = pm_rads * distance_km  # km/s

        return pm_kms

    @property
    def A_G_inv(self):
        return self.__A_G_inv
