import numpy as np
import os

from pathlib import Path

from .const import PC2KM, YR2S, PC2KPC


class Data:
    astrometry_cols = ['source_id', 'ra', 'dec',
                       'pmra', 'pmdec', 'dr2_radial_velocity',
                       'ref_epoch']
    dist_cols = ['Source',
                 'rgeo', 'b_rgeo', 'B_rgeo',
                 'rpgeo', 'b_rpgeo', 'B_rpgeo',
                 'Flag']

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
        self.dist_dtype = np.dtype([('Source', np.int64),
                                    ('rgeo', float),
                                    ('b_rgeo', float),
                                    ('B_rgeo', float),
                                    ('rpgeo', float),
                                    ('b_rpgeo', float),
                                    ('B_rpgeo', float),
                                    ('Flag', int)])
        self.def_dtype = np.dtype([('source_id', np.int64),
                                   ('Source', np.int64),
                                   ('ra', float),
                                   ('dec', float),
                                   ('parallax', float),
                                   ('pmra', float),
                                   ('pmdec', float),
                                   ('dr2_radial_velocity', float),
                                   ('ref_epoch', float),
                                   ('rgeo', float),
                                   ('b_rgeo', float),
                                   ('B_rgeo', float),
                                   ('rpgeo', float),
                                   ('b_rpgeo', float),
                                   ('B_rpgeo', float),
                                   ('Flag', int)])

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

        # genfromtxt appears to dislike the use of dtype right away.
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
                             filling_values=np.nan,
                             dtype=self.eq_dtype)
        # Sorting dataset by source_id
        data = np.sort(data, order='source_id')

        dist_data = np.genfromtxt(self.dist_path,
                                  delimiter=',',
                                  names=True,
                                  filling_values=np.nan,
                                  dtype=self.dist_dtype)
        # Sorting dataset by source_id
        dist_data = np.sort(dist_data, order='Source')

        # Deleting dist_data elements that do not belong to gaia dataset
        # This is executed twice to ensure the dataset is of the correct
        # shape. The complete I/352 dataset is smaller than Gaia EDR3,
        # so next check should be enough but since this code was tested
        # on two datasets downloaded separately, this check ensures no source
        # mismatch happened because of differences in sky patch at data
        # retrieval.
        save_idx = np.where(np.isin(dist_data['Source'],
                                    data['source_id']))
        dist_data = dist_data[:][save_idx]

        # Deleting gaia elements that do not belong to dist_data dataset.
        # This check deletes gaia sources for which distance was not
        # estimated in I/352 catalogue. If data is downloaded in a single
        # query, previous check *should* not be needed.
        save_idx = np.where(np.isin(data['source_id'],
                                    dist_data['Source']))
        data = data[:][save_idx]

        # Creating single catalogue containing all data
        def_data = np.empty(data['source_id'].shape[0],
                            dtype=self.def_dtype)
        for col in self.def_dtype.names:
            if col in self.astrometry_cols:
                def_data[col] = data[col]
            elif col in self.dist_cols:
                def_data[col] = dist_data[col]

        # If for some reason a row has two different source_ids associated,
        # an error is raised
        missed_ids = np.nonzero(def_data['source_id'] != def_data['Source'])
        if missed_ids[0] > 0:
            raise ValueError

        # Data containing NaNs in astrometry columns are discarded (useful
        # for gaiaedr3.gaia_source_dr3_radial_velocity, since many objects do
        # not have this measure).
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(def_data[col]))
            def_data = def_data[idx]

        cart_data = self.__eq_to_galcartesian(def_data)

        if outpath is None:
            outpath = os.getcwd()
            outpath = Path(outpath).joinpath('gaia_galactic.txt')
        else:
            outpath = Path(outpath)

        np.savetxt(outpath,
                   cart_data,
                   header='source_id,x,y,z,vx,vy,vz,ref_epoch',
                   delimiter=',',
                   comments='')

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
            galactic_cartesian = self.__eq_to_galactic(data[:][s])

            source_id.append(data['source_id'][s])
            x.append(galactic_cartesian['x'][0])
            y.append(galactic_cartesian['y'][0])
            z.append(galactic_cartesian['z'][0])
            vx.append(galactic_cartesian['vx'][0])
            vy.append(galactic_cartesian['vy'][0])
            vz.append(galactic_cartesian['vz'][0])
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

    def __eq_to_galactic(self, data):
        ra = np.deg2rad(data['ra'])
        dec = np.deg2rad(data['dec'])

        distances = self.__select_dist(data)  # pc

        pmra = self.__masyr_to_kms(data['pmra'], distances)  # pm_ra_cosdec
        pmdec = self.__masyr_to_kms(data['pmdec'], distances)

        # position of star in ICRS cartesian coordinates.
        pos_icrs = np.array([np.cos(dec) * np.cos(ra) * distances,
                             np.cos(dec) * np.sin(ra) * distances,
                             np.sin(dec) * distances])
        pos_icrs *= PC2KPC

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
            r_icrs * data['dr2_radial_velocity']
        mu_gal = self.A_G_inv.dot(mu_icrs)

        cartesian_data = np.array([(data['source_id'],
                                    pos_gal[0], pos_gal[1], pos_gal[2],
                                    mu_gal[0], mu_gal[1], mu_gal[2],
                                    data['ref_epoch'])], dtype=self.dtype)

        return cartesian_data

    @staticmethod
    def __select_dist(dist):
        data = np.where(np.isnan(dist['rpgeo']),
                        dist['rgeo'],
                        dist['rpgeo']).flatten()

        return data[0]  # pc

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


class EquatorialData:
    astrometry_cols = ['source_id', 'ra', 'dec', 'l', 'b',
                       'pmra', 'pmdec', 'dr2_radial_velocity',
                       'ref_epoch']

    def __init__(self, path, *, convert=True):
        """
        Class to open an existing file containing astrometric data.

        At instantiation this class reads data from a Gaia EDR3 csv file.
        This dataset is to be convertend into galactic equatorial coordinates
        using the method from Hobbs et al., 2021, Ch.4 to convert Gaia
        EDR3 equatorial data to galactic equatorial coordinates.

        Use `convert=True` if data is in RA-DEC coordiantes, enabling
        automatic conversion into galactic coordinates. While galactic
        longitude and latitude is extracted from data, proper motion in
        galactic coordinates is evaluated following Hobbs et al., 2021,
        Ch.4. If data was already converted and saved use `convert=False`,
        instead, and the dataset will be read as in galactic components.

        Remember that the use of `convert=True` only stores the dataset
        inside the class instance. Use method `save_dataset()` to save a copy of
        the converted dataset.

        Gaia EDR3 csv dasatet must contain the `ref_epoch` column. Any row
        containing missing data (imported as `Nan`s) is deleted before
        conversion into galactic coordinates. Finally, the dataset must
        contain the `source_id` column as it is used to identify stars.

        Parameters
        ----------
        path:
            'str' or 'Path-like'. The path to the file containing data. For
            differences between datasets in galactic cartesian components and as
            a gaia dataset, read below.
        convert:
            'bool'. If `True`, data is assumed in RA-DEC coordinates and is
            converted into galactic coordinates, stored inside the `gal`
            attribute. When `False` data is assumed as already converted into
            galactic coordinates and is directly stored into the `gal`
            attribute. Default is `True`.
        """
        self.path = Path(path)
        self.gal = None

        self.__names = ['source_id',
                        'l', 'b', 'plx', 'pml_star', 'pmb', 'v_rad',
                        'ref_epoch']
        self.eq_dtype = np.dtype([('source_id', np.int64),
                                  ('l', float),
                                  ('b', float),
                                  ('ra', float),
                                  ('dec', float),
                                  ('parallax', float),
                                  ('pmra', float),
                                  ('pmdec', float),
                                  ('dr2_radial_velocity', float),
                                  ('ref_epoch', float)])
        self.gal_dtype = np.dtype([('source_id', np.int64),
                                   ('l', float),
                                   ('b', float),
                                   ('plx', float),
                                   ('pml_star', float),
                                   ('pmb', float),
                                   ('v_rad', float),
                                   ('ref_epoch', float)])

        # The matrix to convert from equatorial cartesian coordinates to
        # galactic cartesian components. See Hobbs et al., 2021, Ch.4
        self.__A_G_inv = np.array([
            [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
            [0.4941094278755837, -0.4448296299600112,  0.7469822444972189],
            [-0.8676661490190047, -0.1980763734312015,  0.4559837761750669]])

        if convert:
            self.gal = self.__open_equatorial()
        else:
            self.gal = self.__open_galactic()

    def save_dataset(self, outpath=None):
        """
        Method of the `Data` class to save the extracted or converted
        galactic equatorial dataset into an external file, whose path is
        define by `outpath` argument.

        Celestial positios are expressed in degrees and parallax in mas.
        Proper motion's units are mas/yr (note that proper motion in
        galactic longitude is expressed as `pml* = pml * cos(b)`. Radial
        velocity are expressed in km/s.

        The saved dataset contains the equivalend of a numpy structured array
        with fields 'source_id', 'l', 'b', 'plx', 'pml_star', 'pmb', 'v_rad'
        and 'ref_epoch', containing the galactic coordinates for each star.

        Parameters
        ----------
        outpath:
            `None`, 'str' or 'Path-like'. The path of the output file to save
            cartesian coordinates data if a gaia dataset is read. If `None`,
            it saves the data in the current working directory in a file
            named 'gaia_edr3_galactic.txt'.
        """
        if outpath is None:
            outpath = Path(os.getcwd()).joinpath('gaia_edr3_galactic.txt')

        np.savetxt(outpath,
                   self.gal,
                   header='source_id,l,b,plx,pml_star,pmb,v_rad,ref_epoch',
                   delimiter=',',
                   comments='')

    def as_array(self):
        l = self['l']
        b = self['b']
        plx = self['plx']
        pml = self['pml_star']
        pmb = self['pmb']
        v_rad = self['v_rad']

        return np.vstack((l, b, plx, pml, pmb, v_rad)).T

    def __open_equatorial(self):
        ## FIXME: check these papers and update references
        # 10.1051/0004-6361/201832964 - parallax
        # 10.1051/0004-6361/201832727 - astrometric solution
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan,
                             dtype=self.eq_dtype)

        # Data containing NaNs in astrometry columns are discarded (useful
        # for gaiaedr3.gaia_source_dr2_radial_velocity, since many objects do
        # not have this measure).
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        data_gal = self.__eq_to_gal(data)

        return data_gal

    def __open_galactic(self):
        data = np.genfromtxt(self.path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan)

        # genfromtxt appears to dislike the use of dtype right away.
        new_data = np.zeros(data['source_id'].shape, dtype=self.gal_dtype)
        for name in new_data.gal_dtype.names:
            if name != 'source_id':
                new_data[name] = data[name]
            else:
                new_data['source_id'] = data['source_id'].astype(np.int64)

        return new_data

    def __eq_to_gal(self, data):
        pml_star = []
        pmb = []

        for s in range(data['source_id'].shape[0]):
            pml_star_s, pmb_s = self.__pm_conversion(data[:][s])

            pml_star.append(pml_star_s)
            pmb.append(pmb_s)

        data_gal = np.zeros(len(data), dtype=self.gal_dtype)
        data_gal['source_id'] = data['source_id']
        data_gal['l'] = data['l']
        data_gal['b'] = data['b']
        data_gal['plx'] = data['parallax']
        data_gal['pml_star'] = pml_star
        data_gal['pmb'] = pmb
        data_gal['v_rad'] = data['dr2_radial_velocity']
        data_gal['ref_epoch'] = data['ref_epoch']

        return data_gal

    def __pm_conversion(self, data):
        ra = np.deg2rad(data['ra'])
        dec = np.deg2rad(data['dec'])
        l = np.deg2rad(data['l'])
        b = np.deg2rad(data['b'])

        # unit vector for proper motion component in RA [mas/yr]
        p_icrs = np.array([-np.sin(ra),
                           np.cos(ra),
                           0])

        # unit vector for proper motion component in DEC [mas/yr]
        q_icrs = np.array([-np.cos(ra) * np.sin(dec),
                           -np.sin(ra) * np.sin(dec),
                           np.cos(dec)])

        # same vectors, but in galactic coordinates
        p_gal = np.array([-np.sin(l),
                          np.cos(l),
                          0])
        q_gal = np.array([-np.cos(l) * np.sin(b),
                          -np.sin(l) * np.sin(b),
                          np.cos(b)])

        # total proper motion in ICRS system and then converted to galactic.
        mu_icrs = p_icrs * data['pmra'] + q_icrs * data['pmdec']
        mu_gal = self.A_G_inv.dot(mu_icrs)

        pml_star = np.dot(p_gal, mu_gal)
        pmb = np.dot(q_gal, mu_gal)

        return pml_star, pmb

    @property
    def A_G_inv(self):
        return self.__A_G_inv

    def __getitem__(self, item):
        if item in self.gal_dtype.names:
            return self.gal[item]
        else:
            return self.gal[:][item]
