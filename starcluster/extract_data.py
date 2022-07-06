import h5py
import numpy as np

from zero_point import zpt
from pathlib import Path

from . import utils
from .utils.bias_corrections import (v_rad_bias_correction,
                                     pmra_bias_correction,
                                     pmdec_bias_correction)


class EquatorialData:
    astrometry_cols = ['source_id',
                       'ra', 'dec', 'parallax',
                       'pmra', 'pmdec', 'ruwe',
                       'radial_velocity',
                       'l', 'b',
                       'astrometric_params_solved',
                       'ecl_lat', 'grvs_mag',
                       'rv_template_teff', 'astrometric_primary_flag',
                       'phot_g_mean_mag',
                       'in_qso_candidates', 'in_galaxy_candidates']

    nu_for_solution = ['nu_eff_used_in_astrometry', 'pseudocolour']

    photometry_cols = ['phot_g_mean_mag',
                       'phot_bp_mean_mag',
                       'phot_rp_mean_mag']

    def __init__(self, path, *, convert=True,
                 no_extragalactic=False,
                 galaxy_cand=None, quasar_cand=None):
        # fixme: check docstring
        """
        Class to open an existing file containing astrometric data.

        At instantiation this class reads data from a Gaia DR3 csv file or
        from a file containing data already converted from a Gaia DR3 dataset.

        Use of `convert=True` enables automatic conversion into galactic
        coordinates. It is to be used if data has not been processed before.
        If used, longitude and latitude are extracted from data, while proper
        motion in galactic coordinates is evaluated following Hobbs et al.,
        2021, Ch.4.

        If data was already converted and saved use `convert=False`,
        instead, and the dataset will be read as in galactic components.

        Remember that the use of `convert=True` only stores the dataset
        inside the class instance. See method `save_dataset()` to save a copy of
        the converted dataset in HDF5 format.

        A Gaia DR3 dataset needs to contain (at least) the equatorial
        coordinates and proper motions ('ra', 'dec', 'pmra', 'pmdec'),
        'parallax' and 'dr2_radial_velocity' columns. It also needs 'l' and
        'b' coordinates and the three columns containing mean photometric
        magnitudes 'phot_g_mean_mag', 'phot_bp_mean_mag' and
        'phot_rp_mean_mag'. Finally, the dataset must
        contain the `source_id` column as it is used to identify stars. Any row
        containing missing data ('NULL' in Gaia DR3 and imported as `Nan`s) is
        deleted before conversion into galactic coordinates.

        When an instance of the class is called, with an argument `item`,
        if `item` is one of ['source_id', 'l', 'b', 'plx', 'pml_star', 'pmb',
        'v_rad', 'g_mag', 'bp_mag', 'rp_mag', 'bp_rp', 'bp_g', 'g_rp'],
        the corresponding column from the `gal` attribute is returned.
        Otherwhise it returns the item-th star from the `gal` attribute.

        Parameters
        ----------
        path:
            'str' or 'Path-like'. The path to the file containing data. For
            differences between datasets in galactic cartesian components and as
            a gaia dataset, read below.
        convert:
            'bool'. If `True`, data is assumed in ICRS coordinates and is
            converted into galactic coordinates, stored inside the `gal`
            attribute. When `False` data is assumed as already converted into
            galactic coordinates and is directly stored into the `gal`
            attribute. Default is `True`.

        """
        self.gal = None

        self.__names = ['source_id',
                        'l', 'b', 'plx', 'pml_star', 'pmb', 'v_rad']
        self.__phot_names = ['g_mag', 'bp_mag', 'rp_mag']
        self.eq_dtype = np.dtype([('source_id', np.int64),
                                  ('l', float),
                                  ('b', float),
                                  ('ra', float),
                                  ('dec', float),
                                  ('parallax', float),
                                  ('pmra', float),
                                  ('pmdec', float),
                                  ('radial_velocity', float)])
        self.gal_dtype = np.dtype([('source_id', np.int64),
                                   ('l', float),
                                   ('b', float),
                                   ('plx', float),
                                   ('pml_star', float),
                                   ('pmb', float),
                                   ('v_rad', float)])

        if convert:
            self.gal = self.__open_dr3(path,
                                       no_extragalactic=no_extragalactic,
                                       galaxy_cand=galaxy_cand,
                                       quasar_cand=quasar_cand)
        else:
            self.gal = self.__open_galactic(path)

    def save_dataset(self, name=None):
        # fixme: check docstring
        """
        Method of the `Data` class to save the extracted and converted
        galactic equatorial dataset into an external '.hdf5' file, in the
        current working directory.

        Position on the celestial sphere is expressed in degrees and
        parallax in mas. Proper motion's units are mas/yr (note that proper
        motion in galactic longitude is expressed as `pml* = pml * cos(b)`.
        Radial velocity is expressed in km/s. Photometric quantities are
        expressed in magnitudes.

        The saved dataset contains the equivalend of a numpy structured array
        with fields 'source_id', 'l', 'b', 'plx', 'pml_star', 'pmb', 'v_rad',
        'g_mag', 'bp_mag', 'rp_mag', 'bp_rp', 'bp_g' and 'g_rp'.

        Parameters
        ----------
        name:
            `None`, 'string' or 'Path'. The name of the output file to save
            galactic coordinates data in. If `None`, it saves the data in a file
            named 'gaia_edr3_galactic.hdf5'. File is saved in the current
            working directory.

        """
        if name is None:
            name = 'gaia_dr3_galactic.hdf5'
        elif isinstance(name, str) and name.find(".hdf5") < 0:
            name += ".hdf5"
            name = Path(name)
        elif isinstance(name, Path) and name.suffix == "":
            name.suffix = ".hdf5"

        with h5py.File(name, "w") as f:
            dset = f.create_dataset('data',
                                    shape=self.gal.shape,
                                    dtype=self.gal_dtype)

            dset[0:] = self.gal

    def as_array(self):#, *, mag=None, c_index=None):
        # fixme: check docstring
        """
        Method that converts the structured array contained within the `gal`
        attribute into an array of shape (N_stars, 8). The eight fields are,
        in order (l, b, plx, pml_star, pmb, v_rad, mag, c_index), where

            `pml_star = mu_l * cos(b)`

        and `mag` and `c_index` are the photometric band and the color index
        that will be displayed on the plots. These are defined via the `mag`
        and `c_index` keyword arguments.

        Parameters
        ----------
        mag:
            `str`, either 'g_mag', 'bp_mag', 'rp_mag'. The magnitude in the
            selected photometric band to be saved in this array.
            Default is 'g_mag'.
        c_index:
            `str`, either 'bp_rp', 'bp_g' or 'g_rp'. The Color index to be
            saved in this array. Default is 'bp_rp'.

        Returns
        -------
        `np.ndarray` of shape (N_stars, 8), containing the six kinematic
        columns and the selected magnitude and color index for the N_stars
        stars in the catalog.

        """
        # if mag is None:
        #     mag = 'g_mag'
        # if c_index is None:
        #     c_index = 'bp_rp'
        #
        # if mag not in ['g_mag', 'bp_mag', 'rp_mag']:
        #     raise ValueError(
        #         "Magnitude has to be either `g_mag`, `bp_mag` or `rp_mag`.")
        # if c_index not in ['bp_rp', 'bp_g', 'g_rp']:
        #     raise ValueError(
        #         "Magnitude has to be either `bp_rp`, `bp_g` or `g_rp`.")

        l = self('l')
        b = self('b')
        plx = self('plx')
        pml = self('pml_star')
        pmb = self('pmb')
        v_rad = self('v_rad')

        # if mag == 'g_mag':
        #     mag = self('g_mag')
        # elif mag == 'bp_mag':
        #     mag = self('bp_mag')
        # elif mag == 'rp_mag':
        #     mag = self('rp_mag')
        #
        # if c_index == 'bp_rp':
        #     c_index = self('bp_rp')
        # elif c_index == 'bp_g':
        #     c_index = self('bp_g')
        # elif c_index == 'g_rp':
        #     c_index = self('g_rp')

        # return np.vstack((l, b, plx, pml, pmb, v_rad, mag, c_index)).T
        return np.vstack((l, b, plx, pml, pmb, v_rad)).T

    def __open_dr3(self, path,
                   no_extragalactic=False,
                   galaxy_cand=None,
                   quasar_cand=None):
        # FIXME: checked papers:
        # 10.1051/0004-6361/202039587 - Photometric (Riello+2021)

        data = np.genfromtxt(path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan,
                             dtype=None)

        if no_extragalactic and (galaxy_cand is not None and
                                 quasar_cand is not None):
            # Removing galaxy and quasar candidates as in
            # Gaia Collaboration, “Gaia Data Release 3: The extragalactic
            # content”, arXiv e-prints, 2022.

            galaxy_cand = np.genfromtxt(galaxy_cand,
                                        delimiter=',',
                                        names=True,
                                        filling_values=np.nan,
                                        dtype=None)
            quasar_cand = np.genfromtxt(quasar_cand,
                                        delimiter=',',
                                        names=True,
                                        filling_values=np.nan,
                                        dtype=None)

            # Condition to obtain a "purer" quasar subsample from the
            # qso_candidates table, as in Gaia Collaboration (2022), Table 11.
            idx_qso = np.where(quasar_cand['gaia_crf_source'] |
                               quasar_cand['host_galaxy_flag'] < 6 |
                               quasar_cand['classlabel_dsc_joint'] == 'quasar' |
                               quasar_cand['vari_best_class_name'] == 'AGN')
            # Indentification of the source ids of the qso candidates
            source_id_qso = quasar_cand['source_id'][idx_qso]
            # Index to be deteled from the data structured array.
            idx_qso_delete = np.where(data['source_id'] in source_id_qso)
            data = np.delete(data, idx_qso_delete)

            # Condition to obtain a "purer" galaxy subsample from the
            # galaxy_candidates table, as in Gaia Collaboration (2022),
            # Table 12.
            idx_gal = np.where(~np.isnan(galaxy_cand['radius_sersic']) |
                               galaxy_cand['calsslabel_dsc_joint'] == 'galaxy' |
                               galaxy_cand['vari_best_class_name'] == 'GALAXY')
            # Indentification of the source ids of the galaxy candidates
            source_id_gal = galaxy_cand['source_id'][idx_gal]
            # Index to be deteled from the data structured array.
            idx_gal_delete = np.where(data['source_id'] in source_id_gal)
            data = np.delete(data, idx_gal_delete)

        # Data containing NaNs in astrometry columns are discarded.
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        # Checking for nu_eff_used_in_astrometry vs pseudocolour. At least
        # one must be available for the bias corrections.
        idx = np.where(np.isnan(data['nu_eff_used_in_astrometry']) &
                       np.isnan(data['pseudocolour']))
        data = data[idx]

        # FIXME: Riello+2021 10.1051/0004-6361/202039587 (check)
        # Parallax zero point correction
        # (Lindegren et al (A&A, 649, A4, 2021))
        zpt.load_tables()
        zero_point = zpt.get_zpt(data['phot_g_mean_mag'],
                                 data['nu_eff_used_in_astrometry'],
                                 data['pseudocolour'],
                                 data['ecl_lat'],
                                 data['astrometric_params_solved'])

        data['parallax'] -= zero_point

        # Radial Velocity bias correction
        # (Katz et al. (2022), Blomme et al. (2022))
        v_rad_corr = v_rad_bias_correction(data)
        data['radial_velocity'] -= v_rad_corr

        # Proper motion bias correction
        # (Cantat-Gaudin and Brandt, 2021)
        pmra_corr = pmra_bias_correction(data)
        data['pmra'] -= pmra_corr

        # Proper motion bias correction
        # (Cantat-Gaudin and Brandt, 2021)
        pmdec_corr = pmdec_bias_correction(data)
        data['pmdec'] -= pmdec_corr

        data_gal = self.__eq_to_gal(data)

        return data_gal

    def __open_galactic(self, path):
        dset = h5py.File(path, 'r')

        return dset['data']

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
        data_gal['v_rad'] = data['radial_velocity']
        # data_gal['g_mag'] = data['phot_g_mean_mag']
        # data_gal['bp_mag'] = data['phot_bp_mean_mag']
        # data_gal['rp_mag'] = data['phot_rp_mean_mag']
        # data_gal['bp_rp'] = data_gal['bp_mag'] - data_gal['rp_mag']
        # data_gal['bp_g'] = data_gal['bp_mag'] - data_gal['g_mag']
        # data_gal['g_rp'] = data_gal['g_mag'] - data_gal['rp_mag']

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
        mu_gal = utils.A_G_INV.dot(mu_icrs)

        pml_star = np.dot(p_gal, mu_gal)
        pmb = np.dot(q_gal, mu_gal)

        return pml_star, pmb

    def __call__(self, item):
        if item in self.gal_dtype.names:
            return self.gal[item]
        else:
            return self.gal[:][item]


Data = EquatorialData
