import h5py
import numpy as np

from zero_point import zpt
from pathlib import Path

from .const import A_G_INV, VAR_SYS_PLX_SINGLE
from .utils.bias_corrections import (radial_velocity_bias_correction,
                                     pmra_bias_correction,
                                     pmdec_bias_correction)


# noinspection PyTypeChecker
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
                       'in_qso_candidates', 'in_galaxy_candidates',
                       'parallax_over_error']

    nu_for_solution = ['nu_eff_used_in_astrometry', 'pseudocolour']

    photometry_cols = ['phot_g_mean_mag',
                       'phot_bp_mean_mag',
                       'phot_rp_mean_mag']

    def __init__(self, path, *,
                 convert=True,
                 ruwe=None,
                 galaxy_cand=None, quasar_cand=None,
                 parallax_over_error=None):
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
        inside the class instance. See method `save_dataset()` to save a copy
        of the converted dataset in HDF5 format.

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
        if `item` is one of ['source_id', 'l', 'b', 'parallax', 'pml',
        'pmb', 'radial_velocity', 'g_mag', 'bp_mag', 'rp_mag', 'bp_rp', 'bp_g',
        'g_rp'], the corresponding column from the `gal` attribute is returned.
        Otherwhise it returns the item-th star from the `gal` attribute.

        Parameters
        ----------
        path:
            'str' or 'Path-like'. The path to the file containing data. For
            differences between datasets in galactic cartesian components and
            as a gaia dataset, read below.
        convert:
            'bool'. If `True`, data is assumed in ICRS coordinates and is
            converted into galactic coordinates, stored inside the `gal`
            attribute. When `False` data is assumed as already converted into
            galactic coordinates and is directly stored into the `gal`
            attribute. Default is `True`.
        ruwe:
        galaxy_cand:
        quasar_cand:
        parallax_over_error:

        """
        self.gal = None
        self.cov = None

        self.data_path = Path(path)
        self.project_folder = self.data_path.parent.parent

        self.__names = ['source_id',
                        'l', 'b', 'parallax', 'pml', 'pmb', 'radial_velocity']
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
                                   ('parallax', float),
                                   ('pml', float),
                                   ('pmb', float),
                                   ('radial_velocity', float)])
        self.gal_cov_dtype = np.dtype([('source_id', np.int64),
                                       ('cov', np.ndarray)])

        if convert:
            self.gal, self.cov = self.__open_dr3(
                self.data_path,
                galaxy_cand=galaxy_cand,
                quasar_cand=quasar_cand,
                ruwe=ruwe,
                parallax_over_error=parallax_over_error
            )
        else:
            self.gal, self.cov = self.__open_galactic(self.data_path)

    def save_dataset(self, name=None):
        # fixme: check docstring
        """
        Method of the `Data` class to save the extracted and converted
        galactic equatorial dataset into an external '.hdf5' file, in the
        current working directory.

        Position on the celestial sphere is expressed in degrees and
        parallax in mas. Proper motion's units are mas/yr (note that proper
        motion in galactic longitude is expressed as `pml* = pml * cos(b)`).
        Radial velocity is expressed in km/s. Photometric quantities are
        expressed in magnitudes.

        The saved dataset contains the equivalend of a numpy structured array
        with fields 'source_id', 'l', 'b', 'parallax', 'pml', 'pmb',
        'radial_velocity', 'g_mag', 'bp_mag', 'rp_mag', 'bp_rp', 'bp_g' and
        'g_rp'.

        Parameters
        ----------
        name:
            `None`, 'string' or 'Path'. The name of the output file to save
            galactic coordinates data in. If `None`, it saves the data in a
            file named 'gaia_edr3_galactic.hdf5'. File is saved in the current
            working directory.

        """
        if name is None:
            name = 'gaia_dr3_galactic.h5'
        elif isinstance(name, str) and name.find(".h5") < 0:
            name += ".h5"
            name = Path(name)
        elif isinstance(name, Path) and name.suffix == "":
            name.suffix = ".h5"

        with h5py.File(name, "w") as f:
            dset_data = f.create_dataset('data',
                                         shape=self.gal.shape,
                                         dtype=self.gal_dtype)
            dset_data[0:] = self.gal

            for _, source in enumerate(self.cov):
                dset_cov = f.create_dataset(str(source['source_id']),
                                            shape=(6, 6),
                                            dtype=float)
                dset_cov[0:] = self.cov['cov'][_]

    def as_array(self, dataset='data'):
        # fixme: check docstring
        """
        Method that converts the structured array contained within the `gal`
        attribute into an array of shape (N_stars, 8). The eight fields are,
        in order (l, b, parallax, pml, pmb, radial_velocity), where

            `pml = mu_l * cos(b)`.

        Returns
        -------
        `np.ndarray` of shape (N_stars, 6), containing the six kinematic
        columns N_stars stars in the catalog.
        """
        if dataset == 'data':
            l = self['l']
            b = self['b']
            parallax = self['parallax']
            pml = self['pml']
            pmb = self['pmb']
            radial_velocity = self['radial_velocity']

            return np.vstack((l, b, parallax, pml, pmb, radial_velocity)).T
        elif dataset == 'cov':
            covs = np.zeros(shape=(self['source_id'].shape[0], 6, 6))

            for _, c in enumerate(self.cov['cov']):
                covs[_] = c

            return covs
        else:
            raise ValueError(
                f"Unknown dataset type {dataset}. Expect `data` or `cov`."
            )

    def __open_dr3(self, path,
                   ruwe=None,
                   galaxy_cand=None,
                   quasar_cand=None,
                   parallax_over_error=None):
        # FIXME: checked papers:
        # 10.1051/0004-6361/202039587 - Photometric (Riello+2021)

        path = Path(path)

        data = np.genfromtxt(path,
                             delimiter=',',
                             names=True,
                             filling_values=np.nan,
                             dtype=None)

        if galaxy_cand is not None:
            # Removing galaxy candidates, identified as in Gaia Collaboration,
            # “Gaia Data Release 3: The extragalactic content”, arXiv e-prints,
            # 2022.

            galaxy_cand = np.genfromtxt(galaxy_cand,
                                        delimiter=',',
                                        names=True,
                                        dtype=None,
                                        filling_values={
                                            'radius_sersic': np.nan,
                                            'classlabel_dsc_joint': '',
                                            'vari_best_class_name': ''
                                        },
                                        encoding=None)

            # Condition to obtain a "purer" galaxy subsample from the
            # galaxy_candidates table, as in Gaia Collaboration (2022),
            # Table 12.
            idx_gal = np.where((~np.isnan(galaxy_cand['radius_sersic'])) |
                               (galaxy_cand['classlabel_dsc_joint'] ==
                                'galaxy') |
                               (galaxy_cand['vari_best_class_name'] ==
                                'GALAXY'))
            # Identification of the source ids of the galaxy candidates
            source_id_gal = galaxy_cand['source_id'][idx_gal]
            # Index to be deleted from the data structured array.
            idx_gal_delete = np.where(data['source_id'] in source_id_gal)
            data = np.delete(data, idx_gal_delete)

        if quasar_cand is not None:
            # Removing qso candidates, identified as in Gaia Collaboration,
            # “Gaia Data Release 3: The extragalactic content”, arXiv e-prints,
            # 2022.

            quasar_cand = np.genfromtxt(quasar_cand,
                                        delimiter=',',
                                        names=True,
                                        dtype=None,
                                        filling_values={
                                            'gaia_crf_source': None,
                                            'host_galaxy_flag': 9,
                                            'classlabel_dsc_joint': '',
                                            'vari_best_class_name': ''
                                        },
                                        encoding=None)

            # Condition to obtain a "purer" quasar subsample from the
            # qso_candidates table, as in Gaia Collaboration (2022), Table 11.
            idx_qso = np.where((quasar_cand['gaia_crf_source']) |
                               (quasar_cand['host_galaxy_flag'] < 6) |
                               (quasar_cand['classlabel_dsc_joint'] ==
                                'quasar') |
                               (quasar_cand['vari_best_class_name'] == 'AGN'))
            # Identification of the source ids of the qso candidates
            source_id_qso = quasar_cand['source_id'][idx_qso]
            # Index to be deleted from the data structured array.
            idx_qso_delete = np.where(data['source_id'] in source_id_qso)
            data = np.delete(data, idx_qso_delete)

        # Data containing NaNs in astrometry columns are discarded.
        for col in self.astrometry_cols:
            idx = np.where(~np.isnan(data[col]))
            data = data[idx]

        # Data with a precision higher than the given parallax_over_error.
        # Data is selected both on the positive and negative side of
        # parallaxes, to ensure the least bias is inserted into the dataset.
        # See G. Collaboration, "Gaia Early Data Release 3. Summary of the
        # contents and survey properties", Astronomy and Astrophysics,
        # vol. 649, p. A1, May 2021, doi: 10.1051/0004-6361/202039657 and C.
        # Fabricius et al., "Gaia Early Data Release 3. Catalogue
        # validation", Astronomy and Astrophysics, vol. 649, p. A5, May 2021,
        # doi: 10.1051/0004-6361/202039834.
        if parallax_over_error is not None:
            idx = np.where(
                (data['parallax_over_error'] >= parallax_over_error) |
                (data['parallax_over_error'] <= -parallax_over_error)
            )
            data = data[idx]

        # RUWE correction as in Lindegren, L. 2018, technical note
        # GAIA-C3-TN-LU-LL-124. Their suggested value is 1.4
        if ruwe is not None:
            idx = np.where(data['ruwe'] <= ruwe)
            data = data[idx]

        # Checking for nu_eff_used_in_astrometry vs pseudocolour. At least
        # one must be available for the bias corrections.
        idx = np.where(~(np.isnan(data['nu_eff_used_in_astrometry']) &
                         np.isnan(data['pseudocolour'])))
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

        # From Flynn et al (2022)
        affected_stars = np.where((data['phot_g_mean_mag'] < 12) &
                                  (data['bp_rp'] > 1))
        affected_stars_source_id = data['source_id'][affected_stars]
        zpt_correction = 0.01  # mas
        correction = np.zeros(shape=zero_point.shape)
        correction[affected_stars] += zpt_correction

        analytics_folder = self.project_folder.joinpath('other-datasets')
        with h5py.File(analytics_folder.joinpath('zpt_corr.h5'), "w") as f:
            dset_plx = f.create_dataset('orig_parallax',
                                        shape=zero_point.shape,
                                        dtype=float)
            dset_plx[0:] = data['parallax']

            dset_zpt = f.create_dataset('zpt',
                                        shape=zero_point.shape,
                                        dtype=float)
            dset_zpt[0:] = zero_point

            dset_corr = f.create_dataset('corr',
                                         shape=zero_point.shape,
                                         dtype=float)
            dset_corr[0:] = correction

        data['parallax'] -= zero_point
        # data['parallax'] -= correction

        # Radial Velocity bias correction
        # (Katz et al. (2022), Blomme et al. (2022))
        radial_velocity_corr = radial_velocity_bias_correction(data)
        data['radial_velocity'] -= radial_velocity_corr

        # Proper motion bias correction
        # (Cantat-Gaudin and Brandt, 2021)
        pmra_corr = pmra_bias_correction(data)
        data['pmra'] -= pmra_corr

        # Proper motion bias correction
        # (Cantat-Gaudin and Brandt, 2021)
        pmdec_corr = pmdec_bias_correction(data)
        data['pmdec'] -= pmdec_corr

        data_gal, data_err = self.__eq_to_gal(data, affected_stars_source_id)

        return data_gal, data_err

    def __open_galactic(self, path):
        dset = h5py.File(path, 'r')

        sources = list(dset.keys())
        sources.remove('data')

        data = dset['data']
        cov = np.zeros(shape=data.shape,
                       dtype=self.gal_cov_dtype)
        source_ids = np.array(sources).astype(np.int64)
        cov['source_id'] = source_ids

        cov_matrices = []
        for _, source in enumerate(sources):
            cov_matrix = np.array(dset[source]).reshape((6, 6))
            cov_matrices.append(cov_matrix)
        cov['cov'] = cov_matrices

        return dset['data'], cov

    def __eq_to_gal(self, data, correction_source_id):
        pml = []
        pmb = []

        for s in range(data['source_id'].shape[0]):
            pml_s, pmb_s = self.__pm_conversion(data[:][s])

            pml.append(pml_s)
            pmb.append(pmb_s)

        data_gal = np.zeros(len(data), dtype=self.gal_dtype)
        data_gal['source_id'] = data['source_id']
        data_gal['l'] = data['l']
        data_gal['b'] = data['b']
        data_gal['parallax'] = data['parallax']
        data_gal['pml'] = pml
        data_gal['pmb'] = pmb
        data_gal['radial_velocity'] = data['radial_velocity']
        # data_gal['g_mag'] = data['phot_g_mean_mag']
        # data_gal['bp_mag'] = data['phot_bp_mean_mag']
        # data_gal['rp_mag'] = data['phot_rp_mean_mag']
        # data_gal['bp_rp'] = data_gal['bp_mag'] - data_gal['rp_mag']
        # data_gal['bp_g'] = data_gal['bp_mag'] - data_gal['g_mag']
        # data_gal['g_rp'] = data_gal['g_mag'] - data_gal['rp_mag']

        cov_matrices = []
        for _, s in enumerate(data['source_id']):
            cov_matrices.append(self.__error_propagation(data[:][_],
                                                         correction_source_id))

        data_err = np.zeros(data.shape[0], dtype=self.gal_cov_dtype)
        data_err['source_id'] = data['source_id']
        data_err['cov'] = cov_matrices

        return data_gal, data_err

    def __pm_conversion(self, data):
        # [1] Butkevich, A. and Lindegren, L., ‘4.1.7 Transformations of
        # astrometric data and error propagation ‣ Gaia Data Release 3
        # Documentation release 1.1’.
        #   https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/
        #   chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
        # (accessed Dec. 01, 2022).

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
        mu_gal = A_G_INV.dot(mu_icrs)

        pml = np.dot(p_gal, mu_gal)
        pmb = np.dot(q_gal, mu_gal)

        return pml, pmb

    def __error_propagation(self, data, correction_source_id):
        # [1] Butkevich, A. and Lindegren, L., ‘4.1.7 Transformations of
        # astrometric data and error propagation ‣ Gaia Data Release 3
        # Documentation release 1.1’.
        #   https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/
        #   chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
        # (accessed Dec. 01, 2022).

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

        rot_gal = np.column_stack((p_gal, q_gal))
        rot_ICRS = np.column_stack((p_icrs, q_icrs))

        G = np.dot(rot_gal.T, np.dot(A_G_INV, rot_ICRS))
        J = np.array([[G[0, 0], G[0, 1], 0,       0,       0],
                      [G[1, 0], G[1, 1], 0,       0,       0],
                      [      0,       0, 1,       0,       0],
                      [      0,       0, 0, G[0, 0], G[0, 1]],
                      [      0,       0, 0, G[1, 0], G[1, 1]]])

        errs = [
            'ra_error', 'dec_error', 'parallax_error', 'pmra_error',
            'pmdec_error'
        ]
        cov_icrs_astrometric = np.zeros(shape=(5, 5), dtype=float)
        for i, err_i in enumerate(errs):
            for j, err_j in enumerate(errs):
                cov = data[err_i] * data[err_j]
                if i != j:
                    errname_i = err_i.split('_error')[0]
                    errname_j = err_j.split('_error')[0]
                    try:
                        corr_name = f'{errname_i}_{errname_j}_corr'
                        corr = data[corr_name]
                    except ValueError:
                        corr_name = f'{errname_j}_{errname_i}_corr'
                        corr = data[corr_name]

                    cov *= corr
                # elif i == j and err_i.find('parallax') >= 0:
                #     cov += VAR_SYS_PLX_SINGLE

                    # if data['source_id'] in correction_source_id:
                    #     # extra correction error, if used
                    #     cov += 2e-3**2  # mas^2
                cov_icrs_astrometric[i, j] = cov

        CJ_t = np.dot(cov_icrs_astrometric, J.T)
        cov_gal_astrometric = np.dot(J, CJ_t)
        cov_gal = np.zeros(shape=(6, 6), dtype=float)
        cov_gal[0:5, 0:5] = cov_gal_astrometric
        cov_gal[5, 5] = data['radial_velocity_error']**2

        return cov_gal

    def __getitem__(self, item):
        return np.copy(self.gal[item])


Data = EquatorialData
