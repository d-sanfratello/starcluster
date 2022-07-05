import numpy as np
import optparse as op
import pandas as pd

from scipy.sparse import block_diag
from scipy.stats import multivariate_normal as mn
from pathlib import Path
from tqdm import tqdm
from zero_point import zpt

'''
['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
       'parallax_error', 'parallax_over_error', 'pmra', 'pmra_error', 'pmdec',
       'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr',
       'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
       'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'ruwe',
       'radial_velocity', 'radial_velocity_error', 'l', 'b',
       'nu_eff_used_in_astrometry', 'pseudocolour', 'astrometric_params_solved',
       'ecl_lat', 'grvs_mag', 'rv_template_teff', 'astrometric_primary_flag', 
       'phot_g_mean_mag']
'''


def fill_matrix(dra, ddec, dplx, dpmra, dpmdec, dvrad, dradec,
                draplx, drapmra, drapmdec, ddecplx, ddecpmra, ddecpmdec, 
                dplxpmra, dplxpmdec, dpmrapmdec):

    errors = np.array([dra, ddec, dplx, dpmra, dpmdec, dvrad])
    
    row1 = np.array([0.5, dradec, draplx, drapmra, drapmdec, 0])*errors * dra
    row2 = np.array([0, 0.5, ddecplx, ddecpmra, ddecpmdec, 0])*errors * ddec
    row3 = np.array([0, 0, 0.5, dplxpmra, dplxpmdec, 0])*errors * dplx
    row4 = np.array([0, 0, 0, 0.5, dpmrapmdec, 0])*errors * dpmra
    row5 = np.array([0, 0, 0, 0,  0.5, 0])*errors * dpmdec
    row6 = np.array([0, 0, 0, 0, 0, 0.5])*errors * dvrad
    
    cov_mat = np.array([row1, row2, row3, row4, row5, row6])
    cov_mat = cov_mat + cov_mat.T
    
    return cov_mat


def _f_sigma_polynomial(x, a, b, c):
    f_sigma = a + b * x + c * x**2
    return f_sigma


def f_sigma_wrapper(df_row):
    # Radial Velocity error correction
    # for g_rvs > 8 mag we apply formula (1) from `Gaia Data Release 3
    # Catalogue Validation`, parameters from table (1)
    g_rvs_mag = np.array([df_row.grvs_mag])
    fainter_src = np.where(g_rvs_mag > 12)
    brighter_src = np.where(8 <= g_rvs_mag <= 12)
    other_src = np.where(g_rvs_mag < 8)

    pars_brighter = (0.318, 0.3884, -0.02778)
    pars_fainter = (16.554, -2.4899, 0.09933)

    f_sigma = np.zeros_like(g_rvs_mag)

    f_sigma[other_src] = np.ones_like(other_src)
    f_sigma[fainter_src] = _f_sigma_polynomial(g_rvs_mag[fainter_src],
                                               *pars_fainter)
    f_sigma[brighter_src] = _f_sigma_polynomial(g_rvs_mag[brighter_src],
                                                *pars_brighter)

    return f_sigma


def v_rad_wrapper(df_row):
    # Radial velocity bias correction
    # for rv_template_teff < 8500 K the correction is the one in equation (5)
    # from `Properties and validation of the radial velocities` by Katz, D.,
    # et al. (2022). For rv_template_teff >= 8500 K the correction is derived
    # from Blomme et al. (2022).
    g_rvs_mag = np.array([df_row.grvs_mag])
    rv_teff = np.array([df_row.rv_template_teff])

    cold_stars = np.where(rv_teff < 8500)
    hot_stars = np.where(rv_teff >= 8500)

    v_rad_bias = np.zeros_like(g_rvs_mag)

    # Colder stars (correction by Katz et al.)
    brighter_cold = np.where(g_rvs_mag[cold_stars] < 11)
    fainter_cold = np.where(g_rvs_mag[cold_stars] >= 11)

    v_rad_bias[brighter_cold] = 0
    v_rad_bias[fainter_cold] = 0.02755 * g_rvs_mag[fainter_cold]**2 \
        - 0.55863 * g_rvs_mag[fainter_cold] + 2.81129

    # Hotter stars (correction by Blomme et al.)
    v_rad_bias[hot_stars] = 7.98 - 1.135 * g_rvs_mag[hot_stars]

    return v_rad_bias


def _sind(x):
    # Function copied from function edr3ToICRF as showed in Cantat-Gaudin,
    # T. and Brandt, T. D., “Characterizing and correcting the proper motion
    # bias of the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol.
    # 649, 2021. doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    return np.sin(np.radians(x))


def _cosd(x):
    # Function copied from function edr3ToICRF as showed in Cantat-Gaudin,
    # T. and Brandt, T. D., “Characterizing and correcting the proper motion
    # bias of the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol.
    # 649, 2021. doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    return np.cos(np.radians(x))


table1 = '''0.0 9.0 18.4 33.8 -11.3
            9.0 9.5 14.0 30.7 -19.4
            9.5 10.0 12.8 31.4 -11.8
            10.0 10.5 13.6 35.7 -10.5
            10.5 11.0 16.2 50.0 2.1
            11.0 11.5 19.4 59.9 0.2
            11.5 11.75 21.8 64.2 1.0
            11.75 12.0 17.7 65.6 -1.9
            12.0 12.25 21.3 74.8 2.1
            12.25 12.5 25.7 73.6 1.0
            12.5 12.75 27.3 76.6 0.5
            12.75 13.0 34.9 68.9 -2.9 '''
table1 = np.fromstring(table1, sep=' ').reshape((12, 5)).T


def pmra_wrapper(df_row):
    # Function copied from function edr3ToICRF as showed in Cantat-Gaudin,
    # T. and Brandt, T. D., “Characterizing and correcting the proper motion
    # bias of the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol.
    # 649, 2021. doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    """
    Input: source position , coordinates ,
    and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    g_mag = np.array([df_row.phot_g_mean_mag])
    ra = np.array([df_row.ra])
    dec = np.array([df_row.dec])

    pmra_corr = np.zeros_like(g_mag)
    corrected = np.where(g_mag < 13)

    g_mag_min = table1[0]
    g_mag_max = table1[1]

    if g_mag < 13:
        # pick the appropriate omegaXYZ for the source’s magnitude:
        omega_x = table1[2][(g_mag_min <= g_mag) & (g_mag_max > g_mag)][0]
        omega_y = table1[3][(g_mag_min <= g_mag) & (g_mag_max > g_mag)][0]
        omega_z = table1[4][(g_mag_min <= g_mag) & (g_mag_max > g_mag)][0]

        pmra_corr[corrected] = -1 * _sind(dec) * _cosd(ra) * omega_x \
                               - _sind(dec) * _sind(ra) * omega_y \
                               + _cosd(dec) * omega_z

    return pmra_corr / 1000.


def pmdec_wrapper(df_row):
    # Function copied from function edr3ToICRF as showed in Cantat-Gaudin,
    # T. and Brandt, T. D., “Characterizing and correcting the proper motion
    # bias of the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol.
    # 649, 2021. doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    """
    Input: source position , coordinates ,
    and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    g_mag = np.array([df_row.phot_g_mean_mag])
    ra = np.array([df_row.ra])

    pmdec_corr = np.zeros_like(g_mag)
    corrected = np.where(g_mag < 13)

    g_mag_min = table1[0]
    g_mag_max = table1[1]

    if g_mag < 13:
        # pick the appropriate omegaXYZ for the source’s magnitude:
        omega_x = table1[2][(g_mag_min <= g_mag) & (g_mag_max > g_mag)][0]
        omega_y = table1[3][(g_mag_min <= g_mag) & (g_mag_max > g_mag)][0]

        pmdec_corr[corrected] = _sind(ra) * omega_x - _cosd(ra) * omega_y

    return pmdec_corr / 1000.


def cov_from_attitude(df, covs):
    # Following the naming convention from Holl and Lindegren (2012)
    primary_flag = np.array(df.astrometric_primary_flag)

    primary_sources = np.where(primary_flag)
    p_src = primary_sources

    secondary_sources = np.where(~primary_flag)
    s_src = secondary_sources

    U = block_diag(covs)

    return None
    

def read_data(file):
    names = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 
             'pmdec_error', 'radial_velocity_error', 'ra_dec_corr', 
             'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 
             'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 
             'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']
    df = pd.read_csv(file)

    # Parallax zero point correction
    # (Lindegren et al (A&A, 649, A4, 2021))
    zpt.load_tables()
    zero_point = df.apply(zpt.zpt_wrapper, axis=1)

    # Radial Velocity error correction
    # (Babusiaux et al. (2022))
    f_sigma = df.apply(f_sigma_wrapper, axis=1)
    df['radial_velocity_error'] *= f_sigma

    # Radial Velocity bias correction
    # (Katz et al. (2022), Blomme et al. (2022))
    v_rad_corr = df.apply(v_rad_wrapper, axis=1)

    # Proper motion bias correction
    # (Cantat-Gaudin and Brandt, 2021)
    pmra_corr = df.apply(pmra_wrapper, axis=1)
    pmdec_corr = df.apply(pmdec_wrapper, axis=1)

    means = np.array([df['ra'],
                      df['dec'],
                      df['parallax'] - zero_point,
                      df['pmra'] - pmra_corr,
                      df['pmdec'] - pmdec_corr,
                      df['radial_velocity'] - v_rad_corr]).T
    covs = np.array([
        fill_matrix(*[df[name][i] for name in names]
                    ) for i in range(len(means))])

    # Error characterization as in Holl, B. and Lindegren L., A&A, 543,
    # A14 (2012)
    covs_primary = cov_from_attitude(df, covs)

    print([df['ra'].max(), df['dec'].max(), df['parallax'].max(),
           df['pmra'].max(), df['pmdec'].max(), df['radial_velocity'].max()])
    print([df['ra'].min(), df['dec'].min(), df['parallax'].min(),
           df['pmra'].min(), df['pmdec'].min(), df['radial_velocity'].min()])
    return means, covs


def main():
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type="string",
                      dest="input_folder", help=".csv from GAIA query")
    parser.add_option("-o", "--output", type="string",
                      dest="output_folder", help="Output folder", default='.')
    parser.add_option("--n_samples", type="int", dest="n_samples",
                      help="Number of samples for each star", default=1000)

    (options, args) = parser.parse_args()
    
    options.input_folder = Path(options.input_folder).resolve()
    options.output_folder = Path(options.output_folder).resolve()
    if not options.output_folder.exists():
        options.output_folder.mkdir(parents=True)
    
    means, covs = read_data(options.input_folder)
    np.savetxt('means.txt', means)

    for i, (m, s) in tqdm(enumerate(zip(means, covs)),
                          total=len(means), desc='Stars'):
        samples = mn(m, s).rvs(options.n_samples)
        np.savetxt(Path(options.output_folder, '{0}.txt'.format(i)), samples)


if __name__ == "__main__":
    main()
