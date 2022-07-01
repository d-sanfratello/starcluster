import numpy as np
import optparse as op
import pandas as pd

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
       'ecl_lat', 'grvs_mag']
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

    # if other_src[0].size:
    f_sigma[other_src] = np.ones_like(other_src)
    # if fainter_src[0].size:
    f_sigma[fainter_src] = _f_sigma_polynomial(g_rvs_mag[fainter_src],
                                                   *pars_fainter)
    # if brighter_src[0].size:
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
    f_sigma = df.apply(f_sigma_wrapper, axis=1)
    df['radial_velocity_error'] *= f_sigma

    # Radial Velocity bias correction
    v_rad_bias = df.apply(v_rad_wrapper, axis=1)

    means = np.array([df['ra'],
                      df['dec'],
                      df['parallax'] - zero_point,
                      df['pmra'],
                      df['pmdec'],
                      df['radial_velocity'] - v_rad_bias]).T
    covs = np.array([
        fill_matrix(*[df[name][i] for name in names]
                    ) for i in range(len(means))])

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
