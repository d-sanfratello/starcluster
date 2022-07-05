import numpy as np


def v_rad_bias_correction(data):
    # Radial velocity bias correction
    # for rv_template_teff < 8500 K the correction is the one in equation (5)
    # from `Properties and validation of the radial velocities` by Katz, D.,
    # et al. (2022). For rv_template_teff >= 8500 K the correction is derived
    # from Blomme et al. (2022).
    g_rvs_mag = data['grvs_mag']
    rv_teff = data['rv_template_teff']

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
    # Function from function edr3ToICRF as showed in Cantat-Gaudin, T. and
    # Brandt, T. D., “Characterizing and correcting the proper motion bias of
    # the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol. 649, 2021.
    # doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    return np.sin(np.radians(x))


def _cosd(x):
    # Function from function edr3ToICRF as showed in Cantat-Gaudin, T. and
    # Brandt, T. D., “Characterizing and correcting the proper motion bias of
    # the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol. 649, 2021.
    # doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    return np.cos(np.radians(x))


# Variable from
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


def pmra_bias_correction(data):
    # Function from function edr3ToICRF as showed in Cantat-Gaudin, T. and
    # Brandt, T. D., “Characterizing and correcting the proper motion bias of
    # the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol. 649, 2021.
    # doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    """
    Input: source position , coordinates ,
    and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    g_mag = data['phot_g_mean_mag']
    ra = data['ra']
    dec = data['dec']

    pmra_corr = np.zeros_like(g_mag)
    corrected = np.where(g_mag < 13)

    g_mag_min = table1[0]
    g_mag_max = table1[1]

    # pick the appropriate omegaXYZ for the source’s magnitude:
    omega_x = np.array([
        table1[2][(g_mag_min <= mag) &
                  (g_mag_max > mag)][0]
        for mag in g_mag[corrected]
    ])
    omega_y = np.array([
        table1[3][(g_mag_min <= mag) &
                  (g_mag_max > mag)][0]
        for mag in g_mag[corrected]
    ])
    omega_z = np.array([
        table1[4][(g_mag_min <= mag) &
                  (g_mag_max > mag)][0]
        for mag in g_mag[corrected]
    ])

    ra = ra[corrected]
    dec = dec[corrected]

    pmra_corr[corrected] = -1 * _sind(dec) * _cosd(ra) * omega_x \
                           - _sind(dec) * _sind(ra) * omega_y \
                           + _cosd(dec) * omega_z

    return pmra_corr / 1000.


def pmdec_bias_correction(data):
    # Function from function edr3ToICRF as showed in Cantat-Gaudin, T. and
    # Brandt, T. D., “Characterizing and correcting the proper motion bias of
    # the bright Gaia EDR3 sources”, Astronomy and Astrophysics, vol. 649, 2021.
    # doi:10.1051/0004-6361/202140807.
    #
    # Code was slightly modified to improve readability in context with our
    # code.
    """
    Input: source position , coordinates ,
    and G magnitude from Gaia EDR3.
    Output: corrected proper motion.
    """
    g_mag = data['phot_g_mean_mag']
    ra = data['ra']

    pmdec_corr = np.zeros_like(g_mag)
    corrected = np.where(g_mag < 13)

    g_mag_min = table1[0]
    g_mag_max = table1[1]

    # pick the appropriate omegaXYZ for the source’s magnitude:
    omega_x = np.array([
        table1[2][(g_mag_min <= mag) &
                  (g_mag_max > mag)][0]
        for mag in g_mag[corrected]
    ])
    omega_y = np.array([
        table1[3][(g_mag_min <= mag) &
                  (g_mag_max > mag)][0]
        for mag in g_mag[corrected]
    ])

    ra = ra[corrected]

    pmdec_corr[corrected] = _sind(ra) * omega_x - _cosd(ra) * omega_y

    return pmdec_corr / 1000.
