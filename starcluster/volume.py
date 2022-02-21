import numpy as np


def volume_cone(data, aperture):
    """

    Parameters
    ----------
    data
    aperture:
        In arcmin

    Returns
    -------

    """
    aperture_deg = aperture / 60

    if aperture_deg >= 90:
        pi_angle_arcmin = 180 * 60
        return volume_sphere(data) - volume_cone(data,
                                                 pi_angle_arcmin - aperture)

    aperture = np.deg2rad(aperture_deg)

    distances = np.sqrt((data['x'][:]**2 +
                         data['y'][:]**2 +
                         data['z'][:]**2))

    logvol = np.log(np.pi / 3) + \
        3 * np.log(distances.max()) +\
        2 * np.log(np.tan(aperture))

    return np.exp(logvol)


def volume_sphere(data):
    distances = np.sqrt((data['x'][:] ** 2 +
                         data['y'][:] ** 2 +
                         data['z'][:] ** 2))

    logvol = np.log(4 * np.pi / 3) + \
        3 * np.log(distances.max())

    return np.exp(logvol)
