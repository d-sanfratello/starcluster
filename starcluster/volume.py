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
    if aperture_deg >= 1:
        return volume_sector(data, aperture)

    aperture = np.deg2rad(aperture_deg)

    distances = np.sqrt((data['x'][:]**2 +
                         data['y'][:]**2 +
                         data['z'][:]**2))

    logvol = np.log(np.pi / 3) + \
        3 * np.log(distances.max()) +\
        2 * np.log(np.tan(aperture))

    return np.exp(logvol)


def volume_sector(data, aperture):
    aperture_deg = aperture / 60
    if aperture_deg == 180:
        return volume_sphere(data)

    aperture = np.deg2rad(aperture_deg)

    distances = np.sqrt((data['x'][:] ** 2 +
                         data['y'][:] ** 2 +
                         data['z'][:] ** 2))

    logvol = np.log(2 * np.pi / 3) + \
             3 * np.log(distances.max()) + \
             np.log(1 - np.cos(aperture))

    return np.exp(logvol)


def volume_sphere(data):
    distances = np.sqrt((data['x'][:] ** 2 +
                         data['y'][:] ** 2 +
                         data['z'][:] ** 2))

    logvol = np.log(4 * np.pi / 3) + \
        3 * np.log(distances.max())

    return np.exp(logvol)
