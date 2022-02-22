import numpy as np


def log_volume_cone(data, aperture):
    """
    Function that gives an estimate of the logarithm of the volume of the
    observed cone, given a dataset and the angular aperture of the cone from
    the center of the observed field.

    If the field is wider than 2° (aperture larger than 1°) this function
    calls `log_volume_sector`, assuming the conical approximation is no
    longer suitable.

    If the aperture is small enough, the depth of the cone is estimated from
    the farthest star of the field.

    Parameters
    ----------
    data:
        numpy structured array of type `Data.dtype'. The data that will be
        used to estimate the observed volume of sky, over the cone.
    aperture:
        `number`. The observation radius from the center of the field, expressed
        in arcminutes.

    Returns
    -------
    `float`:
        The natural logarithm of the volume of the observed cone.

    """
    aperture_deg = aperture / 60
    if aperture_deg >= 1:
        return log_volume_sector(data, aperture)

    aperture = np.deg2rad(aperture_deg)

    distances = np.sqrt((data['x'][:]**2 +
                         data['y'][:]**2 +
                         data['z'][:]**2))

    logvol = np.log(np.pi / 3) + \
        3 * np.log(distances.max()) +\
        2 * np.log(np.tan(aperture))

    return logvol


def log_volume_sector(data, aperture):
    """
    Function that gives an estimate of the logarithm of the volume of the
    observed spherical sector, given a dataset and the angular aperture of the
    sector from the center of the observed field.

    If the field is as wide as the whole sphere (aperture equal to 180°) this
    function calls `log_volume_sphere`, to reduce computing error.

    If the aperture is small enough, the radius of the sphere the sector
    belongs to is estimated from the farthest star of the field.

    Parameters
    ----------
    data:
        numpy structured array of type `Data.dtype'. The data that will be
        used to estimate the observed volume of sky, over the cone.
    aperture:
        `number`. The observation radius from the center of the field, expressed
        in arcminutes.

    Returns
    -------
    `float`:
        The natural logarithm of the volume of the observed spherical sector.

    """
    aperture_deg = aperture / 60
    if aperture_deg == 180:
        return log_volume_sphere(data)

    aperture = np.deg2rad(aperture_deg)

    distances = np.sqrt((data['x'][:] ** 2 +
                         data['y'][:] ** 2 +
                         data['z'][:] ** 2))

    logvol = np.log(2 * np.pi / 3) + \
             3 * np.log(distances.max()) + \
             np.log(1 - np.cos(aperture))

    return logvol


def log_volume_sphere(data):
    """
    Function that gives an estimate of the logarithm of the volume of the
    observed celestial sphere, given a dataset.

    Te radius of the sphere is estimated from the farthest star of the field.

    Parameters
    ----------
    data:
        numpy structured array of type `Data.dtype'. The data that will be
        used to estimate the observed volume of sky, over the cone.

    Returns
    -------
    `float`:
        The natural logarithm of the volume of the observed celestial sphere.

    """

    distances = np.sqrt((data['x'][:] ** 2 +
                         data['y'][:] ** 2 +
                         data['z'][:] ** 2))

    logvol = np.log(4 * np.pi / 3) + \
        3 * np.log(distances.max())

    return logvol
