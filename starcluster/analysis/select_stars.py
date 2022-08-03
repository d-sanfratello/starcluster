import numpy as np


def by_parallax(data, parallax_min=None, parallax_max=None):
    if (parallax_min is not None and parallax_max is not None) and \
            parallax_min >= parallax_max:
        raise ValueError("parallax_min should be smaller than parallax_max.")

    new_data = data[:]

    if parallax_min is not None:
        new_data = new_data[new_data['parallax'] >= parallax_min]
    if parallax_max is not None:
        new_data = new_data[new_data['parallax'] <= parallax_max]

    return new_data


def by_proper_motion(data, axis, pm_min=None, pm_max=None):
    if (pm_min is not None and pm_max is not None) and \
            pm_min >= pm_max:
        raise ValueError("pm_min should be smaller than pm_max.")

    new_data = data[:]

    if axis not in ['l', 'b']:
        raise ValueError("`axis` must either be `l` or `b`.")
    else:
        axis = f'pm{axis}'

    if pm_min is not None:
        new_data = new_data[new_data[axis] >= pm_min]
    if pm_max is not None:
        new_data = new_data[new_data[axis] <= pm_max]

    return new_data


def by_radial_velocity(data,
                       radial_velocity_min=None, radial_velocity_max=None):
    if (radial_velocity_min is not None
        and radial_velocity_max is not None) \
            and radial_velocity_min >= radial_velocity_max:
        raise ValueError(
            "radial_velocity_min should be smaller than radial_velocity_max.")

    new_data = data[:]

    if radial_velocity_min is not None:
        new_data = new_data[new_data['radial_velocity'] >= radial_velocity_min]
    if radial_velocity_max is not None:
        new_data = new_data[new_data['radial_velocity'] <= radial_velocity_max]

    return new_data
