import numpy as np


def by_parallax(data, parallax_min=None, parallax_max=None):
    if (parallax_min is not None and parallax_max is not None) and \
            parallax_min >= parallax_max:
        raise ValueError("parallax_min should be smaller than parallax_max.")

    new_data = np.copy(data.gal)

    if parallax_min is not None:
        new_data = new_data[new_data['plx'] >= parallax_min]
    if parallax_max is not None:
        new_data = new_data[new_data['plx'] <= parallax_max]

    return new_data
