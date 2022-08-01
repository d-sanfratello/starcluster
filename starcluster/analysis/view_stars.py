import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from starcluster.const import PC2KM, YR2S, PC2KPC


__dtype = np.dtype([('x', float),
                    ('y', float),
                    ('z', float),
                    ('vx', float),
                    ('vy', float),
                    ('vz', float)])


def __masyr_to_kms(pm, dist):
    pm_asyr = pm * 1e-3  # arcsec/yr
    pm_degyr = pm_asyr / 3600  # deg/yr
    pm_radyr = np.deg2rad(pm_degyr)  # rad/yr
    pm_rads = pm_radyr / YR2S  # rad/s
    distance_km = dist * PC2KM  # km
    pm_kms = pm_rads * distance_km  # km/s
    return pm_kms


def __eq_to_galactic(data):
    distances = (1 / data['plx']) / PC2KPC  # pc
    pml = __masyr_to_kms(data['pml_star'], distances)  # pm_l_cosb
    pmb = __masyr_to_kms(data['pmb'], distances)

    l = np.deg2rad(data['l'])
    b = np.deg2rad(data['b'])

    # conversion to galactic coordinates.
    pos_gal = np.array([np.cos(b) * np.cos(l),
                        np.cos(b) * np.sin(l),
                        np.sin(b)]) * distances

    # unit vector for proper motion component in galactic longitude l
    # (expressed in km/s, see above)
    p_gal = np.array([-np.sin(l),
                      np.cos(l),
                      0])
    # unit vector for proper motion component in galactic latitude b
    # (expressed in km/s, see above)
    q_gal = np.array([-np.cos(l) * np.sin(b),
                      -np.sin(l) * np.sin(b),
                      np.cos(b)])

    # unit vector for radial velocity component as cross product between
    # p_gal and q_gal
    r_gal = np.cross(p_gal, q_gal)

    # total proper motion in ICRS system and then converted to galactic.
    mu_gal = p_gal * pml + q_gal * pmb + r_gal * data['v_rad']

    cartesian_data = np.array([(pos_gal[0], pos_gal[1], pos_gal[2],
                                mu_gal[0], mu_gal[1], mu_gal[2])],
                              dtype=__dtype)

    return cartesian_data


def as_cartesian_array(data):
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []

    for s, star in enumerate(data):
        galactic_cartesian = __eq_to_galactic(data[:][s])
        x.append(galactic_cartesian['x'][0])
        y.append(galactic_cartesian['y'][0])
        z.append(galactic_cartesian['z'][0])
        vx.append(galactic_cartesian['vx'][0])
        vy.append(galactic_cartesian['vy'][0])
        vz.append(galactic_cartesian['vz'][0])

    data_cart = np.zeros(len(data), dtype=__dtype)
    data_cart['x'] = np.array(x)
    data_cart['y'] = np.array(y)
    data_cart['z'] = np.array(z)

    data_cart['vx'] = (np.array(vx) - np.mean(vx))
    data_cart['vy'] = (np.array(vy) - np.mean(vy))
    data_cart['vz'] = (np.array(vz) - np.mean(vz))

    v = np.sqrt(data_cart['vx'] ** 2
                + data_cart['vy'] ** 2
                + data_cart['vz'] ** 2)

    return data_cart, v


def __exp_to_cartesian_array(data):
    distances = (1 / data[2]) / PC2KPC  # pc
    pml = __masyr_to_kms(data[3], distances)  # pm_l_cosb
    pmb = __masyr_to_kms(data[4], distances)

    l = np.deg2rad(data[0])
    b = np.deg2rad(data[1])

    # conversion to galactic coordinates.
    pos_gal = np.array([np.cos(b) * np.cos(l),
                        np.cos(b) * np.sin(l),
                        np.sin(b)]) * distances

    # unit vector for proper motion component in galactic longitude l
    # (expressed in km/s, see above)
    p_gal = np.array([-np.sin(l),
                      np.cos(l),
                      0])
    # unit vector for proper motion component in galactic latitude b
    # (expressed in km/s, see above)
    q_gal = np.array([-np.cos(l) * np.sin(b),
                      -np.sin(l) * np.sin(b),
                      np.cos(b)])

    # unit vector for radial velocity component as cross product between
    # p_gal and q_gal
    r_gal = np.cross(p_gal, q_gal)

    # total proper motion in ICRS system and then converted to galactic.
    mu_gal = p_gal * pml + q_gal * pmb + r_gal * data[5]
    mu_gal -= mu_gal

    cartesian_data = np.array([(pos_gal[0], pos_gal[1], pos_gal[2],
                                mu_gal[0], mu_gal[1], mu_gal[2])],
                              dtype=__dtype)

    return cartesian_data


def quiver_plot(data,
                out_folder='.', name='cartesian_3d',
                units=[r'pc', r'pc', r'pc', r'km/s', r'km/s', r'km/s'],
                show=False, save=True, subfolder=False,
                true_value=None,
                figsize=7,
                elev=45,
                azim=45,
                scale=3e1,
                line_of_sight=False):

    labels = ['x', 'y', 'z']
    if units is not None:
        labels = [lab + f' $[{u}]$'
                  if not u == '' else lab for lab, u in zip(labels, units[:3])]

    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection='3d')

    if not hasattr(elev, '__iter__'):
        elev = [elev]
    if not hasattr(azim, '__iter__'):
        azim = [azim]

    lower_bounds = np.min([data['x'], data['y'], data['z']], axis=1)
    upper_bounds = np.max([data['x'], data['y'], data['z']], axis=1)

    mid_point = 0.5 * (upper_bounds + lower_bounds)

    for el in elev:
        for az in azim:
            ax.view_init(el, az)
            ax.quiver(data['x'], data['y'], data['z'],
                      data['vx'], data['vy'], data['vz'],
                      arrow_length_ratio=0.1,
                      length=1/scale)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

            ax.set_xlim(lower_bounds[0], upper_bounds[0])
            ax.set_ylim(lower_bounds[1], upper_bounds[1])
            ax.set_zlim(lower_bounds[2], upper_bounds[2])

            if line_of_sight:
                ax.quiver(0, 0, 0,
                          mid_point[0], mid_point[1], mid_point[2],
                          arrow_length_ratio=1/np.linalg.norm(mid_point),
                          length=1,
                          color='red',
                          label='Line of sight')

            if true_value is not None:
                true_value = true_value[:6]
                exp_value = __exp_to_cartesian_array(true_value)

                ax.scatter(exp_value['x'], exp_value['y'], exp_value['z'],
                           color='orangered',
                           label='Known position')

            ax.legend(loc='best')

            if show:
                plt.show()
            if save:
                if not subfolder:
                    fig.savefig(Path(out_folder,
                                     f'{name}_az{az}_el{el}.pdf'),
                                format='pdf',
                                bbox_inches='tight')
                else:
                    if not Path(out_folder, 'cart_3d').exists():
                        try:
                            Path(out_folder, 'cart_3d').mkdir()
                        except FileExistsError:
                            pass
                    fig.savefig(Path(out_folder,
                                     'cart_3d',
                                     f'{name}_az{az}_el{el}.pdf'),
                                format='pdf',
                                bbox_inches='tight')
