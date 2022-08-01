import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from starcluster.const import PC2KM, YR2S


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
    distances = 1 / data['plx']
    pml = __masyr_to_kms(data['pml_star'], distances)  # pm_l_cosdec
    pmb = __masyr_to_kms(data['pmb'], distances)

    l = data['l']
    b = data['b']

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


def as_cartesian_array(data, scale=10):
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
    data_cart['x'] = np.array(x) * 1e3
    data_cart['y'] = np.array(y) * 1e3
    data_cart['z'] = np.array(z) * 1e3

    data_cart['vx'] = (np.array(vx) - np.mean(vx)) / scale
    data_cart['vy'] = (np.array(vy) - np.mean(vy)) / scale
    data_cart['vz'] = (np.array(vz) - np.mean(vz)) / scale

    v = np.sqrt(data_cart['vx'] ** 2
                + data_cart['vy'] ** 2
                + data_cart['vz'] ** 2)

    return data_cart, v


def quiver_plot(data,
                out_folder='.', name='cartesian_3d',
                units=None,
                show=False, save=True, subfolder=False,
                true_value=None,
                figsize=7,
                elev=45,
                azim=45):

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

    for el in elev:
        for az in azim:
            ax.view_init(el, az)
            ax.quiver(data['x'], data['y'], data['z'],
                      data['vx'], data['vy'], data['vz'],
                      arrow_length_ratio=0.3)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])

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
    plt.close()
