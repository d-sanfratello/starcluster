import numpy as np

from figaro.mixture import DPGMM
from figaro.utils import get_priors
from pathlib import Path

from .extract_data import EquatorialData


# The transformation matrix from ICRS coordinates to galactic coordinates,
# named as the inverse matrix from galactic to ICRS coordinates. The exact
# values are as defined in Hobbs et al., 2021, Ch.4.
A_G_INV = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])


class ExpectedValues:
    __keys = ['ra', 'dec', 'l', 'b', 'plx',
              'pmra', 'pmdec', 'v_rad']
    __mags = ['g_mag', 'bp_mag', 'rp_mag',
              'bp_rp', 'bp_g', 'g_rp']
    __data = ['l', 'b', 'plx', 'pml', 'pmb', 'v_rad']

    def __init__(self, expected):
        for k in self.__keys:
            try:
                setattr(self, k, expected[k])
            except KeyError:
                raise KeyError(
                    f"Must include an expected value for {k}."
                    f"If unknown, use `None`.")

        for m in self.__mags[:3]:
            if m in expected.keys():
                self.__mag = expected[m]
            else:
                self.__mag = np.nan
        for c in self.__mags[3:]:
            if c in expected.keys():
                self.__c_index = expected[c]
            else:
                self.__c_index = np.nan

        ra_rad = np.deg2rad(self.ra)
        dec_rad = np.deg2rad(self.dec)
        l_rad = np.deg2rad(self.l)
        b_rad = np.deg2rad(self.b)

        p_icrs = np.array([-np.sin(ra_rad),
                           np.cos(ra_rad),
                           0])
        q_icrs = np.array([-np.cos(ra_rad) * np.sin(dec_rad),
                           -np.sin(ra_rad) * np.sin(dec_rad),
                           np.cos(dec_rad)])
        p_gal = np.array([-np.sin(l_rad),
                          np.cos(l_rad),
                          0])
        q_gal = np.array([-np.cos(l_rad) * np.sin(b_rad),
                          -np.sin(l_rad) * np.sin(b_rad),
                          np.cos(b_rad)])

        mu_icrs = p_icrs * self.pmra + q_icrs * self.pmdec
        mu_gal = A_G_INV.dot(mu_icrs)

        self.pml = np.dot(p_gal, mu_gal)
        self.pmb = np.dot(q_gal, mu_gal)

    def __call__(self):
        arr = []
        for name in self.__data:
            arr.append(getattr(self, name))
        arr.append(self.__mag)
        arr.append(self.__c_index)

        return arr


def dpgmm(
        path, outpath=None,
        *, convert=True, std=None, epsilon=1e-3,
        mag='g_mag', c_index='bp_rp'):
    dataset = EquatorialData(path=path, convert=convert)

    if convert:
        if outpath is None:
            outpath = Path(str(path).strip('.csv') + '.txt')
        dataset.save_dataset(name=outpath)

    l = dataset('l')
    b = dataset('b')
    plx = dataset('plx')
    pml = dataset('pml_star')
    pmb = dataset('pmb')
    v_rad = dataset('v_rad')
    mag_ds = dataset(mag)
    c_index_ds = dataset(c_index)

    bounds = [[l.min() - epsilon, l.max() + epsilon],
              [b.min() - epsilon, b.max() + epsilon],
              [plx.min() - epsilon, plx.max() + epsilon],
              [pml.min() - epsilon, pml.max() + epsilon],
              [pmb.min() - epsilon, pmb.max() + epsilon],
              [v_rad.min() - epsilon, v_rad.max() + epsilon],
              [mag_ds.min() - epsilon, mag_ds.max() + epsilon],
              [c_index_ds.min() - epsilon, c_index_ds.max() + epsilon]]

    samples = dataset.as_array(mag=mag, c_index=c_index)

    if std is None:
        prior = get_priors(bounds, samples)
    else:
        prior = get_priors(bounds, std=std)

    mix = DPGMM(bounds=bounds,
                prior_pars=prior)

    density = mix.density_from_samples(samples)

    return dataset, bounds, prior, mix, density
