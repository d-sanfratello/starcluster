import json
import os

import numpy as np

from figaro.mixture import DPGMM
from figaro.mixture import mixture
from figaro.utils import get_priors
from pathlib import Path

from .extract_data import Data


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
        """
        Class to interpret the expected values for a cluster for the quantities
        used with the DPGMM, from the equatorial coordinates and relative
        proper motions, parallax, radial velocity and galactic latitude and
        longitude.

        Right ascension, declination, galactic longitude and latitude,
        parallax, radial_velocity, and proper motions in ra and dec are
        mandatory keys, but photometric band and color index can be omitted
        and will be set to `np.nan`.

        Parameters
        ----------
        expected:
            'dictionary'. The expected keys are 'ra', 'dec', 'pmra', 'pmdec',
            'plx', 'v_rad', 'l' and 'b'. Optionally, any of the photometric
            bands from Gaia can be set (as 'g_mag', 'bp_mag', 'rp_mag') and
            any of the color indices can be set (as 'bp_rp', 'bp_g' or 'g_rp').

        Returns
        -------
        expected:
            'numpy.ndarray'. When the instance is called, it returns a
            'np.ndarray' containing the expected values, in order, for 'l',
            'b', 'plx', 'pml', 'pmb', 'v_rad' and, the photometric band and
            color index used.

        """
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
        *, convert=True, std=None, epsilon=0,
        mag='g_mag', c_index='bp_rp'):
    """
    Function running in an automated way the DPGMM analysis. It first loads
    the data (See 'starcluster.extract_data.Data'), then it determines the
    bounds from minimum and maximum value for each axis of the dataset.

    If an `epsilon != 0` is passed, then it is subtracted and added to each
    bound limit, to avoid the issues described in figaro documentation about
    the probit transform raising an exception when data is sampled on the
    boundaries. If `epsilon` is passed as `0` (as in the default case),
    the points corresponding to the boundaries are removed from the sample.

    Then the priors on the hyperparameters are determined with the
    `figaro.utils.get_priors` function, passing the option argument 'std' to
    the call.

    Finally, the DPGMM is run.

    Parameters
    ----------
    path:
        'string' or 'Path' object containing the path to the file with the data
        for the DP.
    outpath:
        'string or 'Path' object containing the path to the output file to
        save the converted data to. It is ignored if `convert = False`.
        Optional (default = None. See `export_data.Data` documentation for
        information on this behaviour).
    convert:
        'bool'. Keyword argument passed to the `extract_data.Data` class. If
        data was already converted into pure galactic coordinates, set it to
        `False`. If data is raw from Gaia Archive, set it to `True`. Optional
        (default = True).
    std:
        `None` or array-like. It contains the keyword argument 'std' to be
        passed to 'figaro.utils.get_priors`. See `get_priors` docuementation
        for its behaviour. Optional (default = None).
    epsilon:
        number. The margin to be removed from each boundary before passing
        the bounds keyword argument to `figaro.utils.get_priors` and to
        `figaro.mixture.DPGMM`. If set to `0`, it removes the data sitting on
        the bounds from the sample. Optional (default = 0).
    mag:
        'string', either `g_mag`, `bp_mag` or `rp_mag`. The photometric band
        to be used in the inference. Optional (default = 'g_mag').
    c_index:
        'string', either `bp_rp`, `bp_g` or `g_rp`. The color index to be used
        in the inference. Optional (default = 'bp_rp').

    Returns
    -------
    'tuple':
        A tuple of values containing, in order: the dataset as a
        `export_data.Data` object, the bounds and priors passed to the DPGMM,
        the mix and density obtained from the inference.

    """
    dataset = Data(path=path, convert=convert)

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

    columns_list = [l, b, plx, pml, pmb, v_rad, mag_ds, c_index_ds]

    bounds = [[l.min() - epsilon, l.max() + epsilon],
              [b.min() - epsilon, b.max() + epsilon],
              [plx.min() - epsilon, plx.max() + epsilon],
              [pml.min() - epsilon, pml.max() + epsilon],
              [pmb.min() - epsilon, pmb.max() + epsilon],
              [v_rad.min() - epsilon, v_rad.max() + epsilon],
              [mag_ds.min() - epsilon, mag_ds.max() + epsilon],
              [c_index_ds.min() - epsilon, c_index_ds.max() + epsilon]]

    samples = dataset.as_array(mag=mag, c_index=c_index)

    if epsilon == 0:
        # FIXME: to be tested.
        del_idx = []
        for idx, data_column in enumerate(columns_list):
            del_idx.append(np.argmin(data_column))
            del_idx.append(np.argmax(data_column))

        samples = np.delete(samples, del_idx, axis=0)

    if std is None:
        prior = get_priors(bounds, samples)
    else:
        prior = get_priors(bounds, std=std)

    mix = DPGMM(bounds=bounds,
                prior_pars=prior)

    density = mix.density_from_samples(samples)

    return dataset, bounds, prior, mix, density


def save_density(density, *, folder=None, file='mixture.json'):
    """
    Function that, passed a `figaro.mixture` object, exports it into
    a json file.

    Parameters
    ----------
    density:
        `figaro.mixture` object, to be saved for later analysis.
    folder:
        `string` or `Path` object. The folder in which the output json file
        will be saved. Optional (default = `os.getcwd()`).
    file:
        `string`. The name of the file to output the json to. Optional (
        default = `mixture.json').

    """
    dict_ = density.__dict__

    for key in dict_.keys():
        value = dict_[key]

        if isinstance(value, np.ndarray):
            value = value.tolist()

        dict_[key] = value

    s = json.dumps(dict_, indent=4)

    if folder is None:
        folder = os.getcwd()
    if file.find('.json') < 0:
        file += '.json'
    with open(Path(folder).joinpath(file), 'w') as f:
        json.dump(s, f)


def import_density(file):
    """
    Function that reads a json file containing the parameters for a saved
    `figaro.mixture` object and returns an instance of such object.

    Parameters
    ----------
    file:
        `string` or `Path`. The path to the json file of the mixture.

    Returns
    -------
    density:
        `figaro.mixture`. An instance of the class containing the data stored ù
        in the json file.

    """
    with open(Path(file), 'r') as fjson:
        dictjson = json.load(fjson)

    dict_ = json.loads(dictjson)
    dict_.pop('log_w')

    for key in dict_.keys():
        value = dict_[key]

        if isinstance(value, list):
            dict_[key] = np.array(value)

    density = mixture(**dict_)

    return density
