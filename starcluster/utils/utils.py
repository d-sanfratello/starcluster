import json
import os

import matplotlib.pyplot as plt
import numpy as np

from deprecated import deprecated
from distutils.spawn import find_executable
from figaro.credible_regions import ConfidenceArea
from figaro.marginal import marginalise
from figaro.mixture import DPGMM
from figaro.mixture import mixture
from figaro.transform import transform_to_probit
from figaro.utils import get_priors, recursive_grid
from matplotlib import rcParams
from pathlib import Path
from scipy.special import erfinv
from scipy.stats import norm as ndist

from starcluster.extract_data import Data

if find_executable('latex'):
    rcParams["text.usetex"] = True
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.fontsize"] = 12
rcParams["axes.labelsize"] = 16
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.6
rcParams["contour.negative_linestyle"] = 'solid'


# The transformation matrix from ICRS coordinates to galactic coordinates,
# named as the inverse matrix from galactic to ICRS coordinates. The exact
# values are as defined in Hobbs et al., 2021, Ch.4.
A_G_INV = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])


class ExpectedValues:
    __keys = ['ra', 'dec', 'l', 'b', 'parallax',
              'pmra', 'pmdec', 'radial_velocity']
    __mags = ['g_mag', 'bp_mag', 'rp_mag',
              'bp_rp', 'bp_g', 'g_rp']
    __data = ['l', 'b', 'parallax', 'pml', 'pmb', 'radial_velocity']

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
            'parallax', 'radial_velocity', 'l' and 'b'. Optionally, any of the
            photometric bands from Gaia can be set (as 'g_mag', 'bp_mag',
            'rp_mag') and any of the color indices can be set (as 'bp_rp',
            'bp_g' or 'g_rp').

        Returns
        -------
        expected:
            'numpy.ndarray'. When the instance is called, it returns a
            'np.ndarray' containing the expected values, in order, for 'l',
            'b', 'parallax', 'pml', 'pmb', 'radial_velocity' and, the
            photometric band and color index used.

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
        *, convert=True, std=None, epsilon=0):#,
        # mag='g_mag', c_index='bp_rp'):
    # fixme: check docstring
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
    parallax = dataset('parallax')
    pml = dataset('pml')
    pmb = dataset('pmb')
    radial_velocity = dataset('radial_velocity')
    # mag_ds = dataset(mag)
    # c_index_ds = dataset(c_index)

    columns_list = [l, b, parallax, pml, pmb, radial_velocity]

    bounds = [[l.min() - epsilon, l.max() + epsilon],
              [b.min() - epsilon, b.max() + epsilon],
              [parallax.min() - epsilon, parallax.max() + epsilon],
              [pml.min() - epsilon, pml.max() + epsilon],
              [pmb.min() - epsilon, pmb.max() + epsilon],
              [radial_velocity.min() - epsilon,
               radial_velocity.max() + epsilon]]

    samples = dataset.as_array()

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


@deprecated("This function is here for test purposes only and will likely be "
            "removed soon. Use figaro.utils.plot_multidim instead.")
def plot_multidim(draws, samples=None, out_folder='.', name='density',
                  labels=None, units=None, show=False,
                  save=True, subfolder=False, n_pts=200, true_value=None,
                  figsize=7, levels=[0.5, 0.68, 0.9]):
    """
    From figaro.utils.py at https://github.com/sterinaldi/figaro

    Plot the recovered multidimensional distribution along with samples from
    the true distribution (if available) as corner plot, with the single
    mixture gaussians added over the plot

    Parameters
    ----------
        draws:
            iterable. Container for mixture instances.
        samples:
            np.ndarray. Samples from the true distribution (if available)
        out_folder:
            str or Path. Output folder
        name:
            str. Name to be given to outputs
        labels:
            list-of-str. LaTeX-style quantity label, for plotting purposes
        units:
            list-of-str. LaTeX-style quantity unit, for plotting purposes
        save:
            bool. Whether to save the plot or not
        show:
            bool. Whether to show the plot during the run or not
        subfolder:
            bool. Whether to save in a dedicated subfolder
        n_pts:
            int. Number of grid points (same for each dimension)
        true_value:
            iterable. True value to plot
        figsize:
            double. Figure size (matplotlib)
        levels:
            iterable. Credible levels to plot
    """

    dim = draws[0].dim
    rec_label = r'\mathrm{DPGMM}'

    if labels is None:
        labels = ['$x_{0}$'.format(i + 1) for i in range(dim)]
    else:
        labels = ['${0}$'.format(l) for l in labels]

    if units is not None:
        labels = [l[:-1] + r'\ [{0}]$'.format(u) if not u == '' else l for l, u
                  in zip(labels, units)]

    levels = np.atleast_1d(levels)

    all_bounds = np.atleast_2d([d.bounds for d in draws])
    x_min = np.min(all_bounds, axis=-1).max(axis=0)
    x_max = np.max(all_bounds, axis=-1).min(axis=0)

    bounds = np.array([x_min, x_max]).T
    K = dim
    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.1  # w/hspace size
    plotdim = factor * dim + factor * (K - 1.0) * whspace
    dim_plt = lbdim + plotdim + trdim

    fig, axs = plt.subplots(K, K, figsize=(figsize, figsize))
    # Format the figure.
    lb = lbdim / dim_plt
    tr = (lbdim + plotdim) / dim_plt
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr, wspace=whspace,
                        hspace=whspace)

    # 1D plots (diagonal)
    for column in range(K):
        ax = axs[column, column]
        # Marginalise over all uninterested columns
        dims = list(np.arange(dim))
        dims.remove(column)
        marg_draws = marginalise(draws, dims)
        # Credible regions
        lim = bounds[column]
        x = np.linspace(lim[0], lim[1], n_pts + 2)[1:-1]
        dx = x[1] - x[0]

        probs = np.array([d.pdf(x) for d in marg_draws])

        percentiles = [50, 5, 16, 84, 95]
        p = {}
        for perc in percentiles:
            p[perc] = np.percentile(probs, perc, axis=0)
        norm = p[50].sum() * dx
        for perc in percentiles:
            p[perc] = p[perc] / norm

        # Samples (if available)
        if samples is not None:
            ax.hist(samples[:, column],
                    bins=int(np.sqrt(len(samples[:, column]))),
                    histtype='step',
                    density=True)
        # CR
        ax.fill_between(x, p[95], p[5], color='mediumturquoise', alpha=0.5)
        ax.fill_between(x, p[84], p[16], color='darkturquoise', alpha=0.5)
        if true_value is not None:
            if true_value[column] is not None:
                ax.axvline(true_value[column], c='orangered', lw=0.5)
        ax.plot(x, p[50], lw=0.7, alpha=0.2, color='steelblue')

        # new again, to correctly setup limits for plots on the diagonals.
        ax.set_xlim(bounds[column, 0], bounds[column, 1])

        # plot gaussians - This is new here
        pts = np.linspace(bounds[column, 0], bounds[column, 1], 102)[1:-1]
        x_probit = transform_to_probit(pts, np.atleast_2d(bounds[column]))
        parallax_thr = transform_to_probit(6.58, np.atleast_2d(bounds[2]))
        # Marginalise over all uninteresting columns
        for w, mu, var in zip(draws[0].w, draws[0].means, draws[0].covs):
            mu_g = mu[column]
            sigma = np.sqrt(var[column, column])
            pdf = ndist(loc=mu_g, scale=sigma)
            y_pts = w * pdf.pdf(x_probit)

            interval = bounds[column, 1] - bounds[column, 0]
            cdf = (pts - bounds[column, 0]) / interval
            jacobian_probit = np.sqrt(2*np.pi) * np.exp(erfinv(2 * cdf - 1)**2)
            cdf_probit = 1 / interval

            y_pts *= jacobian_probit * cdf_probit
            if mu[2] > parallax_thr:
                ax.plot(pts, y_pts, lw=0.5, color='red')
            else:
                ax.plot(pts, y_pts, lw=0.3, color='darkgreen')

        if column < K - 1:
            ax.set_xticks([])
            ax.set_yticks([])
        elif column == K - 1:
            ax.set_yticks([])
            if labels is not None:
                ax.set_xlabel(labels[-1])
            ticks = np.linspace(lim[0], lim[1], 5)
            ax.set_xticks(ticks)
            [l.set_rotation(45) for l in ax.get_xticklabels()]

    # 2D plots (off-diagonal)
    for row in range(K):
        for column in range(K):
            ax = axs[row, column]
            ax.grid(visible=False)
            if column > row:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif column == row:
                continue

            # Marginalise
            dims = list(np.arange(dim))
            dims.remove(column)
            dims.remove(row)
            marg_draws = marginalise(draws, dims)

            # Credible regions
            lim = bounds[[row, column]]
            grid, dgrid = recursive_grid(lim[::-1],
                                         np.ones(2, dtype=int) * int(n_pts))

            x = np.linspace(lim[0, 0], lim[0, 1], n_pts + 2)[1:-1]
            y = np.linspace(lim[1, 0], lim[1, 1], n_pts + 2)[1:-1]

            dd = np.array([d.pdf(grid) for d in marg_draws])
            median = np.percentile(dd, 50, axis=0)
            median = median / (median.sum() * np.prod(dgrid))
            median = median.reshape(n_pts, n_pts)

            X, Y = np.meshgrid(x, y)
            with np.errstate(divide='ignore'):
                logmedian = np.nan_to_num(np.log(median), nan=-np.inf,
                                          neginf=-np.inf)
            _, _, levs = ConfidenceArea(logmedian, x, y, adLevels=levels)
            ax.contourf(Y, X, np.exp(logmedian), cmap='Blues', levels=100)
            if true_value is not None:
                if true_value[row] is not None:
                    ax.axhline(true_value[row], c='orangered', lw=0.5)
                if true_value[column] is not None:
                    ax.axvline(true_value[column], c='orangered', lw=0.5)
                if true_value[column] is not None and true_value[
                    row] is not None:
                    ax.plot(true_value[column], true_value[row],
                            color='orangered', marker='s', ms=3)
            c1 = ax.contour(Y, X, logmedian, np.sort(levs), colors='k',
                            linewidths=0.3)
            if rcParams["text.usetex"] is True:
                ax.clabel(c1, fmt={l: '{0:.0f}\\%'.format(100 * s) for l, s in
                                   zip(c1.levels, np.sort(levels)[::-1])},
                          fontsize=3)
            else:
                ax.clabel(c1, fmt={l: '{0:.0f}\\%'.format(100 * s) for l, s in
                                   zip(c1.levels, np.sort(levels)[::-1])},
                          fontsize=3)
            ax.set_xticks([])
            ax.set_yticks([])

            if column == 0:
                ax.set_ylabel(labels[row])
                ticks = np.linspace(lim[0, 0], lim[0, 1], 5)
                ax.set_yticks(ticks)
                [l.set_rotation(45) for l in ax.get_yticklabels()]
            if row == K - 1:
                ticks = np.linspace(lim[1, 0], lim[1, 1], 5)
                ax.set_xticks(ticks)
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[column])

            elif row < K - 1:
                ax.set_xticks([])
            elif column == 0:
                ax.set_ylabel(labels[row])

    if show:
        plt.show()
    if save:
        if not subfolder:
            fig.savefig(Path(out_folder, '{0}.pdf'.format(name)),
                        bbox_inches='tight')
        else:
            if not Path(out_folder, 'density').exists():
                try:
                    Path(out_folder, 'density').mkdir()
                except FileExistsError:
                    pass
            fig.savefig(Path(out_folder, 'density', '{0}.pdf'.format(name)),
                        bbox_inches='tight')
    plt.close()
