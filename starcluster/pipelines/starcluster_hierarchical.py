import optparse as op
import os

from pathlib import Path
from tqdm import tqdm

# import corner
# import matplotlib.pyplot as plt
import numpy as np

from starcluster.extract_data import Data
from starcluster.utils import ExpectedValues

# from corner import corner as crn
from figaro.mixture import HDPGMM
from figaro.utils import get_priors, make_single_gaussian_mixture
from figaro.plot import plot_multidim
from figaro.load import save_density, load_density

from starcluster.exceptions import StarclusterException


def main():
    parser = op.OptionParser()

    parser.add_option("-b", "--bounds",
                      type="list", dest="bounds",
                      help="Sets the bounds for both the inference and the "
                           "plot.")
    parser.add_option("-m", "--means",
                      type="list", dest="means",
                      default=None,
                      help="Sets the means to be used as prior.")
    parser.add_option("-s", "--stds",
                      type="list", dest="stds",
                      default=None,
                      help="Sets the standard deviations to be used as prior.")
    parser.add_option("-n", "--n_draws",
                      type=int, dest="n_draws",
                      default=1,
                      help="The number of draws to be extracted from the "
                           "HDPGMM.")
    parser.add_option("-p", "--post-process",
                      action="store_true", dest="postprocess",
                      help="If set, it does not run the hierarchical "
                           "analysis, but opens a saved density distribution.")
    parser.add_option("-e", "--expected",
                      action="store_true", dest="show_expected",
                      help="If set, this shows the expected values for the "
                           "cluster in the final plot.")
    parser.add_option("-r", "--random-subsample",
                      type="int", dest="random_subsample",
                      default=None,
                      help="Sets the size of a random subsample of the full "
                           "dataset (for testing purposes only).")
    (options, args) = parser.parse_args()

    project_dir = Path(os.getcwd())
    name = project_dir.name.lower()

    bounds = eval(options.bounds)

    expected = None
    if options.show_expected:
        expected_file = project_dir.joinpath('expected', 'expected.csv')
        expected = ExpectedValues.load(expected_file)

    data_file = project_dir.joinpath('data', 'data.h5')
    dataset = Data(path=data_file, convert=False)

    samples = dataset.as_array('data')
    covs = dataset.as_array('cov')

    if options.random_subsample is not None:
        n_stars = options.random_subsample
        rng = np.random.default_rng(seed=None)

        all_idx = np.linspace(0, samples.shape[0], samples.shape[0],
                              endpoint=False, dtype=int)
        idx = rng.choice(all_idx, size=n_stars, replace=False)

        samples = samples[idx, :]
        covs = covs[idx, :, :]

    samples_folder = project_dir.joinpath('samples')
    np.savetxt(samples_folder, samples)

    density_folder = project_dir.joinpath('density')
    if not options.postprocess:
        means = options.means
        if means is not None:
            means = np.array(eval(options.means))

            if means.shape[0] != 6:
                raise StarclusterException(
                    "Passed prior means list is not 6D."
                )
        stds = options.stds
        if stds is not None:
            stds = np.array(eval(options.stds))

            if stds.shape[0] != 6:
                raise StarclusterException(
                    "Passed prior standard deviation list is not 6D."
                )

        prior = get_priors(
            bounds,
            samples,
            mean=means,
            std=stds,
            probit=False
        )

        gaussians_folder = project_dir.joinpath('gaussian_stars')
        gaussians = make_single_gaussian_mixture(
            samples, covs, bounds,
            save=True, out_folder=gaussians_folder,
            probit=False
        )

        mix = HDPGMM(
            bounds=bounds,
            prior_pars=prior,
            probit=False
        )

        draws = [
            mix.density_from_samples(gaussians)
            for _ in tqdm(range(options.n_draws))
        ]

        save_density(
            draws, folder=density_folder,
            name='density',
            ext='json'
        )
    else:
        draws = load_density(density_folder.joinpath('density.json'))

    plot_folder = project_dir.joinpath('plots')
    plot_multidim(draws,
                  out_folder=plot_folder,
                  name='density',
                  labels=[r'l',
                          r'b',
                          r'plx',
                          r'\mu_{l*}',
                          r'\mu_b',
                          r'v_{rad}'],
                  units=[r'deg',
                         r'deg',
                         r'mas',
                         r'mas\,yr^{-1}',
                         r'mas\,yr^{-1}',
                         r'km\,s^{-1}'],
                  show=False, save=True, subfolder=False,
                  n_pts=1000,
                  true_value=expected(),
                  samples=samples,
                  bounds=bounds,
                  levels=[0.5, 0.68, 0.9],
    )


if __name__ == "__main__":
    main()
