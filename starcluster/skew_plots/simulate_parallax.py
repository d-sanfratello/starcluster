import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path
from scipy.stats import skew, kurtosis, norm


class SimulateParallax:
    @staticmethod
    def simulate(seed=1234, *,
                 dist_true=10,  # kpc
                 relative_error=0.1, norm_sigma_target=1,
                 n_stars=1e6):
        rng = np.random.default_rng(seed=seed)

        n_stars = int(n_stars)

        true_parallax = 1 / dist_true  # mas

        sigma = relative_error * true_parallax  # mas
        measured_parallax = rng.normal(loc=true_parallax,
                                       scale=sigma,
                                       size=n_stars)  # mas

        q = np.percentile(measured_parallax, [5, 95])  # mas
        q_dist = 1 / q  # kpc

        recovered_dist = 1 / measured_parallax  # kpc

        std = np.std(recovered_dist, ddof=1)  # kpc

        error_over_recovered = (recovered_dist - dist_true) / std

        ratio_good = np.argwhere(error_over_recovered
                                 <= norm_sigma_target).shape[0] / n_stars

        return {
            'n_stars': n_stars,
            'relative_error': relative_error,
            'true_dist': dist_true,  # kpc
            'true_parallax': true_parallax,  # mas
            'meas_parallax': measured_parallax,  # mas
            'recovered_dist': recovered_dist,  # kpc
            'recovered_std': std,  # kpc
            'error_over_recovered': error_over_recovered,
            'relative_error_target': norm_sigma_target,
            'ratio_good': ratio_good,
            '5-95_perc': q_dist  # kpc
        }

    @staticmethod
    def plot_distributtion(sim, figsize=None):
        n_stars = sim['n_stars']
        recovered = sim['recovered_dist']
        mean = recovered.mean()
        std = sim['recovered_std']
        true_dist = sim['true_dist']
        rel_parallax = sim['relative_error']
        q = sim['5-95_perc']

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.grid()

        x = np.linspace(recovered.min(), recovered.max(), 1000)
        # normal = norm(loc=true_dist,
        #               scale=std)

        n, bins = ax.hist(recovered, bins='auto', density=True)[:-1]

        mode_idx = np.argmax(n)
        mode = (bins[mode_idx] + bins[mode_idx+1]) / 2
        ax.axvline(mode, 0, 1,
                   color='blue', linestyle='solid', linewidth=1,
                   label=f'$mode_r$ = {mode:.3f} kpc')
        ax.axvline(mean, 0, 1,
                   color='cyan', linestyle='solid', linewidth=1,
                   label=f'$\mu_r$ = {mean:.3f} kpc')
        # ax.plot(x, normal.pdf(x),
        #         color='black', linestyle='dashed', linewidth=1,
        #         label='$N(r | r_t, \sigma_d)$')
        ax.axvline(true_dist, 0, 1,
                   color='red', linestyle='dashed', linewidth=1,
                   label=f'$median_r = true_r$ = {true_dist:.3f} kpc')

        normal_approx = norm(loc=mean,
                             scale=std)
        q_approx = normal_approx.interval(0.9)
        out95perc = recovered[recovered > q_approx[1]].shape[0]
        in5perc = recovered[recovered <= q_approx[0]].shape[0]
        ax.axvline(q_approx[0], 0, 1,
                   color='orange', linestyle='solid', linewidth=1,
                   label='5-95% in approx. gaussian')
        ax.axvline(q_approx[1], 0, 1,
                   color='orange', linestyle='solid', linewidth=1)

        fig.suptitle(f'$\mu_r$ = {mean:.3f} kpc; '
                     f'$\sigma_r \sim$ {std:.3f} kpc\n'
                     f'$r_t$ = {true_dist} kpc; $\sigma_\pi$ ='
                     f' {100 * rel_parallax:.1f}%\n'
                     f'Data outside 95% (approx. from $N(r | \mu_r, '
                     f'\sigma_r)$) = '
                     f'{100 * out95perc / n_stars:.3f}%\n'
                     f'Data within 5% (approx. from $N(r | \mu_r, '
                     f'\sigma_r)$) = '
                     f'{100 * in5perc / n_stars:.3f}%\n'
                     f'skew:{skew(recovered, bias=False):.3f}    '
                     f'kurt:{kurtosis(recovered, bias=False):.3f}; '
                     f'{n_stars} points')

        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('Probability')

        ax.legend(loc='best')

        plt.tight_layout()
        path = Path(os.getcwd()).joinpath('../../skew_plots')
        file = path.joinpath(f'{true_dist:.0f}kpc_'
                             f'{100*rel_parallax:.0f}percent')
        plt.savefig(file,
                    format='pdf')


if __name__ == '__main__':
    simulation = SimulateParallax()

    sim = simulation.simulate(dist_true=1,
                              relative_error=0.2, norm_sigma_target=1,
                              n_stars=1e8)

    simulation.plot_distributtion(sim=sim, figsize=(10, 8))
