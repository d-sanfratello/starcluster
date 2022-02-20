import matplotlib.pyplot as plt
import numpy as np

from scipy.special import erf


class SkewPlots:
    def plot_fixed_relative(self, rel_error=0.10, L=50, N=20,
                            min_parallax=1e-1, max_parallax=10,
                            sigma_lim=None):
        mu = np.logspace(np.log10(min_parallax), np.log10(max_parallax), N)
        sigma = mu * rel_error

        if sigma_lim is not None:
            sigma = np.where(sigma <= sigma_lim, sigma, sigma_lim)

        r = np.linspace(1e-5, L, 1000)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f'Test with fixed relative error at '
                     f'{rel_error:.3f} (normalized)\n'
                     f'Sigma lim set to {sigma_lim}')
        ax.grid()
        for m, s in zip(mu[:10], sigma[:10]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='-',
                    label=f'mu = {m:.3f} +- {s:.1f} mas')
        for m, s in zip(mu[10:], sigma[10:]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='--',
                    label=f'mu = {m:.3f} +- {s:.1f} mas')

        ax.set_xlabel('$r$ [kpc]')
        ax.legend(loc='best')

    def plot_fixed_loc(self, mu=5, L=50, N=20):
        mu_over_err = np.linspace(2, 1e3, N)
        sigma = mu / mu_over_err

        r = np.linspace(1e-5, L, 1000)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f'Test with fixed mu at {mu:.3f} mas (normalized)')
        ax.grid()
        for m, s in zip(mu_over_err[:10], sigma[:10]):
            distr = self.__distribution(r, mu, s, L)
            ax.plot(r, distr/max(distr), linestyle='-',
                    label=f'rel_sigma = {1/m:.2e}')
        for m, s in zip(mu_over_err[10:], sigma[10:]):
            distr = self.__distribution(r, mu, s, L)
            ax.plot(r, distr/max(distr), linestyle='--',
                    label=f'rel_sigma = {1/m:.2e}')

        ax.set_xlabel('$r$ [kpc]')
        ax.legend(loc='best')

    @staticmethod
    def __distribution(r, mu, sigma, L=1e2):
        normalization = np.sqrt(2/np.pi) / (sigma * (1 - erf((1 / L - mu)/(
                sigma*np.sqrt(2)))))

        return normalization * (1/r**2) * np.exp(-(1/r - mu)**2 / (2*sigma**2))
