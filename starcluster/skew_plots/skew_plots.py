import matplotlib.pyplot as plt
import numpy as np

from scipy.special import erf


class SkewPlots:
    def plot_fixed_relative(self, mu_over_err=10, L=1e2, N=20):
        mu = np.linspace(1e-5, 1, N)
        sigma = mu / mu_over_err

        r = np.linspace(1e-10, L, 1000)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f'Test with fixed relative error at '
                     f'{1 / mu_over_err:.3f} (normalized)')
        ax.grid()
        for m, s in zip(mu[:10], sigma[:10]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='-',
                    label=f'mu = {m:.3f}')
        for m, s in zip(mu[10:], sigma[10:]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='--',
                    label=f'mu = {m:.3f}')

        ax.legend(loc='best')
        plt.show()

    def plot_fixed_loc(self, mu=1e-3, L=1e2, N=20):
        mu_over_err = np.linspace(1e-4, 0.5, N)
        sigma = mu / mu_over_err

        r = np.linspace(1e-10, L, 1000)

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(f'Test with fixed mu at {mu:.3f} (normalized)')
        ax.grid()
        for m, s in zip(mu_over_err[:10], sigma[:10]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='-',
                    label=f'rel_sigma = {1/m:.3f}')
        for m, s in zip(mu_over_err[10:], sigma[10:]):
            distr = self.__distribution(r, m, s, L)
            ax.plot(r, distr/max(distr), linestyle='--',
                    label=f'rel_sigma = {1/m:.3f}')

        ax.legend(loc='best')
        plt.show()

    @staticmethod
    def __distribution(r, mu, sigma, L=1e2):
        normalization = np.sqrt(2/np.pi) / (sigma * (1 - erf((1 / L - mu)/(
                sigma*np.sqrt(2)))))

        return normalization * (1/r**2) * np.exp(-(1/r - mu)**2 / (2*sigma**2))
