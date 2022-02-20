import matplotlib.pyplot as plt
import numpy as np

from scipy.special import erf


class SkewTest:
    def test_relative(self, mu_over_err=10, L=10e3):
        mu = np.linspace(1e-6, 1, 20)
        sigma = mu / mu_over_err

        r = np.linspace(0, L, 1000)

        fig = plt.figure()
        fig.title = f'Test with fixed relative error at {1 / mu_over_err}'

        ax = fig.gca()
        ax.grid()
        for m, s in mu[:10], sigma[:10]:
            ax.plot(r, self.__distribution(r, m, s, L), linestyle='-',
                    label=f'{m}')
        for m, s in mu[10:], sigma[10:]:
            ax.plot(r, self.__distribution(r, m, s, L), linestyle='--',
                    label=f'{m}')

        ax.legend(loc='best')
        plt.show()

    def test_loc(self, mu=1e-2, L=10e3):
        mu_over_err = np.linspace(1e-4, 0.5, 20)
        sigma = mu / mu_over_err

        r = np.linspace(0, L, 1000)

        fig = plt.figure()
        fig.title = f'Test with fixed mean at {1 / mu}'

        ax = fig.gca()
        ax.grid()
        for m, s in mu_over_err[:10], sigma[:10]:
            ax.plot(r, self.__distribution(r, m, s, L), linestyle='-',
                    label=f'{m}')
        for m, s in mu_over_err[10:], sigma[10:]:
            ax.plot(r, self.__distribution(r, m, s, L), linestyle='--',
                    label=f'{m}')

        ax.legend(loc='best')
        plt.show()

    @staticmethod
    def __distribution(r, mu, sigma, L=10e3):
        normalization = np.sqrt(2/np.pi) / (sigma * (1 - erf((1 / L - mu)/(
                sigma*np.sqrt(2)))))

        return normalization * (1/r**2) * np.exp(-(1/r - mu)**2 / (2*sigma**2))
