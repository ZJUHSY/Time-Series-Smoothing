import numpy as np
import pandas as pd
from scipy.special import comb
import os
import particles
from particles import state_space_models as ssm
from particles import distributions as dists

PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, './data/')
IMAGE_DIR = os.path.join(PROJECT_DIR, './images/')


def read_all_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'all_stocks_2006-01-01_to_2018-01-01.csv'))


def mae(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / np.sum(np.abs(y))


class ma_smoother:
    def __init__(self, h, data, by_val='Close', by_owa='Volume'):
        """

        :param h:window size
        :param data: data
        :param by_val: values of the data for smoothing
        """
        self.h = h
        self.data = data
        self.len = len(data)
        self.owa_val = self.data[by_owa]
        self.series = self.data[by_val]

    def get_weight(self, n):
        return np.array(([0] * (n - self.h) + [1 / self.h] * self.h))

    def get_variance(self, n):
        return np.sum(self.get_weight(n) ** 2)

    def age(self, n):
        return n - np.sum(self.get_weight(n) * np.array(range(1, n + 1)))

    def get_smoothed_res(self, n, owa=False):
        if n < self.h:
            return self.series[n]
        else:
            if owa:
                return np.sum(self.get_weight(n) * self.get_owa_series(n))
            else:
                return np.sum(self.get_weight(n) * self.series[:n])

    def get_smoothed_pred(self, owa=False):
        return [self.get_smoothed_res(i, owa) for i in range(self.len)]

    '''
    owa results for getting weight and smoothing
    '''

    def get_owa_series(self, n):
        perm_data = self.series[:n]
        owa_arr = self.owa_val[:n]
        owa_dict = [(perm_data[i], owa_arr[i]) for i in range(n)]
        owa_dict = sorted(owa_dict, key=lambda x: x[1], reverse=True)
        owa_series = np.array([i[0] for i in owa_dict])
        return owa_series


class exp_smoother(ma_smoother):
    def __init__(self, h, data, alpha, by_val='Close', by_owa='Volume'):
        """

        :param h:
        :param data:
        :param alpha: extra parameters for initializing exponential smoothers
        :param by_val: values of the data for smoothing
        """

        super().__init__(h, data, by_val, by_owa)
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def get_weight(self, n):
        p1 = [0] * (n - self.h)
        p2 = [self.alpha * (self.beta ** (self.h - i - 1)) for i in range(self.h)]
        res = np.array(p1 + p2)
        res = res / np.sum(res)
        return res

    # def get_variance(self, n):
    #     return np.sum(self.get_weight(n) ** 2)
    #
    # def age(self, n):
    #     n - np.sum(self.get_weight(n) * np.array(range(1, n + 1)))


class ld_smoother(ma_smoother):
    def __init__(self, h, data, by_val='Close', by_owa='Volume'):
        super().__init__(h, data, by_val, by_owa)

    def get_weight(self, n):
        p1 = [0] * (n - self.h)
        T = comb(self.h + 1, 2)
        p2 = [i / T for i in range(1, self.h + 1)]
        res = np.array(p1 + p2)
        return res


class sqd_smoother(ma_smoother):
    def __init__(self, h, data, by_val='Close', by_owa='Volume'):
        super().__init__(h, data, by_val, by_owa)

    def get_weight(self, n):
        p1 = [0] * (n - self.h)
        T = (self.h * (self.h + 1) * (1 + 2 * self.h)) / 6
        p2 = [i ** 2 / T for i in range(1, self.h + 1)]
        res = np.array(p1 + p2)
        return res


class inv_smoother(ma_smoother):
    def __init__(self, h, data, by_val='Close', by_owa='Volume'):
        super().__init__(h, data, by_val, by_owa)

    def get_weight(self, n):
        p1 = [0] * (n - self.h)
        p2 = [sum([1 / i for i in range(j, self.h + 1)]) / self.h for j in range(self.h, 0, -1)]
        res = np.array(p1 + p2)
        return res


class mc_smoother(ma_smoother):
    def __init__(self, h, data, mu=-1., rho=.9, sigma=.1):
        super().__init__(h, data)
        self.series = np.log(self.series)
        self.mu = mu
        self.rho = rho
        self.sigma = sigma

    def get_model(self, n):
        u_y = super().get_smoothed_res(n)  # get moving average
        return StochVol(mu=-1., rho=.9, sigma=.1, avg_y=u_y)

    def get_smoothed_res(self, n, owa=False):
        model = self.get_model(n)
        fk_boot = ssm.Bootstrap(ssm=model, data=[np.array(self.series[n])])
        alg_with_mom = particles.SMC(fk=fk_boot, N=100, moments=True)
        alg_with_mom.run()
        mu, sigma = model.avg_y, np.exp(alg_with_mom.summaries.moments[0]['mean']/2) # scale
        res = np.mean(np.random.normal(mu, sigma, 100))
        return res





'''
stock volatility model\\for simplicty
'''


class StochVol(ssm.StateSpaceModel):
    def __int__(self, mu, rho, sigma, avg_y):
        """

        :param mu: paras for volatility
        :param rho: paras for volatility
        :param sigma: paras for volatility
        :param avg_y: mean log y
        :return:
        """

        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = sigma
        self.avg_y = avg_y

    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho ** 2))

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=self.avg_y, scale=np.exp(x))
