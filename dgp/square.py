#! /usr/bin/python3

from .dgp import DGP

import numpy as np
from scipy.stats import multivariate_normal


class SquareDGP(DGP):
    def __init__(self, N=100, pate=1, ice_sd=0, noise_mu=0, noise_sd=1, X_dist=None, num_covariates=2):


        if not X_dist:
            X_dist = multivariate_normal(0 * np.ones(num_covariates),  np.eye(num_covariates))

        X_mean = X_dist.mean


        assert num_covariates == X_mean.shape[0]
        self._X_dist = X_dist
        self._X = self._X_dist.rvs(size=N).reshape(N,num_covariates)

        self.y0 = np.sum(np.square(self._X), axis = 1)

        self.y1 = 2 * self.y0 + np.random.normal(
            size=N, loc=noise_mu, scale=noise_sd
        )


        self.y0 += np.random.normal(
            size=N, loc=noise_mu, scale=noise_sd
        )


        self._ATE = (np.trace(X_dist.cov + np.outer(X_dist.mean, X_dist.mean)))



        super(SquareDGP, self).__init__(N=N)

    def ATE(self):

        return self._ATE

    def Y(self, A: np.ndarray) -> np.ndarray:
        return np.where(np.array(A).flatten() == 1, self.y1, self.y0)

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def X_dist(self):
        return self._X_dist


class SquareFactory(object):
    def __init__(self, N):
        self.N = N

    def create_dgp(self, X_dist = None):
        if X_dist is not None:
            num_covariates = X_dist.mean.shape[0]
        else:
            num_covariates = 2
        return SquareDGP(N=self.N, X_dist = X_dist, num_covariates = num_covariates)
