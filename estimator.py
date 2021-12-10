#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
from typing import NamedTuple

from design import Design

import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import dia_matrix

import statsmodels.api as sm


class Estimate(NamedTuple):
    estimate: np.ndarray
    std_error: np.ndarray


class Estimator(metaclass=ABCMeta):
    def __init__(self, design: Design) -> None:
        self.design = design

    @abstractmethod
    def ATE(self, X, A, YA) -> Estimate:
        pass

    @abstractmethod
    def ITE(self, X, A, YA) -> Estimate:
        pass




class DifferenceInMeans(Estimator):
    def _diff_in_means(self, Y, A):
        return np.average(Y[A == 1]) - np.average(Y[A == 0])

    def _compute_weight(self, source, target, X, weight_threshold , i ):
        w = np.exp(target.logpdf(X[i])-source.logpdf(X[i]))
        w[w>weight_threshold] = weight_threshold
        return w

    def _weighted_diff_in_means(self, source, target, X_source,  A, Y_source, weight_threshold):
        n = Y_source.shape[0]
        w = self._compute_weight(source, target, X_source, weight_threshold, np.arange(n)).reshape((-1,1))

        Yw = np.multiply(Y_source.reshape(-1,1),w)

        diff = np.dot(Yw.T,  (2*A-1))/(n/2)

        return diff[0]



    def ATE(
        self, X, A, YA
    ) -> Estimate:
        return Estimate(
            estimate=self._diff_in_means(YA, A),
            std_error= None
        )
    def weighted_ATE(
        self, source, target, X_source,  A, YA_source, weight_threshold
    ) -> Estimate:
        estimate = self._weighted_diff_in_means(source, target, X_source,  A, YA_source, weight_threshold )
        return Estimate(
            estimate= estimate, std_error = None
        )

    def ITE(self, X, A, YA) -> Estimate:
        return None



