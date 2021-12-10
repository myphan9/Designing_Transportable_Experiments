#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Union, List

import numpy as np
from scipy.stats import norm

from dgp import DGP


NORMAL_QUANTILE = norm.ppf(0.975)


class Evaluator(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:
        pass


class ATEError(Evaluator):
    def evaluate(self, X, Y0, Y1, ATE, ITE, A, YA, ATEhat, ITEhat) -> Number:

        return ATE - ATEhat.estimate


