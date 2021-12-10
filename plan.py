#! /usr/bin/python3

from abc import ABCMeta, abstractmethod
import time
from typing import Type

from design import Design
from estimator import Estimator
from evaluator import Evaluator

import numpy as np
import pandas as pd

class Plan(metaclass=ABCMeta):
    def __init__(self):
        self.evaluators = {}
        self.designs = {}

    def add_design(self, design_name, design_class: Type[Design], estimator_class: Type[Estimator], design_kwargs = None):
        self.designs[design_name] = (design_class, estimator_class, design_kwargs)

    def add_evaluator(self, evaluator_name: str, evaluator: Evaluator):
        self.evaluators[evaluator_name] = evaluator()

    def add_env(self, dgp_factory, seed,  X_source_dist = None,  X_target_dist = None):
        np.random.seed(seed)
        dgp_source = dgp_factory.create_dgp(X_dist = X_source_dist)
        self.X_source = dgp_source.X
        self.Y0_source = dgp_source.Y([0] * dgp_source.n)
        self.Y1_source = dgp_source.Y([1] * dgp_source.n)
        self.ITE_source = dgp_source.ITE()
        self.ATE_source = dgp_source.ATE()
        self.source = dgp_source.X_dist



        dgp_target = dgp_factory.create_dgp(X_dist = X_target_dist)

        self.ITE_target = dgp_target.ITE()
        self.ATE_target = dgp_target.ATE()

        self.target = dgp_target.X_dist
        self.weight_target = self.target

    def use_weighted_estimator(self, weighted_estimator = False):
        if not weighted_estimator:
            self.weight_target = self.source


    def execute(self,  design_name, weight_threshold, weighted_estimator = False):

        results = []
        design_class, estimator_class, design_kwargs = self.designs[design_name]

        def make_row(name, value):
            return pd.DataFrame({"design": [design_name + "_weighted-estimator" + str(weighted_estimator)], "metric": [name], "value": [value]})
        time_start = time.time()

        if design_kwargs is None:
            design_kwargs = {}
        design = design_class(**design_kwargs)
        design.fit(self.X_source)
        A = design.assign(self.X_source)
        time_end = time.time()
        time_elapsed = time_end - time_start
        results.append(make_row("time_design", time_elapsed))
        YA_source = np.where(A==1, self.Y1_source, self.Y0_source)
        time_start = time.time()
        estimator = estimator_class(design)

        ITEhat = estimator.ITE(self.X_source, A, YA_source)

        if weighted_estimator:
            ATEhat = estimator.weighted_ATE(self.source, self.weight_target, self.X_source, A, YA_source, weight_threshold = weight_threshold)
        else:
            ATEhat = estimator.ATE(self.X_source, A, YA_source)

        time_end = time.time()
        time_elapsed = time_end - time_start
        results.append(make_row("time_estimation", time_elapsed))
        for name, evaluator in self.evaluators.items():

            val = evaluator.evaluate(self.X_source, self.Y0_source, self.Y1_source, self.ATE_target, self.ITE_target, A, YA_source, ATEhat, ITEhat)
            results.append(make_row(name, val))
        return pd.concat(results)
