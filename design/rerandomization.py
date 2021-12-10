#! /usr/bin/python3
import numpy as np

from design.design import Design
from scipy.spatial.distance import mahalanobis
import copy



class ReRandomization(Design):
    def __init__(self, weight_threshold, pr_accept = 0.1, balance_on_target = False, target = None, source = None, A_list = None):
        super(ReRandomization, self).__init__()

        self.pr_accept = pr_accept


        self.target = target
        self.source = source
        self.balance_on_target = balance_on_target
        self.A_list = A_list
        self.weight_threshold = weight_threshold

        self.num_assignments = 100

    def fit(self, X: np.ndarray) -> None:
        if len(X.shape)==1:
            X.reshape(-1,1)
        self.X = X

        if self.balance_on_target:
            w = self.compute_weight(np.arange(X.shape[0])).reshape((-1,1))
            self.Xw = np.multiply(X,w)
        else:
            self.Xw = X
        self.N = self.X.shape[0]
        N = self.X.shape[0]
        self.inv_cov_w = np.linalg.inv(self.Xw.T @ (N*np.eye(N) - np.ones((N,N))) @ self.Xw)/(N-1)

        n2 = int(N / 2)
        A = np.array([0] * n2 + [1] * (N - n2))


        np.random.shuffle(A)
        self.draws = [A]


        if self.pr_accept < 0.99:

            nrand = int(self.num_assignments / self.pr_accept)
            A_list = np.array([A[np.random.permutation(np.arange(N))] for i in range(nrand)])
            distance_list = np.apply_along_axis(self.balance_distance, 1, A_list)
            partial_order = np.argpartition(distance_list, self.num_assignments)
            self.draws = A_list[partial_order][:self.num_assignments]

        if self.pr_accept >= 0.99:
            assert len(self.draws)==1

    def compute_weight(self, i):
        w= np.exp(self.target.logpdf(self.X[i])-self.source.logpdf(self.X[i]))
        w[w> self.weight_threshold] = self.weight_threshold
        return w


    def balance_distance(self, A):

        #this formula is only for the case when the sizes of the treatment and control group are equal
        assert sum(A) == sum(1-A)
        diff = self.Xw.T @ (2*A-1) / (self.N/2)


        d = diff @ self.inv_cov_w @ diff.T

        return d


    def assign(self, X: np.ndarray):
        idx = np.random.choice(range(len(self.draws)), 1).item()
        return self.draws[idx] #(np.array(self.draws[idx])).astype(int)





