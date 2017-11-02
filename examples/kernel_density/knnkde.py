# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>

import numpy as np
from pynene import Index

class KNNKernelDensity():
    SQRT2PI = 2.50662827463 # sqrt(2 * PI)

    def __init__(self, X, k=20, bandwidth=1.0):
        self.X = X
        self.index = Index(X)
        self.index.add_points(X.shape[0])
        self.k = k
        self.bandwidth = bandwidth

    def add_points(self, n):
        self.index.add_points(n)

    def score_samples(self, X):
        ids, dists = self.index.knn_search_points(X, k=self.k)

        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            scores[i] = self._gaussian_score(dists[i]) / self.k

        return scores

    def _gaussian_score(self, dists):
        g = 0

        logg = -0.5 * (dists / self.bandwidth) ** 2

        g = np.exp(logg) / self.bandwidth / self.SQRT2PI

        return g.sum()
