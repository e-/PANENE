# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>

import numpy as np
from pynene import Index

class KNNKernelDensity():
    SQRT2PI = 2.50662827463 # sqrt(2 * PI)

    def __init__(self, X):
        self.X = X
        self.index = Index(X)
        self.index.add_points(X.shape[0])

    def add_points(self, n):
        self.index.add_points(n)

    def score_samples(self, X, k=10, bandwidth=0.2):
        ids, dists = self.index.knn_search_points(X, k=k)

        n = X.shape[0]
        scores = np.zeros(n)

        for i in range(n):
            scores[i] = self._gaussian_score(dists[i], k, bandwidth) / k

        return scores

    def _gaussian_score(self, dists, k, bandwidth):
        g = 0

        logg = -0.5 * (dists / bandwidth) ** 2

        g = np.exp(logg) / bandwidth / self.SQRT2PI

        return g.sum()
