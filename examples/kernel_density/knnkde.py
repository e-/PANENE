# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>

import numpy as np
from pynene import Index

class KNNKernelDensity():
    SQRT2PI = np.sqrt(2 * np.pi)

    def __init__(self, X):
        self.X = X
        self.index = Index(X)
        self.index.add_points(len(X))

    def add_points(self, n):
        self.index.add_points(n)

    def score_samples(self, X, k=10, bandwidth=0.2):
        _, dists = self.index.knn_search_points(X, k=k)
        scores = self._gaussian_score(dists, bandwidth) / k
        return scores

    def _gaussian_score(self, dists, bandwidth):
        logg = -0.5 * (dists / bandwidth) ** 2
        g = np.exp(logg) / bandwidth / self.SQRT2PI
        return g.sum(axis=1)
