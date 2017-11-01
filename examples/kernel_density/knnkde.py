# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>

import numpy as np
from pynene import Index

class KNNKernelDensity():
    def __init__(self, X, k=20, bandwidth=1.0):
        self.X = X
        self.index = Index(X)
        self.index.add_points(X.shape[0])
        self.k = k
        self.bandwidth = bandwidth

    def add_points(self, n):
        self.index.add_points(n)

    def score_samples(self, X):
        print("wer")
        ids, dists = self.index.knn_search_points(X.astype(np.float32), k=self.k)
        print("werr")
        n = X.shape[0]
        scores = np.zeros(n)

#        for i in range(n):
#            dists.
        print(ids)
        print(dists)

