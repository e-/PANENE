# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>

import numpy as np
import sys
from pynene import Index

class KNNRegressor():
    def __init__(self, X, y, n_neighbors=5, weights='uniform', online=False):
        self.X = X
        self.y = y
        self.index = Index(X)
        self.n_neighbors = n_neighbors
        self.weights = weights

        if not online: # if offline
            self.index.add_points(len(X))

    def run(self, ops):
        return self.index.run(ops)

    def predict(self, X):
        indices, dists = self.index.knn_search_points(X, k=self.n_neighbors)
        weights = self._get_weights(dists)

        if self.weights == 'uniform':
            y_pred = np.mean(self.y[indices], axis = 1)
        else:
            y_pred = np.empty((X.shape[0], self.y.shape[1]))
            denom = np.sum(weights, axis = 1)

            for j in range(self.y.shape[1]):
                num = np.sum(self.y[indices, j] * weights, axis=1)
                y_pred[:, j] = num / denom
                    
        if self.y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def _get_weights(self, dists):
        if self.weights == 'uniform':
            return None
        
        for i, dist in enumerate(dists):
            if 0. in dist:
                dists[i] = dist == 0.
            else:
                dists[i] = 1. / dist
        
        return dists
