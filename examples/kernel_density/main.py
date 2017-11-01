from sklearn.neighbors.kde import KernelDensity
from knnkde import KNNKernelDensity

import numpy as np
X = np.random.random((100, 2))

kde1 = KNNKernelDensity(X)
print(kde1.score_samples(X))

#kde2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
#print(kde2.score_samples(X))

