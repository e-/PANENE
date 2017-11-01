from sklearn.neighbors.kde import KernelDensity
from progressive_kde import ProgressiveKernelDensity

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

kde1 = ProgressiveKernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
print(kde1.score_samples(X))

kde2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
print(kde2.score_samples(X))

