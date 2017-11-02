from sklearn.neighbors.kde import KernelDensity
from knnkde import KNNKernelDensity

import numpy as np
N = 200
X = np.random.random((N, 2)).astype(np.float32)

def mv(n, mean, cov):
    return np.random.multivariate_normal(mean, cov, size=(n)).astype(np.float32)

X1 = np.array([np.random.normal([0.1, 0.3], [0.1, 0.3]) for x in range(N)]).astype(np.float32)
X2 = np.array([np.random.normal([0.7, 0.5], [0.2, 0.1]) for x in range(N)]).astype(np.float32)

X = np.concatenate((
    mv(N, [0.1, 0.3], [[0.01, 0], [0, 0.09]]),
    mv(N, [0.7, 0.5], [[0.04, 0], [0, 0.01]]),
    mv(N, [-0.4, -0.3], [[0.09, 0.04], [0.04, 0.02]])
    ), axis=0)

sampleN = 30
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5

kde = KNNKernelDensity(X, bandwidth = 0.2, k=10)
scores = kde.score_samples(samples.astype(np.float32))

import json

with open('result.json', 'w', encoding='utf8') as outfile:
    json.dump({
        'points': X.tolist(),
        'bins': sampleN,
        'samples': [
            (sample, score) for sample, score in zip(samples.tolist(), scores.tolist())
            ]
        }, outfile, indent=2)


    
#kde2 = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
#print(kde2.score_samples(X))

