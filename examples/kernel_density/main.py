from sklearn.neighbors.kde import KernelDensity
from knnkde import KNNKernelDensity

import numpy as np
N = 500
X = np.random.random((N, 2)).astype(np.float32)

X1 = np.array([np.random.normal([0.1, 0.3], [0.1, 0.3]) for x in range(N)]).astype(np.float32)

X2 = np.array([np.random.normal([0.7, 0.5], [0.2, 0.1]) for x in range(N)]).astype(np.float32)

X = np.concatenate((X1, X2), axis=0)

sampleN = 30
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5
kde = KNNKernelDensity(X, bandwidth = 0.2)
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

