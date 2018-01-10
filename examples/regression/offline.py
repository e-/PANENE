import numpy as np
import json

from sklearn.neighbors import KNeighborsRegressor
from knnreg import KNNRegressor

def save(path, X, y, sampleN, samples, y_pred):
    with open(path, 'w', encoding='utf8') as outfile:
        json.dump({
            'points': [
                (x, y) for x, y in zip(X.tolist(), y.tolist())
            ],
            'sampleN': sampleN,
            'samples': [
                (sample, score) for sample, score in zip(samples.tolist(), y_pred.tolist())
                ]
            }, outfile, indent=2)
     
N = 300
X = np.expand_dims(np.linspace(0, np.pi * 2, N), axis = 1).astype(np.float32)
y = np.sin(X).astype(np.float32)

X += np.random.normal(0, 0.1, X.shape)
y += np.random.normal(0, 0.1, y.shape)

sampleN = 100
samples = np.expand_dims(np.linspace(0, np.pi * 2, sampleN), axis = 1).astype(np.float32)

n_neighbors = 20

# numpy version

neigh = KNeighborsRegressor(n_neighbors=n_neighbors) #, weights='distance')
neigh.fit(X, y)
y_pred1 = neigh.predict(samples)

save('result1.json', X.reshape(N), y.reshape(N),
        sampleN, samples.reshape(sampleN), y_pred1.reshape(sampleN))

# PANENE version

neigh = KNNRegressor(X, y, n_neighbors=n_neighbors) #, weights='distance')
y_pred2 = neigh.predict(samples)

save('result2.json', X.reshape(N), y.reshape(N),
        sampleN, samples.reshape(sampleN), y_pred2.reshape(sampleN))
