"""
`python -m http.server`
"""

import numpy as np
import json

from sklearn.neighbors.kde import KernelDensity
from knnkde import KNNKernelDensity
import time

def mv(n, mean, cov):
    return np.random.multivariate_normal(mean, cov, size=(n)).astype(np.float32)

def save(path, sampleN, samples, scores, running, bandwidth, k, iteration, all_sample):
    print(running)
    with open(path, 'w', encoding='utf8') as outfile:
        json.dump({
            'time': running,
            'bandwidth': bandwidth,
            'k': k,
            'bins': sampleN,
            'iteration': iteration,
            'all_sample': all_sample,
            'samples': [
                (sample, score) for sample, score in zip(samples.tolist(), scores.tolist())
                ]
            }, outfile)
     
N = 350000 # the number of points
N2 = 300000
X1 = np.concatenate((
    mv(N, [0.1, 0.3], [[0.01, 0], [0, 0.09]]),
    mv(N, [0.7, 0.5], [[0.04, 0], [0, 0.01]]),
    ), axis=0)
X2 = mv(N2, [-0.4, -0.3], [[0.09, 0.04], [0.04, 0.02]])

np.take(X1, np.random.permutation(X1.shape[0]), axis=0, out=X1)

X = np.concatenate((X1, X2), axis=0)

sampleN = 20 # resolution of density maps
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5

# using scikit-learn
bw = 0.2
start = time.time()
sci_kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X)
scores = np.exp(sci_kde.score_samples(samples))
total = time.time() - start
print(total)
save('result/ground_truth.json', sampleN, samples, scores, total, bw, -1, -1, True)

# using knn

ks = [1500, 3000, 5000]
bws = [0.2]
ops = 100000

with open('result/X1_samples.json', 'w', encoding='utf8') as outfile:
    X1_samples = X1[np.random.randint(X1.shape[0], size=100)]
    json.dump(X1_samples.tolist(), outfile)

with open('result/X_samples.json', 'w', encoding='utf8') as outfile:
    X_samples = np.concatenate((X1_samples, X2[np.random.randint(X2.shape[0], size=50)]), axis=0)
    json.dump(X_samples.tolist(), outfile)

meta = []

for k in ks:
    if k > X.shape[0]:
        continue

    for bw in bws:
        start = time.time()
        kde = KNNKernelDensity(X, online=True)
        inserted = 0
        iteration = 0
        
        total = 0

        while inserted < len(X):
            res = kde.run(ops)
            inserted = res['numPointsInserted']
            print("Iter #{}: {} / {}".format(iteration, inserted, X.shape[0]))

            if inserted >= k:
                scores = kde.score_samples(samples.astype(np.float32), k=k, bandwidth=bw)
                total += time.time() - start
                save('result/knn_{}_{}_{}.json'.format(bw, k, iteration),
                        sampleN, samples, scores, total, bw, k, iteration, inserted >= 2 * N)
                start = time.time()

            iteration += 1

        meta.append({
            'k': k,
            'bandwidth': bw,
            'max_iter': iteration
        })
        
with open('result/metadata.json', 'w', encoding='utf8') as outfile:
    json.dump(meta, outfile)
