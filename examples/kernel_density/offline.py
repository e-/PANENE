"""
This script performs density estimation for randomly generated data.
We use two different methods: scikit-learn's KernelDensity and our density 
estimation method based on KNN.
The results are stored in the "result" directory and can be visualized using 
"offline_visualizer.html". Note that you need to run a local webserver before
opening the visualizer on a web browser. It can be done by running the following 
command:

`python -m http.server`

and visit 'localhost:8000/offlie_visualizer.html' on the browser
"""
import numpy as np
import json

from sklearn.neighbors.kde import KernelDensity
from .knnkde import KNNKernelDensity

def mv(n, mean, cov):
    return np.random.multivariate_normal(mean, cov, size=(n)).astype(np.float32)

def save(path, X, sampleN, samples, scores):
    with open(path, 'w', encoding='utf8') as outfile:
        json.dump({
            'points': X.tolist(),
            'bins': sampleN,
            'samples': [
                (sample, score) for sample, score in zip(samples.tolist(), scores.tolist())
                ]
            }, outfile, indent=2)
     
N = 200
X = np.concatenate((
    mv(N, [0.1, 0.3], [[0.01, 0], [0, 0.09]]),
    mv(N, [0.7, 0.5], [[0.04, 0], [0, 0.01]]),
    mv(N, [-0.4, -0.3], [[0.09, 0.04], [0.04, 0.02]])
    ), axis=0)

sampleN = 30
samples = np.indices((sampleN + 1, sampleN + 1)).reshape(2, -1).T / sampleN * 3 - 1.5

kde = KNNKernelDensity(X)

scores = kde.score_samples(samples.astype(np.float32), k=X.shape[0])
save('result/result_knn_n.json', X, sampleN, samples, scores)

for k in [1, 5, 10, 20, 50, 100]:
    if k > X.shape[0]:
        continue
    scores = kde.score_samples(samples.astype(np.float32), k=k)
    save('result/result_knn_{}.json'.format(k), X, sampleN, samples, scores)

sci_kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
scores = np.exp(sci_kde.score_samples(samples))
save('result/result_sci.json', X, sampleN, samples, scores)
