from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import array
import struct

n = 1000000
dim = 100
centers = 100
test_n = 1000

X, y = make_blobs(n_samples=n, centers=centers, n_features=dim, shuffle=False)
print(X)
X = np.asarray(X).astype(np.float32)

def write_floats(floats, outfile):
    float_arr = array.array('d', floats)
    s = struct.pack('f' * len(float_arr), *float_arr)
    outfile.write(s)

with open("blob.original.bin", "wb") as outf:
    for i in range(n):
        write_floats(X[i], outf)

np.random.shuffle(X)

with open("blob.shuffled.bin", "wb") as outf:
    for i in range(n):
        write_floats(X[i], outf)

test, _ = make_blobs(n_samples=test_n, centers=test_n, n_features=dim)

with open("test.bin", "wb") as outf:
    for i in range(test_n):
        write_floats(test[i], outf)
