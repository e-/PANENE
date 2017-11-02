from pynene import Index
import numpy as np
import unittest
import time

class Test_Panene(unittest.TestCase):
    def test(self):
        x = np.random.ranf((100,10)).astype(np.float32)
        index = Index(x)
        self.assertIs(x, index.array)
        index.add_points(100)
        for i in range(100):
            ids, dists = index.knn_search(i, 5)
            self.assertEqual(ids.shape, (1, 5))
            self.assertEqual(dists.shape, (1, 5))

    def test_random(self):
        N = 100
        dim = 10
        dtype=np.float32
        np.random.seed(0)
        x = np.array(np.random.rand(N, dim), dtype=dtype)

        index = Index(x)
        index.add_points(N) # we must add points before querying the index

        pt = np.random.randint(N)
        pts = x[[pt]]
        idx, dists = index.knn_search_points(pts, 1, cores=1)
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], pt)

    def test_openmp(self):
        N = 10000 # must be large
        dim = 10
        dtype=np.float32

        np.random.seed(0)
        x = np.array(np.random.rand(N, dim), dtype=dtype)

        index = Index(x)
        index.add_points(N) # we must add points before querying the index
        
        for r in range(5): # make cache ready
            idx, dists = index.knn_search_points(x, 10)
            
        start = time.time()
        ids1, dists1 = index.knn_search_points(x, 10, cores=1)
        elapsed1 = time.time() - start

        start = time.time()
        ids2, dists2 = index.knn_search_points(x, 10, cores=4)
        elapsed2 = time.time() - start

        print("single thread: {:.2f} ms".format(elapsed1 * 1000))
        print("4 threads: {:.2f} ms".format(elapsed2 * 1000))

    def test_large_k(self):
        N = 100
        k = 200
        dim = 10
        dtype=np.float32

        x = np.array(np.random.rand(N, dim), dtype=dtype)
        q = np.array(np.random.rand(1, dim), dtype=dtype)

        index = Index(x)
        index.add_points(N)

        with self.assertRaises(ValueError):
            index.knn_search(0, k)

        with self.assertRaises(ValueError):
            index.knn_search_points(q, k)
    
    def test_incremental_run1(self):
        N = 100
        dim = 10
        dtype=np.float32
        x = np.array(np.random.rand(N, dim), dtype=dtype)

        index = Index(x, weights=(0.5, 0.5))
        ops = 20

        for i in range(N // ops):
            ur = index.run(ops)

            self.assertEqual(index.size(), (i + 1) * ops)
            self.assertEqual(ur['addPointResult'], ops)

if __name__ == '__main__':
    unittest.main()
