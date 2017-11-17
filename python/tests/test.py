from pynene import Index
import numpy as np
import unittest
import time

def random_vectors(n=100, d=10, dtype=np.float32):
    return np.array(np.random.rand(n, d), dtype=dtype)

class Test_Panene(unittest.TestCase):
    def test_return_shape(self):
        x = random_vectors()

        index = Index(x)
        self.assertIs(x, index.array)

        index.add_points(x.shape[0])

        for i in range(x.shape[0]):
            ids, dists = index.knn_search(i, 5)
            self.assertEqual(ids.shape, (1, 5))
            self.assertEqual(dists.shape, (1, 5))

    def test_random(self):
        x = random_vectors()

        index = Index(x)
        index.add_points(x.shape[0]) # we must add points before querying the index

        pt = np.random.randint(x.shape[0])
        pts = x[[pt]]

        idx, dists = index.knn_search_points(pts, 1, cores=1)

        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], pt)

    def test_openmp(self):
        N = 10000 # must be large enough
        
        x = random_vectors(N)

        index = Index(x)
        index.add_points(x.shape[0]) # we must add points before querying the index
        
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
        x = random_vectors()
        q = random_vectors(1)
        k = x.shape[0] + 1 # make k larger than # of vectors in x

        index = Index(x)
        index.add_points(x.shape[0])

        with self.assertRaises(ValueError):
            index.knn_search(0, k)

        with self.assertRaises(ValueError):
            index.knn_search_points(q, k)
    
    def test_incremental_run1(self):
        x = random_vectors()

        index = Index(x, w=(0.5, 0.5))
        ops = 20

        for i in range(x.shape[0] // ops):
            ur = index.run(ops)

            self.assertEqual(index.size(), (i + 1) * ops)
            self.assertEqual(ur['addPointResult'], ops)

    def test_incremental_run2(self):
        n = 1000
        k = 20
        ops = 100
        test_n = 30

        x = random_vectors(n)
        test_points = random_vectors(test_n)

        index = Index(x)
        
        for i in range(n // ops):
            ur = index.run(ops)
           
            ids1, dists1 = index.knn_search_points(test_points, k, checks = 100)
            ids2, dists2 = index.knn_search_points(test_points, k, checks = 1000)            
            
            """
            The assertion below always holds since the latter search checks a larger number of nodes and the search process is deterministic
            """
            self.assertEqual(np.sum(dists1 >= dists2), test_n * k)

    def test_check_x_type(self):
        x = random_vectors()
        index = Index(x)
        index.add_points(len(x))
        index.knn_search_points(x, 10)

        with self.assertRaises(ValueError):
            x = random_vectors(dtype=np.int32)
            index = Index(x)
            index.add_points(len(x))
            index.knn_search_points(x, 10)

        with self.assertRaises(ValueError):
            x = np.random.rand(100, 10)
            index = Index(x)
            index.add_points(len(x))
            index.knn_search_points(x, 10)

if __name__ == '__main__':
    unittest.main()
