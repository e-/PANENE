from pynene import Index
import numpy as np
import unittest

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

if __name__ == '__main__':
    unittest.main()
