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
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(dists), 5)
        print(len(x))

if __name__ == '__main__':
    unittest.main()
