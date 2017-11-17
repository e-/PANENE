from pynene import KNNTable
import numpy as np
import unittest


def random_vectors(n=100, d=10, dtype=np.float32):
    return np.array(np.random.rand(n, d), dtype=dtype)

class Test_KNNTable(unittest.TestCase):
    def test_table(self):
        n = 100
        d = 10
        k = 5
        array = random_vectors(n, d)
        neighbors = np.zeros((n, 5), dtype=np.int64) # with np.in32, it is not optimized
        distances = np.zeros((n, 5), dtype=np.float32)
        table = KNNTable(array, k, neighbors, distances)
        
        self.assertTrue(table != None)
        updates = table.run(10)
        #print(updates)
        self.assertEqual(len(updates), 10)

    def test_incremental_run(self):
        n = 1000
        ops = 100
        k = 20
    
        neighbors = np.zeros((n, k), dtype=np.int64)
        distances = np.zeros((n, k), dtype=np.float32)

        x = random_vectors(n)
        
        table = KNNTable(x, k, neighbors, distances)

        for i in range(n // ops):
            ur = table.run(ops)
            
            for nn in range(ur['numPointsInserted']):
                for kk in range(k - 1):
                    self.assertTrue(distances[nn][kk] <= distances[nn][kk+1])
                    
                for kk in range(k):
                    idx = neighbors[nn][kk]

                    self.assertAlmostEqual(distances[nn][kk], np.sum((x[nn] - x[idx]) ** 2) ** 0.5, places=3)

                        
if __name__ == '__main__':
    unittest.main()
