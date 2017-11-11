from pynene import KNNTable
import numpy as np
import unittest


def random_vectors(n=100, d=10, dtype=np.float32):
    return np.array(np.random.rand(n, d), dtype=dtype)

class Test_KNNTable(unittest.TestCase):
    def test_table(self):
        n=100
        d=10
        k=5
        array = random_vectors(n, d)
        neighbors = np.zeros((n, 5), dtype=np.int)
        distances = np.zeros((n, 5), dtype=np.float32)
        table = KNNTable(array, k, neighbors, distances)
        
        self.assertTrue(table != None)
        #table.run(10)
