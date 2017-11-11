from pynene import KNNTable
import numpy as np
import unittest
import time

def random_vectors(n=100, d=10, dtype=np.float32):
    return np.array(np.random.rand(n, d), dtype=dtype)

class Test_KNNTable(unittest.TestCase):
    def test_table(self):
        n=100
        d=10
        k=5
        array = random_vectors(n, d)
        table = KNNTable(array, k,
                         np.zeros((n, 5), dtype=np.int),
                         np.zeros((n, 5), dtype=np.float32))
        
        self.assertTrue(table != None)
        #table.run(10)
