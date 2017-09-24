from pynene import Index
import numpy as np
import unittest

class Test_Panene(unittest.TestCase):
    def test(self):
        x = np.random.rand(100,10)
        index = Index(x)
        self.assertIs(x, index.array)
        index.add_points(100)

if __name__ == '__main__':
    unittest.main()
