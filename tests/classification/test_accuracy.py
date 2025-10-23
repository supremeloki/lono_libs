import unittest
import numpy as np
from lono_libs.classification import Accuracy

class TestAccuracy(unittest.TestCase):
    def test_perfect_accuracy(self):
        y_true = np.array(['A', 'B', 'C'])
        y_pred = np.array(['A', 'B', 'C'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 1.0)
