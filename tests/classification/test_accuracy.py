import unittest
import numpy as np
from lono_libs.classification import Accuracy

class TestAccuracy(unittest.TestCase):
    def test_perfect_accuracy(self):
        y_true = np.array(['A', 'B', 'C'])
        y_pred = np.array(['A', 'B', 'C'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 1.0)

    def test_zero_accuracy(self):
        y_true = np.array(['A', 'B', 'C'])
        y_pred = np.array(['X', 'Y', 'Z'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 0.0)

    def test_partial_accuracy(self):
        y_true = np.array(['A', 'B', 'C', 'D'])
        y_pred = np.array(['A', 'X', 'C', 'Y'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 0.5)

    def test_empty_lists(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 0.0)

    def test_single_element_correct(self):
        y_true = np.array(['A'])
        y_pred = np.array(['A'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 1.0)

    def test_single_element_incorrect(self):
        y_true = np.array(['A'])
        y_pred = np.array(['B'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 0.0)

    def test_multi_class_accuracy(self):
        y_true = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
        y_pred = np.array(['cat', 'dog', 'mouse', 'bird', 'dog'])
        self.assertAlmostEqual(Accuracy().calculate(y_true, y_pred), 0.8)