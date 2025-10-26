import unittest
import numpy as np
from lono_libs.classification import ConfusionMatrix

class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.metric = ConfusionMatrix()

    def test_perfect_agreement_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        expected_cm = np.array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_partial_agreement_binary(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        expected_cm = np.array([[1, 1], [1, 2]]) # TN=1, FP=1, FN=1, TP=2
