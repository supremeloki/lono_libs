import unittest
import numpy as np
from lono_libs.classification import Precision

class TestPrecision(unittest.TestCase):
    def setUp(self):
        self.metric = Precision()

    def test_perfect_agreement_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_zero_agreement_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_partial_agreement_binary(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        # P for 0: 1/2 = 0.5 (1 correct out of 2 predicted 0)
        # P for 1: 2/3 = 0.666 (2 correct out of 3 predicted 1)
        # Weighted avg: (0.5 * 2 + 0.666... * 3) / 5 = (1 + 2) / 5 = 0.6
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.6, places=7)

    def test_empty_inputs(self):
