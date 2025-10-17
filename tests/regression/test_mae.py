import unittest
import numpy as np
from lono_libs.regression import MAE

class TestMAE(unittest.TestCase):
    def setUp(self):
        self.metric = MAE()

    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_zero_error(self):
        y_true = np.array([5, 5, 5])
        y_pred = np.array([5, 5, 5])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_simple_errors(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 2.5, 3.5])
        # Absolute Errors: 0.5, 0.5, 0.5
        # MAE: (0.5 + 0.5 + 0.5) / 3 = 0.5
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.5)

    def test_mixed_errors(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        # Absolute Errors: 2, 2, 3
        # MAE: (2 + 2 + 3) / 3 = 7 / 3 ≈ 2.333...
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 7 / 3, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))