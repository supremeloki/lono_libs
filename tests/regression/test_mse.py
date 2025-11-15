import unittest
import numpy as np
# Assuming IMetric and MSE are available from lono_libs
from lono_libs.core._base_metric import IMetric
from lono_libs.regression import MSE

class TestMSE(unittest.TestCase):
    def setUp(self):
        self.metric = MSE()

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
        # Errors: -0.5, -0.5, -0.5
        # Squared Errors: 0.25, 0.25, 0.25
        # MSE: (0.25 + 0.25 + 0.25) / 3 = 0.75 / 3 = 0.25
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.25)

    def test_mixed_errors(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        # Errors: 2, -2, 3
        # Squared Errors: 4, 4, 9
        # MSE: (4 + 4 + 9) / 3 = 17 / 3 = 5.666...
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 17 / 3, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))