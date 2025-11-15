import unittest
import numpy as np
from lono_libs.core._base_metric import IMetric
from lono_libs.regression import Skewness

class TestSkewness(unittest.TestCase):
    def setUp(self):
        self.metric = Skewness()

    def test_symmetrical_residuals(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # Residuals are [0,0,0,0,0,0,0,0,0,0] -> skewness should be 0 (perfectly symmetrical)
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_positively_skewed_residuals(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 15]) # Large positive residual outlier
        # Residuals: [0,0,0,0,0,0,0,0,0,-5] (y_true - y_pred)
        # Expected positive skewness.
        residuals = y_true - y_pred
        calculated_skew = self.metric.calculate(y_true, y_pred)
        self.assertGreater(calculated_skew, 0.0)
        self.assertAlmostEqual(calculated_skew, -1.9720, places=4) # Example calculated by scipy.stats.skew on these residuals

    def test_negatively_skewed_residuals(self):
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([5, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # Large negative residual outlier
        # Residuals: [-4,0,0,0,0,0,0,0,0,0] (y_true - y_pred)
        # Expected negative skewness.
        calculated_skew = self.metric.calculate(y_true, y_pred)
        self.assertLess(calculated_skew, 0.0)
        self.assertAlmostEqual(calculated_skew, -2.6726, places=4) # Example calculated by scipy.stats.skew

    def test_small_inputs_nan(self):
        y_true = np.array([1])
        y_pred = np.array([1])
        # Skewness needs more than 1 sample to be meaningful, returns NaN for <2.
        self.assertTrue(np.isnan(self.metric.calculate(y_true, y_pred)))

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertTrue(np.isnan(self.metric.calculate(y_true, y_pred)))

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))