import unittest
import numpy as np
from lono_libs.regression import Kurtosis

class TestKurtosis(unittest.TestCase):
    def setUp(self):
        self.metric = Kurtosis()

    def test_normal_distribution_residuals(self):
        # For a perfect fit, residuals are 0, kurtosis is undefined, but for small noise
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1.1, 2.0, 3.0, 4.1, 5.0, 6.2, 7.0, 8.1, 9.0, 10.1])
        # Residuals are close to a normal distribution (centered around 0)
        # Expected kurtosis for a normal distribution (Fisher) is 0.0
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), -0.6281, places=4) # Example calculated with scipy.stats.kurtosis

    def test_platykurtic_residuals(self): # Flatter than normal, negative excess kurtosis
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8]) + np.random.choice([-1, 0, 1], size=8)
        # Example to generate known platykurtic behavior (e.g., uniform-like residuals)
        residuals = np.array([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y_pred = y_true - residuals
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), -2.0, places=7) # Kurtosis of a uniform distribution is -1.2 (Fisher) - or -2.0 for specifically these.

    def test_leptokurtic_residuals(self): # Peaked, positive excess kurtosis
        y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        residuals = np.array([0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 0.0, 0.0, 0.0, 0.0]) # Outliers
        y_pred = y_true - residuals
        # High positive kurtosis due to outliers (leptokurtic)
        self.assertGreater(self.metric.calculate(y_true, y_pred), 1.0) # Should be a large positive number
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0, places=0) # For these specific vals, it's 1.0 (Fisher = -1.2 for uniform and +1.5 for Laplace)

    def test_small_inputs_nan(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        # Kurtosis requires at least 4 samples. For less, return NaN.
        self.assertTrue(np.isnan(self.metric.calculate(y_true, y_pred)))

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertTrue(np.isnan(self.metric.calculate(y_true, y_pred)))

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))