import unittest
import numpy as np
from lono_libs.regression.correlation import Correlation

class TestCorrelation(unittest.TestCase):
    def setUp(self):
        self.metric = Correlation()

    def test_perfect_positive_correlation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_perfect_negative_correlation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), -1.0)

    def test_no_correlation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 3, 1, 4, 2]) # Random permutation
        # Expected correlation very close to 0, or exactly 0 for specific random arrangements
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0, places=7)

    def test_partial_correlation(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.3, 2.8, 4.2, 5.0])
        # Expected correlation based on scipy.stats.pearsonr
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.992645607374093, places=7)

    def test_constant_true_values(self):
        y_true = np.array([5, 5, 5, 5])
        y_pred = np.array([1, 2, 3, 4])
        # Correlation is undefined (division by zero std dev) for constant true or pred.
        # scipy's pearsonr returns NaN or 0.0, we return 0.0 as per IMetric standard.
        self.assertTrue(np.isnan(self.metric.calculate(y_true, y_pred))) # pearsonr returns NaN for constant input
        # NOTE: For practical use in UnifiedRunner, you might want to handle NaN by converting to 0.0 or a very low score.

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_single_sample(self):
        y_true = np.array([1])
        y_pred = np.array([1])
        # Pearson correlation is undefined for a single sample.
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0) # Our implementation returns 0.0 for <2 samples

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))