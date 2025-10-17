import unittest
import numpy as np
# Assuming IMetric and R2Score are available from lono_libs
from lono_libs.core._base_metric import IMetric
from lono_libs.regression import R2Score

class TestR2Score(unittest.TestCase):
    def setUp(self):
        self.metric = R2Score()

    def test_perfect_prediction(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_no_improvement_over_mean(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([np.mean(y_true), np.mean(y_true), np.mean(y_true), np.mean(y_true)])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_worse_than_mean(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([10, 10, 10]) # Much worse than predicting the mean (2)
        # R2 score can be negative when model is arbitrarily worse than predicting the mean.
        # (1 - (MSE / VAR)) = (1 - (59.33 / 0.666)) = 1 - 89 = -88
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), -88.0, places=7)


    def test_partial_prediction(self):
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        # Based on sklearn.metrics.r2_score, expected value is 0.9486...
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.9486081370449317, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_calculate_from_proba_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate_from_proba(np.array([0.5, 1.5]), np.array([[0.1, 0.9], [0.8, 0.2]]))