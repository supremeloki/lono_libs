import unittest
import numpy as np
from lono_libs.classification import LogLoss

class TestLogLoss(unittest.TestCase):
    def setUp(self):
        self.metric = LogLoss()

    def test_perfect_prediction_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.10536, places=5) # Example value

    def test_imperfect_prediction_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.50970, places=5) # Example value

    def test_perfect_prediction_multiclass(self):
        y_true = np.array([0, 1, 2])
        y_pred_proba = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.10536, places=5) # Example value

    def test_imperfect_prediction_multiclass(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred_proba = np.array([
            [0.6, 0.2, 0.2],
            [0.1, 0.7, 0.2],
            [0.3, 0.3, 0.4],
            [0.2, 0.5, 0.3]
        ])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.07669, places=5) # Example value

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred_proba = np.array([[]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.0)

    def test_calculate_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self.metric.calculate(np.array([0, 1]), np.array([0, 1]))