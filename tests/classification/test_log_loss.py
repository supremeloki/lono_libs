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
