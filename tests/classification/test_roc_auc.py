import unittest
import numpy as np
from lono_libs.core._base_metric import IMetric
from lono_libs.classification import ROCAUC

class TestROCAUC(unittest.TestCase):
    def setUp(self):
        self.metric = ROCAUC()

    def test_perfect_prediction_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)

    def test_zero_prediction_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([[0.1, 0.9], [0.2, 0.8], [0.9, 0.1], [0.8, 0.2]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.0)

    def test_partial_prediction_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
        # Expected ROC AUC for this scenario is 0.5
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.5)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred_proba = np.array([[]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 0.0)

    def test_multiclass_perfect_prediction(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)
