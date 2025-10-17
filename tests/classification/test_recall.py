import unittest
import numpy as np
from lono_libs.core._base_metric import IMetric
from lono_libs.classification import Recall

class TestRecall(unittest.TestCase):
    def setUp(self):
        self.metric = Recall()

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
        # R for 0: 1/2 = 0.5 (1 correct out of 2 true 0)
        # R for 1: 2/3 = 0.666 (2 correct out of 3 true 1)
        # Weighted avg: (0.5 * 2 + 0.666... * 3) / 5 = (1 + 2) / 5 = 0.6
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.6, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_multiclass_perfect_agreement(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_multiclass_partial_agreement(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])
        # R0: 1/2 = 0.5, R1: 1/2 = 0.5, R2: 1/2 = 0.5
        # Weighted R: (0.5*2 + 0.5*2 + 0.5*2) / 6 = (1 + 1 + 1) / 6 = 3/6 = 0.5
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.5, places=7)

    def test_calculate_from_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)