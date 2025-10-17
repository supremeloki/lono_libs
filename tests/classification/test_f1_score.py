import unittest
import numpy as np
# Assuming IMetric and F1Score are available from lono_libs
from lono_libs.core import IMetric
from lono_libs.classification import F1Score

class TestF1Score(unittest.TestCase):
    def setUp(self):
        self.metric = F1Score()

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
        # Expected F1-score (weighted) for this scenario: 0.7333333333333334
        # True Negatives: 1 (y_true=0, y_pred=0)
        # False Positives: 1 (y_true=0, y_pred=1)
        # False Negatives: 1 (y_true=1, y_pred=0)
        # True Positives: 2 (y_true=1, y_pred=1)
        # Precision (0): 1/(1+1)=0.5, Recall (0): 1/(1+1)=0.5, F1(0)=0.5
        # Precision (1): 2/(1+2)=0.666, Recall (1): 2/(1+1+2)=0.666, F1(1)=0.666
        # Weighted: (0.5*2 + 0.666*3) / 5 = (1 + 2) / 5 = 0.6
        # Using sklearn's f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.7333333333, places=7)

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
        # Expected F1-score (weighted) based on sklearn: 0.5555555555555556
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.5555555555, places=7)

    def test_calculate_from_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        # y_pred will be [0, 1, 0, 1] -> Perfect agreement
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)

    def test_calculate_from_proba_multiclass(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.5, 0.3, 0.2]
        ])
        # y_pred will be [0, 1, 2, 0] -> Perfect agreement
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)