import unittest
import numpy as np
# Assuming IMetric and BalancedAccuracy are available from lono_libs
from lono_libs.core import IMetric
from lono_libs.classification import BalancedAccuracy

class TestBalancedAccuracy(unittest.TestCase):
    def setUp(self):
        self.metric = BalancedAccuracy()

    def test_perfect_agreement_balanced(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_perfect_agreement_imbalanced(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_complete_disagreement_balanced(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_complete_disagreement_imbalanced(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([1, 1, 1, 0])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_partial_agreement_imbalanced(self):
        # TN=2, FP=1, FN=0, TP=1 (y_true=[0,0,0,1], y_pred=[0,0,1,1]) -> BA: (0.66 + 1)/2 = 0.833
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        # Recall(0): 2/3, Recall(1): 1/1. (2/3 + 1)/2 = (0.66666 + 1) / 2 = 0.83333
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.8333333333, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_multiclass_agreement(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_multiclass_partial_agreement(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 1, 2, 1, 2, 2])
        # Recall(0): 2/3, Recall(1): 1/3, Recall(2): 2/3
        # (2/3 + 1/3 + 2/3) / 3 = (5/3) / 3 = 5/9 = 0.5555...
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
            [0.5, 0.3, 0.2] # This will be predicted as 0
        ])
        # y_pred will be [0, 1, 2, 0] -> Perfect agreement
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)
