import unittest
import numpy as np
from lono_libs.classification import MatthewsCorrelationCoefficient

class TestMatthewsCorrelationCoefficient(unittest.TestCase):
    def setUp(self):
        self.metric = MatthewsCorrelationCoefficient()

    def test_perfect_agreement_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_perfect_agreement_imbalanced(self):
        y_true = np.array([0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_complete_disagreement_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), -1.0)

    def test_random_prediction(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        # Expected MCC for this case is 0.0
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_partial_agreement_binary(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1])
        # TN=2, FP=1, FN=1, TP=1
        # MCC = (2*1 - 1*1) / sqrt((2+1)(2+1)(1+1)(1+1)) = (2-1) / sqrt(3*3*2*2) = 1 / sqrt(36) = 1/6 = 0.1666...
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.1666666666, places=7)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_multiclass_agreement(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_calculate_from_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        # y_pred will be [0, 1, 0, 1] -> Perfect agreement
        self.assertAlmostEqual(self.metric.calculate_from_proba(y_true, y_pred_proba), 1.0)