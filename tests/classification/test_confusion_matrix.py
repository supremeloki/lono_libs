import unittest
import numpy as np
from lono_libs.classification import ConfusionMatrix

class TestConfusionMatrix(unittest.TestCase):
    def setUp(self):
        self.metric = ConfusionMatrix()

    def test_perfect_agreement_binary(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        expected_cm = np.array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_partial_agreement_binary(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        expected_cm = np.array([[1, 1], [1, 2]]) # TN=1, FP=1, FN=1, TP=2
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_perfect_agreement_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        expected_cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_partial_agreement_multiclass(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 1])
        # T0: 0,1,1 -> P0:1, P1:1, P2:0 (pred 0 is correct, pred 1 is wrong)
        # T1: 0,1,1 -> P0:1, P1:1, P2:0 (pred 1 is correct, pred 0 is wrong)
        # T2: 0,1,1 -> P0:0, P1:1, P2:1 (pred 2 is correct, pred 1 is wrong)
        # True  | Pred 0 | Pred 1 | Pred 2
        # ------|--------|--------|--------
        #   0   |   1    |   1    |   0
        #   1   |   1    |   1    |   0
        #   2   |   0    |   1    |   1
        expected_cm = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1]])
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        expected_cm = np.array([])
        np.testing.assert_array_equal(self.metric.calculate(y_true, y_pred), expected_cm)

    def test_calculate_from_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        # y_pred will be [0, 1, 0, 1] -> Perfect agreement
        expected_cm = np.array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(self.metric.calculate_from_proba(y_true, y_pred_proba), expected_cm)