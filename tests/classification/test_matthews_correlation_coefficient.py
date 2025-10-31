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
