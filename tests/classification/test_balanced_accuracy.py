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
