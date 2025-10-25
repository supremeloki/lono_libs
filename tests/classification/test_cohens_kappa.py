import unittest
import numpy as np
from lono_libs.classification import CohensKappa

class TestCohensKappa(unittest.TestCase):
    def setUp(self):
        self.metric = CohensKappa()

    def test_perfect_agreement(self):
        y_true = np.array([1, 2, 3, 1, 2])
        y_pred = np.array([1, 2, 3, 1, 2])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_no_agreement_above_chance(self):
