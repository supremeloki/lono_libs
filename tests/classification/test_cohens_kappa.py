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
        y_true = np.array([1, 1, 2, 2])
        y_pred = np.array([2, 2, 1, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_partial_agreement(self):
        y_true = np.array([1, 1, 2, 2, 3, 3])
        y_pred = np.array([1, 2, 2, 2, 3, 1])
        result = self.metric.calculate(y_true, y_pred)
        self.assertTrue(0 < result < 1)

    def test_empty_inputs(self):
        y_true = np.array([])
        y_pred = np.array([])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 0.0)

    def test_single_class(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        self.assertAlmostEqual(self.metric.calculate(y_true, y_pred), 1.0)

    def test_binary_classification(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = self.metric.calculate(y_true, y_pred)
        self.assertTrue(isinstance(result, float))

    def test_multiclass_classification(self):
        y_true = np.array(['A', 'B', 'C', 'A', 'B', 'C'])
        y_pred = np.array(['A', 'A', 'C', 'B', 'B', 'C'])
        result = self.metric.calculate(y_true, y_pred)
        self.assertTrue(isinstance(result, float))