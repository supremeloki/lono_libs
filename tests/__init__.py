"""
Lono Libs Test Suite
This module contains comprehensive tests for the Lono Libs evaluation framework.
The tests cover all core functionality including metric calculations, evaluator
operations, and integration scenarios with creative edge cases.
"""
__version__ = "1.0.0"
__author__ = "Lono Libs Development Team"
__description__ = "Test suite for Lono Libs evaluation framework"
TEST_SEED = 42
DEFAULT_TOLERANCE = 1e-6
MAX_TEST_SAMPLES = 1000
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
np.random.seed(TEST_SEED)
def generate_binary_classification_data(n_samples: int = 100) -> Dict[str, np.ndarray]:
    """Generate synthetic binary classification data."""
    np.random.seed(TEST_SEED)
    y_true = np.random.choice([0, 1], n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    y_pred = np.where(y_true + noise > 0.5, 1, 0)
    y_pred_proba = np.random.rand(n_samples)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
def generate_multiclass_classification_data(n_samples: int = 100, n_classes: int = 3) -> Dict[str, np.ndarray]:
    """Generate synthetic multiclass classification data."""
    np.random.seed(TEST_SEED)
    y_true = np.random.choice(range(n_classes), n_samples)
    y_pred = np.random.choice(range(n_classes), n_samples)
    y_pred_proba = np.random.rand(n_samples, n_classes)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
def generate_regression_data(n_samples: int = 100) -> Dict[str, np.ndarray]:
    """Generate synthetic regression data."""
    np.random.seed(TEST_SEED)
    X = np.random.randn(n_samples, 2)
    y_true = X[:, 0] * 2 + X[:, 1] * -1 + np.random.normal(0, 0.1, n_samples)
    y_pred = y_true + np.random.normal(0, 0.2, n_samples)
    return {
        'y_true': y_true,
        'y_pred': y_pred
    }
def assert_almost_equal(actual: float, expected: float, tolerance: float = DEFAULT_TOLERANCE):
    """Assert that two values are almost equal within tolerance."""
    assert abs(actual - expected) < tolerance, f"Expected {expected}, got {actual}"
def assert_score_range(score: float, min_val: float = 0.0, max_val: float = 1.0):
    """Assert that a score is within expected range."""
    assert min_val <= score <= max_val, f"Score {score} not in range [{min_val}, {max_val}]"
PERFECT_PREDICTIONS = {
    'binary': {'y_true': np.array([0, 1, 0, 1]), 'y_pred': np.array([0, 1, 0, 1])},
    'multiclass': {'y_true': np.array([0, 1, 2, 0, 1, 2]), 'y_pred': np.array([0, 1, 2, 0, 1, 2])},
    'regression': {'y_true': np.array([1.0, 2.0, 3.0]), 'y_pred': np.array([1.0, 2.0, 3.0])}
}
RANDOM_PREDICTIONS = {
    'binary': generate_binary_classification_data(50),
    'multiclass': generate_multiclass_classification_data(50, 4),
    'regression': generate_regression_data(50)
}
EDGE_CASES = {
    'empty_arrays': {'y_true': np.array([]), 'y_pred': np.array([])},
    'single_sample': {'y_true': np.array([1]), 'y_pred': np.array([1])},
    'constant_values': {'y_true': np.array([1, 1, 1]), 'y_pred': np.array([1, 1, 1])},
    'extreme_values': {'y_true': np.array([0, 1000]), 'y_pred': np.array([0, 1000])}
}
__all__ = [
    "generate_binary_classification_data",
    "generate_multiclass_classification_data",
    "generate_regression_data",
    "assert_almost_equal",
    "assert_score_range",
    "PERFECT_PREDICTIONS",
    "RANDOM_PREDICTIONS",
    "EDGE_CASES",
    "TEST_SEED",
    "DEFAULT_TOLERANCE",
    "MAX_TEST_SAMPLES"
]
