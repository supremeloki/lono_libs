import unittest
import numpy as np
import pandas as pd
import logging
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional
logging.basicConfig(level=logging.WARNING)
try:
    from . import (
        generate_binary_classification_data,
        generate_multiclass_classification_data,
        generate_regression_data,
        assert_almost_equal,
        assert_score_range,
        PERFECT_PREDICTIONS,
        RANDOM_PREDICTIONS,
        EDGE_CASES,
        DEFAULT_TOLERANCE,
        TEST_SEED
    )
except ImportError:
    from tests import (
        generate_binary_classification_data,
        generate_multiclass_classification_data,
        generate_regression_data,
        assert_almost_equal,
        assert_score_range,
        PERFECT_PREDICTIONS,
        RANDOM_PREDICTIONS,
        EDGE_CASES,
        DEFAULT_TOLERANCE,
        TEST_SEED
    )
from lono_libs.core.evaluator import Evaluator
from lono_libs.core import IMetric
from lono_libs.classification import (
    Accuracy, BalancedAccuracy, CohensKappa, F1Score,
    Precision, Recall, ROCAUC, LogLoss,
    MatthewsCorrelationCoefficient
)
from lono_libs.regression import (
    Correlation, MAE, MSE, R2Score, Kurtosis, Skewness
)
class CreativeMetric(IMetric):
    """A creative metric for testing - calculates the 'harmony' of predictions."""
    name: str = "Harmony"
    is_higher_better: bool = True
    weight: float = 0.5
    target_score: Optional[float] = 0.85
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        diff = np.abs(y_true - y_pred)
        harmony = 1.0 / (1.0 + np.mean(diff))
        return float(harmony)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        pred_from_proba = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
        return self.calculate(y_true, pred_from_proba)
class ChaoticMetric(IMetric):
    """A chaotic metric that simulates unpredictable evaluation patterns."""
    name: str = "Chaos"
    is_higher_better: bool = False
    weight: float = 0.1
    target_score: Optional[float] = 0.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        x, y = 0.1, 0.1
        for i in range(len(y_true)):
            x = np.sin(y * np.pi) + np.cos(x * np.pi)
            y = np.sin(x + y) * np.cos(y)
        return float(np.abs(x + y) / 2.0)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return self.calculate(y_true, np.argmax(y_pred_proba, axis=1))
class TestEvaluator(unittest.TestCase):
    """Comprehensive test suite for the Evaluator class."""
    def setUp(self):
        """Set up test fixtures before each test method."""
        np.random.seed(TEST_SEED)
        self.classification_metrics = [
            Accuracy(), BalancedAccuracy(), CohensKappa(),
            F1Score(), Precision(), Recall(), ROCAUC(), LogLoss(),
            MatthewsCorrelationCoefficient()
        ]
        self.regression_metrics = [
            Correlation(), MAE(), MSE(), R2Score(),
            Kurtosis(), Skewness()
        ]
        self.creative_metrics = [CreativeMetric(), ChaoticMetric()]
    def test_evaluator_initialization(self):
        """Test Evaluator initialization with different metric combinations."""
        evaluator = Evaluator(self.classification_metrics)
        self.assertEqual(len(evaluator._metrics), len(self.classification_metrics))
        mixed_metrics = self.classification_metrics[:3] + self.regression_metrics[:2]
        evaluator = Evaluator(mixed_metrics)
        self.assertEqual(len(evaluator._metrics), 5)
        evaluator = Evaluator([])
        self.assertEqual(len(evaluator._metrics), 0)
    def test_perfect_predictions_scenario(self):
        """Test evaluation with perfect predictions across all scenarios."""
        evaluator = Evaluator(self.classification_metrics)
        evaluation_data = [{
            'model_name': 'PerfectClassifier',
            'model_type': 'classification',
            'y_true': PERFECT_PREDICTIONS['binary']['y_true'],
            'y_pred_train': PERFECT_PREDICTIONS['binary']['y_pred'],
            'y_pred_test': PERFECT_PREDICTIONS['binary']['y_pred'],
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        accuracy_results = results_df[results_df['metric_name'] == 'Accuracy']
        self.assertTrue(len(accuracy_results) > 0)
        self.assertAlmostEqual(accuracy_results.iloc[0]['Testing Score'], 1.0, places=6)
    def test_quantum_inspired_predictions(self):
        """Creative test: Simulate quantum-inspired probabilistic predictions."""
        evaluator = Evaluator([ROCAUC(), LogLoss()])
        n_samples = 100
        y_true = np.random.choice([0, 1], n_samples)
        base_proba = np.random.beta(2, 2, n_samples)
        entangled_proba = np.where(y_true == 1, base_proba, 1 - base_proba)
        y_pred_proba = np.column_stack([1 - entangled_proba, entangled_proba])
        evaluation_data = [{
            'model_name': 'QuantumModel',
            'model_type': 'classification',
            'y_true': y_true,
            'y_pred_train': np.argmax(y_pred_proba, axis=1),
            'y_pred_test': np.argmax(y_pred_proba, axis=1),
            'y_pred_proba_train': y_pred_proba,
            'y_pred_proba_test': y_pred_proba
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        roc_results = results_df[results_df['metric_name'] == 'ROCAUC']
        logloss_results = results_df[results_df['metric_name'] == 'LogLoss']
        self.assertTrue(len(roc_results) > 0)
        self.assertTrue(len(logloss_results) > 0)
        assert_score_range(roc_results.iloc[0]['Testing Score'], 0.4, 1.0)
    def test_fractal_evaluation_patterns(self):
        """Creative test: Simulate fractal-like recursive evaluation patterns."""
        evaluator = Evaluator([Accuracy()])
        def generate_fractal_data(depth: int, base_size: int = 10):
            if depth == 0:
                return generate_binary_classification_data(base_size)
            else:
                data = generate_fractal_data(depth - 1, base_size)
                noise = np.random.normal(0, 0.1, len(data['y_true']))
                data['y_pred'] = np.where(data['y_pred'] + noise > 0.5, 1, 0)
                return data
        fractal_data = generate_fractal_data(3)
        evaluation_data = [{
            'model_name': 'FractalModel',
            'model_type': 'classification',
            'y_true': fractal_data['y_true'],
            'y_pred_train': fractal_data['y_pred'],
            'y_pred_test': fractal_data['y_pred'],
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        self.assertTrue(len(results_df) > 0)
        accuracy_score = results_df.iloc[0]['Testing Score']
        assert_score_range(accuracy_score, 0.0, 1.0)
    def test_multiverse_evaluation_scenarios(self):
        """Creative test: Simulate multiple universe evaluation scenarios."""
        evaluator = Evaluator(self.classification_metrics[:3])  # Use subset for efficiency
        universes_data = []
        for universe_id in range(5):
            np.random.seed(TEST_SEED + universe_id)
            data = generate_multiclass_classification_data(50, 3)
            universes_data.append({
                'model_name': f'UniverseModel_{universe_id}',
                'model_type': 'classification',
                'y_true': data['y_true'],
                'y_pred_train': data['y_pred'],
                'y_pred_test': data['y_pred'],
                'y_pred_proba_train': data['y_pred_proba'],
                'y_pred_proba_test': data['y_pred_proba']
            })
        results_df, best_models = evaluator.evaluate_models(universes_data)
        self.assertEqual(len(results_df), len(universes_data) * 3)  # 3 metrics per universe
        self.assertEqual(len(best_models), 3)  # One best model per metric
    def test_chaotic_system_simulation(self):
        """Creative test: Evaluate models with chaotic system behaviors."""
        chaotic_metric = ChaoticMetric()
        evaluator = Evaluator([chaotic_metric])
        n_samples = 200
        t = np.linspace(0, 4*np.pi, n_samples)
        x = np.sin(t) + 0.1 * np.sin(10*t)
        y_true = x + np.random.normal(0, 0.1, n_samples)
        y_pred = x + np.random.normal(0, 0.2, n_samples)
        evaluation_data = [{
            'model_name': 'ChaosModel',
            'model_type': 'regression',
            'y_true': y_true,
            'y_pred_train': y_pred,
            'y_pred_test': y_pred,
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        chaos_score = results_df.iloc[0]['Testing Score']
        self.assertIsInstance(chaos_score, float)
        self.assertTrue(0.0 <= chaos_score <= 1.0)
    def test_time_travel_debugging_simulation(self):
        """Creative test: Simulate time-travel debugging with historical evaluations."""
        evaluator = Evaluator([Accuracy(), F1Score()])
        time_snapshots = []
        base_data = generate_binary_classification_data(100)
        for time_step in range(10):
            noise_level = time_step * 0.01
            noisy_pred = base_data['y_pred'] + np.random.normal(0, noise_level, len(base_data['y_pred']))
            noisy_pred = np.clip(noisy_pred, 0, 1).astype(int)
            time_snapshots.append({
                'model_name': f'Model_T{time_step}',
                'model_type': 'classification',
                'y_true': base_data['y_true'],
                'y_pred_train': noisy_pred,
                'y_pred_test': noisy_pred,
                'y_pred_proba_train': None,
                'y_pred_proba_test': None
            })
        results_df, best_models = evaluator.evaluate_models(time_snapshots)
        self.assertEqual(len(results_df), len(time_snapshots) * 2)  # 2 metrics per snapshot
        accuracy_scores = results_df[results_df['metric_name'] == 'Accuracy']['Testing Score'].values
        for score in accuracy_scores:
            assert_score_range(score, 0.0, 1.0)
    def test_weighted_average_calculation(self):
        """Test weighted average score calculation with various metric weights."""
        weighted_metrics = [
            Accuracy(weight=0.5, target_score=0.9),
            F1Score(weight=0.3, target_score=0.85),
            Precision(weight=0.2, target_score=0.88)
        ]
        evaluator = Evaluator(weighted_metrics)
        evaluation_data = [{
            'model_name': 'WeightedModel',
            'model_type': 'classification',
            'y_true': np.array([0, 1, 0, 1, 0, 1]),
            'y_pred_train': np.array([0, 1, 0, 1, 0, 1]),
            'y_pred_test': np.array([0, 1, 0, 1, 0, 1]),
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        self.assertTrue('weighted_score' in results_df.columns)
        weighted_score = results_df.iloc[0]['weighted_score']
        self.assertIsInstance(weighted_score, float)
        assert_score_range(weighted_score, 0.0, 1.0)
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling scenarios."""
        evaluator = Evaluator([Accuracy()])
        results_df, best_models = evaluator.evaluate_models([])
        self.assertTrue(results_df.empty)
        self.assertEqual(len(best_models), 0)
        evaluation_data = [{
            'model_name': 'SingleSampleModel',
            'model_type': 'classification',
            'y_true': EDGE_CASES['single_sample']['y_true'],
            'y_pred_train': EDGE_CASES['single_sample']['y_pred'],
            'y_pred_test': EDGE_CASES['single_sample']['y_pred'],
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        self.assertTrue(len(results_df) > 0)
    def test_metric_error_handling(self):
        """Test error handling when metrics fail to calculate."""
        class FailingMetric(IMetric):
            name: str = "Failing"
            is_higher_better: bool = True
            weight: float = 0.0
            target_score: Optional[float] = None
            def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
                raise ValueError("Intentional failure for testing")
            def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
                raise ValueError("Intentional failure for testing")
        evaluator = Evaluator([FailingMetric(), Accuracy()])
        evaluation_data = [{
            'model_name': 'ErrorModel',
            'model_type': 'classification',
            'y_true': np.array([0, 1, 0, 1]),
            'y_pred_train': np.array([0, 1, 0, 1]),
            'y_pred_test': np.array([0, 1, 0, 1]),
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        with patch('lono_libs.core.evaluator._logger') as mock_logger:
            results_df, best_models = evaluator.evaluate_models(evaluation_data)
            mock_logger.error.assert_called()
            self.assertTrue(len(results_df) > 0)
    def test_best_model_selection(self):
        """Test best model selection logic for different metric types."""
        evaluator = Evaluator([Accuracy(), MAE()])  # Higher better and lower better
        evaluation_data = [
            {
                'model_name': 'ModelA',
                'model_type': 'classification',
                'y_true': np.array([0, 1, 0, 1]),
                'y_pred_train': np.array([0, 1, 0, 0]),  # 75% accuracy
                'y_pred_test': np.array([0, 1, 0, 0]),
                'y_pred_proba_train': None,
                'y_pred_proba_test': None
            },
            {
                'model_name': 'ModelB',
                'model_type': 'classification',
                'y_true': np.array([0, 1, 0, 1]),
                'y_pred_train': np.array([0, 1, 0, 1]),  # 100% accuracy
                'y_pred_test': np.array([0, 1, 0, 1]),
                'y_pred_proba_train': None,
                'y_pred_proba_test': None
            }
        ]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        self.assertIn('Accuracy', best_models)
        best_accuracy_model = best_models['Accuracy']
        self.assertEqual(best_accuracy_model['model_name'].iloc[0], 'ModelB')
    def test_regression_evaluation(self):
        """Test evaluation with regression metrics."""
        evaluator = Evaluator(self.regression_metrics)
        reg_data = generate_regression_data(50)
        evaluation_data = [{
            'model_name': 'RegressionModel',
            'model_type': 'regression',
            'y_true': reg_data['y_true'],
            'y_pred_train': reg_data['y_pred'],
            'y_pred_test': reg_data['y_pred'],
            'y_pred_proba_train': None,
            'y_pred_proba_test': None
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        self.assertEqual(len(results_df), len(self.regression_metrics))
        for metric in self.regression_metrics:
            metric_results = results_df[results_df['metric_name'] == metric.name]
            self.assertTrue(len(metric_results) > 0)
            score = metric_results.iloc[0]['Testing Score']
            self.assertIsInstance(score, float)
    def test_probability_based_metrics(self):
        """Test metrics that use probability predictions."""
        evaluator = Evaluator([ROCAUC(), LogLoss()])
        n_samples = 50
        y_true = np.random.choice([0, 1], n_samples)
        y_pred_proba = np.random.rand(n_samples, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
        evaluation_data = [{
            'model_name': 'ProbaModel',
            'model_type': 'classification',
            'y_true': y_true,
            'y_pred_train': np.argmax(y_pred_proba, axis=1),
            'y_pred_test': np.argmax(y_pred_proba, axis=1),
            'y_pred_proba_train': y_pred_proba,
            'y_pred_proba_test': y_pred_proba
        }]
        results_df, best_models = evaluator.evaluate_models(evaluation_data)
        roc_results = results_df[results_df['metric_name'] == 'ROCAUC']
        logloss_results = results_df[results_df['metric_name'] == 'LogLoss']
        self.assertTrue(len(roc_results) > 0)
        self.assertTrue(len(logloss_results) > 0)
if __name__ == '__main__':
    print("🚀 Initiating Lono Libs Evaluator Test Suite with Creative Scenarios")
    print("=" * 70)
    print("🌌 Quantum-inspired evaluations: Preparing...")
    print("🌀 Fractal pattern simulations: Loading...")
    print("🌍 Multiverse scenarios: Generating...")
    print("⏰ Time-travel debugging: Activating...")
    print("⚡ Chaotic system analysis: Initializing...")
    print("=" * 70)
    unittest.main(verbosity=2)
