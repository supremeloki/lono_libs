import pandas as pd
import numpy as np
import os
import shutil
import unittest
from typing import Optional
import logging # For UnifiedRunner mock
# Mock classes for standalone test execution
class IMetric: # Redefining IMetric mock for standalone execution context
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError

class Accuracy(IMetric): # Redefining Accuracy mock for standalone execution context
    name: str = "Accuracy"
    is_higher_better: bool = True
    weight: float = 1.0
    target_score: Optional[float] = 1.0
    def calculate(self, y_true, y_pred) -> float:
        if not isinstance(y_true, list) or not isinstance(y_pred, list):
            raise TypeError("y_true and y_pred must be lists.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if not y_true:
            return 0.0
        correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct_predictions / len(y_true)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError
class UnifiedRunner: # Redefining UnifiedRunner mock for standalone execution context
    def __init__(self, output_base_dir="lono_results", enable_logging=False, enable_visualizations=False, target_score_overrides=None):
        self.output_base_dir = output_base_dir
        self.enable_logging = enable_logging
        self.enable_visualizations = enable_visualizations
        self.target_score_overrides = target_score_overrides if target_score_overrides is not None else {}
        self._metric_classes = {
            "Accuracy": Accuracy,
        }
        if self.enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.info(f"Mock UnifiedRunner initialized. Output dir: {self.output_base_dir}")
        os.makedirs(self.output_base_dir, exist_ok=True)

    def run_pipeline(self, X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg, preprocessing_config, metrics_to_evaluate):
        if self.enable_logging:
            logging.info(f"\nMock UnifiedRunner: Starting pipeline execution for metrics: {metrics_to_evaluate}...")

        full_results = []
        best_models_by_metric = {}

        y_pred_class = y_test_class.tolist()
        if len(y_pred_class) > 0 and np.random.rand() > 0.3:
            idx_to_change = np.random.randint(0, len(y_pred_class))
            original_val = y_pred_class[idx_to_change]
            possible_changes = [val for val in ['A', 'B', 'C', 'D'] if val != original_val]
            if possible_changes:
                y_pred_class[idx_to_change] = np.random.choice(possible_changes)
            else:
                y_pred_class[idx_to_change] = 'Z'

        for metric_name in metrics_to_evaluate:
            if metric_name in self._metric_classes:
                metric_instance = self._metric_classes[metric_name]()
                score = None
                metric_type = "Unknown"

                if metric_name == "Accuracy":
                    score = metric_instance.calculate(y_test_class.tolist(), y_pred_class)
                    metric_type = "Classification"
                
                if score is not None:
                    target_score = self.target_score_overrides.get(metric_name)
                    diff_from_target = abs(score - target_score) if target_score is not None else np.nan
                    
                    result_entry = {
                        "Model": "MockModel",
                        "Type": metric_type,
                        "metric_name": metric_name,
                        "Training Score": "N/A",
                        "Testing Score": score,
                        "Difference": "N/A",
                        "target_score": target_score,
                        "Score_Difference_From_Target": diff_from_target
                    }
                    full_results.append(result_entry)
                    
                    best_models_by_metric[metric_name] = pd.DataFrame([result_entry])

            else:
                if self.enable_logging:
                    logging.warning(f"Mock UnifiedRunner: Warning: Metric '{metric_name}' not supported in this mock runner.")
        
        if self.enable_logging:
            logging.info("Mock UnifiedRunner: Pipeline execution complete.")

        return pd.DataFrame(full_results), best_models_by_metric

    def generate_reports_and_visualizations(self, full_results_df, best_models_by_metric):
        if self.enable_visualizations:
            report_path = os.path.join(self.output_base_dir, "mock_unified_runner_report.md")
            if self.enable_logging:
                logging.info(f"Mock UnifiedRunner: Generating reports and visualizations to {report_path}...")
            
            report_content = "# UnifiedRunner Mock Report\n\n"
            report_content += "This is a mock report generated by the UnifiedRunner. In a real scenario, this would contain detailed analyses, performance metrics, and visualizations.\n\n"
            report_content += "## Full Results\n"
            report_content += full_results_df.to_markdown(index=False)
            report_content += "\n\n## Best Models by Metric\n"
            for metric, df in best_models_by_metric.items():
                report_content += f"\n### {metric}\n"
                report_content += df.to_markdown(index=False)
            report_content += "\n\n--- End Mock Report ---"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            if self.enable_logging:
                logging.info(f"Mock report saved to {report_path}")
        else:
            if self.enable_logging:
                logging.info("Mock UnifiedRunner: Visualizations disabled, no report generated.")

class TestUnifiedRunner(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(10, 2))
        self.X_test = pd.DataFrame(np.random.rand(5, 2))
        self.y_train_class = pd.Series(np.random.choice(['A', 'B'], 10))
        self.y_test_class = pd.Series(np.random.choice(['A', 'B'], 5))
        self.y_train_reg = pd.Series(np.random.rand(10))
        self.y_test_reg = pd.Series(np.random.rand(5))
        self.preprocessing_config = {"add_polynomial_features": False, "apply_scaler": False}
        self.output_dir = "test_lono_results_temp"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_run_pipeline_with_accuracy_metric(self):
        runner = UnifiedRunner(output_base_dir=self.output_dir, enable_logging=True, enable_visualizations=False)
        metrics_to_evaluate = ["Accuracy"]
        target_acc_score = 0.98
        runner.target_score_overrides = {"Accuracy": target_acc_score}
        
        full_results_df, best_models_by_metric = runner.run_pipeline(
            X_train=self.X_train, X_test=self.X_test,
            y_train_class=self.y_train_class, y_test_class=self.y_test_class,
            y_train_reg=self.y_train_reg, y_test_reg=self.y_test_reg,
            preprocessing_config=self.preprocessing_config,
            metrics_to_evaluate=metrics_to_evaluate
        )
        
        self.assertIsInstance(full_results_df, pd.DataFrame)
        self.assertFalse(full_results_df.empty)
        self.assertIn("Accuracy", full_results_df['metric_name'].values)
        self.assertIn("Accuracy", best_models_by_metric)
        self.assertIsInstance(full_results_df['Testing Score'].iloc[0], float)
        self.assertTrue(0.0 <= full_results_df['Testing Score'].iloc[0] <= 1.0)
        self.assertAlmostEqual(full_results_df['target_score'].iloc[0], target_acc_score)
        self.assertIsInstance(full_results_df['Score_Difference_From_Target'].iloc[0], float)

    def test_run_pipeline_with_unsupported_metric(self):
        runner = UnifiedRunner(output_base_dir=self.output_dir, enable_logging=True, enable_visualizations=False)
        metrics_to_evaluate = ["UnsupportedMetric"]
        
        full_results_df, best_models_by_metric = runner.run_pipeline(
            X_train=self.X_train, X_test=self.X_test,
            y_train_class=self.y_train_class, y_test_class=self.y_test_class,
            y_train_reg=self.y_train_reg, y_test_reg=self.y_test_reg,
            preprocessing_config=self.preprocessing_config,
            metrics_to_evaluate=metrics_to_evaluate
        )
        
        self.assertTrue(full_results_df.empty)
        self.assertFalse(bool(best_models_by_metric))

    def test_generate_reports_and_visualizations_enabled(self):
        runner = UnifiedRunner(output_base_dir=self.output_dir, enable_logging=True, enable_visualizations=True)
        full_results_df = pd.DataFrame([{"Model": "TestModel", "Type": "Classification", "metric_name": "Accuracy", "Training Score": 0.95, "Testing Score": 0.90, "Difference": 0.05, "target_score": 0.92, "Score_Difference_From_Target": -0.02}])
        best_models_by_metric = {"Accuracy": pd.DataFrame([{"Model": "TestModel", "Type": "Classification", "Testing Score": 0.90, "target_score": 0.92}])}
        
        runner.generate_reports_and_visualizations(full_results_df, best_models_by_metric)
        
        report_file_path = os.path.join(self.output_dir, "mock_unified_runner_report.md")
        self.assertTrue(os.path.exists(report_file_path))
        with open(report_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("# UnifiedRunner Mock Report", content)
            self.assertIn("Accuracy", content)
            self.assertIn("Testing Score", content)
            self.assertIn("0.90", content)
            self.assertIn("0.95", content)

    def test_generate_reports_and_visualizations_disabled(self):
        runner = UnifiedRunner(output_base_dir=self.output_dir, enable_logging=True, enable_visualizations=False)
        full_results_df = pd.DataFrame([{"Model": "Test", "metric_name": "Accuracy", "Testing Score": 0.9}])
        best_models_by_metric = {"Accuracy": pd.DataFrame([{"Model": "Test", "Testing Score": 0.9}])}
        
        runner.generate_reports_and_visualizations(full_results_df, best_models_by_metric)
        
        report_file_path = os.path.join(self.output_dir, "mock_unified_runner_report.md")
        self.assertFalse(os.path.exists(report_file_path))