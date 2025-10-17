import pandas as pd
import numpy as np
import os
import shutil
import logging
from typing import Optional
# Assume Accuracy is available from 'test/Accuracy_mock.py' or similar context for this mock
class IMetric: # Redefining IMetric mock here for standalone execution context
    name: str = ""
    is_higher_better: bool = False
    weight: float = 0.0
    target_score: Optional[float] = None
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError

class Accuracy(IMetric): # Redefining Accuracy mock here for standalone execution context
    name: str = "Accuracy"
    is_higher_better: bool = True
    weight: float = 1.0
    target_score: Optional[float] = 1.0
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_list = y_true.tolist() if isinstance(y_true, np.ndarray) else y_true
        y_pred_list = y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
        if not isinstance(y_true_list, list) or not isinstance(y_pred_list, list):
            raise TypeError("y_true and y_pred must be array-like.")
        if len(y_true_list) != len(y_pred_list):
            raise ValueError("y_true and y_pred must have the same length.")
        if not y_true_list:
            return 0.0
        correct_predictions = sum(1 for true, pred in zip(y_true_list, y_pred_list) if true == pred)
        return correct_predictions / len(y_true_list)
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        raise NotImplementedError

class UnifiedRunner:
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