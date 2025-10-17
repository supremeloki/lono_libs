import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Protocol, runtime_checkable, Union
_logger = logging.getLogger(__name__)
@runtime_checkable
class IMetric(Protocol):
    name: str
    is_higher_better: bool
    weight: float
    target_score: Optional[float]
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]: ...
    def calculate_from_proba(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Union[float, np.ndarray]: ... # For metrics like ROC_AUC, LogLoss
class Evaluator:
    def __init__(self, metrics: List[IMetric]):
        self._metrics = {m.name: m for m in metrics}
    def _evaluate_model_on_data(self, model_name: str, model_type: str, y_true: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray, y_pred_proba_train: Optional[np.ndarray] = None, y_pred_proba_test: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        results = []
        for metric_name, metric_obj in self._metrics.items():
            try:
                train_score = metric_obj.calculate(y_true, y_pred_train)
                test_score = metric_obj.calculate(y_true, y_pred_test)
                if y_pred_proba_train is not None and y_pred_proba_test is not None:
                    if hasattr(metric_obj, 'calculate_from_proba'):
                        train_score_proba = metric_obj.calculate_from_proba(y_true, y_pred_proba_train)
                        test_score_proba = metric_obj.calculate_from_proba(y_true, y_pred_proba_test)
                        if metric_name.lower() in ['logloss', 'roc_auc']:
                            train_score = train_score_proba
                            test_score = test_score_proba
                results.append({
                    'model_name': model_name,
                    'model_type': model_type,
                    'metric_name': metric_obj.name,
                    'Training Score': train_score,
                    'Testing Score': test_score,
                    'is_higher_better': metric_obj.is_higher_better,
                    'weight': metric_obj.weight,
                    'target_score': metric_obj.target_score
                })
            except Exception as e:
                _logger.error(f"Error evaluating {metric_obj.name} for {model_name}: {e}")
        return results
    def _calculate_weighted_average_score(self, model_results: List[Dict[str, Any]]) -> float:
        total_weighted_score = 0.0
        total_weight = 0.0
        for res in model_results:
            metric_obj = self._metrics.get(res['metric_name'])
            if not metric_obj or metric_obj.weight == 0:
                continue
            score = res['Testing Score']
            normalized_score = score
            if not metric_obj.is_higher_better:
                normalized_score = 1.0 / (1.0 + score) if score >= 0 else 0.0 # Prevent division by zero / negative scores
            total_weighted_score += normalized_score * metric_obj.weight
            total_weight += metric_obj.weight
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    def evaluate_models(self, evaluation_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        all_results_list: List[Dict[str, Any]] = []
        for data_entry in evaluation_data:
            model_name = data_entry['model_name']
            model_type = data_entry['model_type']
            y_true = data_entry['y_true']
            y_pred_train = data_entry['y_pred_train']
            y_pred_test = data_entry['y_pred_test']
            y_pred_proba_train = data_entry.get('y_pred_proba_train')
            y_pred_proba_test = data_entry.get('y_pred_proba_test')
            model_metrics_results = self._evaluate_model_on_data(
                model_name, model_type, y_true, y_pred_train, y_pred_test,
                y_pred_proba_train, y_pred_proba_test
            )
            all_results_list.extend(model_metrics_results)
        if not all_results_list:
            return pd.DataFrame(), {}
        full_results_df = pd.DataFrame(all_results_list)
        model_weighted_scores = {}
        for model_name in full_results_df['model_name'].unique():
            model_df_results_list = full_results_df[full_results_df['model_name'] == model_name].to_dict(orient='records')
            weighted_score = self._calculate_weighted_average_score(model_df_results_list)
            model_weighted_scores[model_name] = weighted_score
        full_results_df['weighted_score'] = full_results_df['model_name'].map(model_weighted_scores)
        best_models_by_metric: Dict[str, pd.DataFrame] = {}
        for metric_name in full_results_df['metric_name'].unique():
            metric_df = full_results_df[full_results_df['metric_name'] == metric_name].copy()
            if metric_df.empty: continue
            is_higher_better = metric_df['is_higher_better'].iloc[0]
            if is_higher_better:
                best_model_for_metric = metric_df.loc[metric_df['Testing Score'].idxmax()].to_frame().T
            else:
                best_model_for_metric = metric_df.loc[metric_df['Testing Score'].idxmin()].to_frame().T
            best_models_by_metric[metric_name] = best_model_for_metric
        return full_results_df, best_models_by_metric
