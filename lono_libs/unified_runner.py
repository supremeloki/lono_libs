import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Type, Tuple
import io
import os
from lono_libs import (
    get_preprocessing_pipeline, create_model_instance, get_classification_models, get_regression_models,
    Evaluator, IMetric,
    ReportingGenerator, SummaryReportGenerator, VisualizationGenerator,
    BestResultFinder, get_all_metrics # New function to get registered metrics
)
_logger = logging.getLogger(__name__)
class UnifiedRunner:
    def __init__(
        self,
        output_base_dir: str = "lono_results",
        enable_logging: bool = True,
        enable_visualizations: bool = True,
        target_score_overrides: Optional[Dict[str, float]] = None,
        metric_weights: Optional[Dict[str, float]] = None
    ):
        self._output_base_dir = output_base_dir
        os.makedirs(self._output_base_dir, exist_ok=True)
        self._enable_logging = enable_logging
        self._enable_visualizations = enable_visualizations
        self._target_score_overrides = target_score_overrides if target_score_overrides is not None else {}
        self._metric_weights = metric_weights if metric_weights is not None else {}
        if not self._enable_logging:
            _logger.setLevel(logging.CRITICAL)
        self._all_available_metrics = get_all_metrics()
        self._active_metrics: List[IMetric] = []
        _logger.info("UnifiedRunner initialized.")
    def _prepare_metrics_for_evaluator(self, requested_metric_names: Optional[List[str]] = None) -> List[IMetric]:
        active_metrics_instances: List[IMetric] = []
        metrics_to_use = requested_metric_names if requested_metric_names is not None else list(self._all_available_metrics.keys())
        for metric_name in metrics_to_use:
            metric_class = self._all_available_metrics.get(metric_name)
            if metric_class:
                metric_instance = metric_class()
                if metric_name in self._target_score_overrides:
                    metric_instance.target_score = self._target_score_overrides[metric_name]
                if metric_name in self._metric_weights:
                    metric_instance.weight = self._metric_weights[metric_name]
                active_metrics_instances.append(metric_instance)
            else:
                _logger.warning(f"Metric '{metric_name}' not found in registered metrics. Skipping.")
        if not active_metrics_instances:
            _logger.error("No active metrics prepared for evaluation.")
            raise ValueError("No metrics available for evaluation.")
        self._active_metrics = active_metrics_instances
        return self._active_metrics
    def run_pipeline(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train_class: pd.Series, # Assuming classification target is Series for simplicity
        y_test_class: pd.Series,
        y_train_reg: pd.Series, # Assuming regression target is Series
        y_test_reg: pd.Series,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        metrics_to_evaluate: Optional[List[str]] = None,
        random_state: int = 42,
        n_estimators: int = 100
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        _logger.info("Starting LONO_LIBS evaluation pipeline.")
        _logger.info("Applying data preprocessing...")
        preprocessing_config = preprocessing_config if preprocessing_config is not None else {}
        preprocessor = get_preprocessing_pipeline(**preprocessing_config)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        if isinstance(X_train_processed, np.ndarray):
            X_train_processed = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out(X_train.columns) if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_processed.shape[1])], index=X_train.index)
            X_test_processed = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out(X_test.columns) if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_test_processed.shape[1])], index=X_test.index)
        _logger.info("Creating models...")
        model_config = model_config if model_config is not None else {}
        classification_models_dict = get_classification_models(random_state=random_state, n_estimators=n_estimators, **model_config.get("classification", {}))
        regression_models_dict = get_regression_models(random_state=random_state, n_estimators=n_estimators, **model_config.get("regression", {}))
        active_metrics = self._prepare_metrics_for_evaluator(metrics_to_evaluate)
        evaluator = Evaluator(metrics=active_metrics)
        evaluation_data_entries: List[Dict[str, Any]] = []
        _logger.info("Evaluating classification models...")
        y_train_class_codes = y_train_class.astype('category').cat.codes
        y_test_class_codes = y_test_class.astype('category').cat.codes
        for name, model in classification_models_dict.items():
            _logger.debug(f"Training and evaluating classification model: {name}")
            try:
                model.fit(X_train_processed, y_train_class_codes)
                y_pred_train = model.predict(X_train_processed)
                y_pred_test = model.predict(X_test_processed)
                y_pred_proba_train = None
                y_pred_proba_test = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba_train = model.predict_proba(X_train_processed)
                        y_pred_proba_test = model.predict_proba(X_test_processed)
                    except AttributeError:
                        _logger.debug(f"Model {name} has predict_proba but it failed for some reason, proceeding without probas.")
                evaluation_data_entries.append({
                    'model_name': name,
                    'model_type': 'Classification',
                    'y_true': y_test_class_codes.values,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'y_pred_proba_train': y_pred_proba_train,
                    'y_pred_proba_test': y_pred_proba_test,
                    'fitted_model': model # Store fitted model for potential later use (e.g., saving)
                })
            except Exception as e:
                _logger.error(f"Error with classification model {name}: {e}")
        _logger.info("Evaluating regression models...")
        for name, model in regression_models_dict.items():
            _logger.debug(f"Training and evaluating regression model: {name}")
            try:
                model.fit(X_train_processed, y_train_reg)
                y_pred_train = model.predict(X_train_processed)
                y_pred_test = model.predict(X_test_processed)
                evaluation_data_entries.append({
                    'model_name': name,
                    'model_type': 'Regression',
                    'y_true': y_test_reg.values,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'y_pred_proba_train': None, # Regression models don't typically have probas
                    'y_pred_proba_test': None,
                    'fitted_model': model
                })
            except Exception as e:
                _logger.error(f"Error with regression model {name}: {e}")
        full_results_df, best_models_by_metric = evaluator.evaluate_models(evaluation_data_entries)
        full_results_df['Difference'] = full_results_df['Testing Score'] - full_results_df['Training Score']
        full_results_df['Score_Difference_From_Target'] = np.nan
        for metric_name in full_results_df['metric_name'].unique():
            metric_target_score = self._target_score_overrides.get(metric_name)
            if metric_target_score is None:
                for metric_instance in self._active_metrics:
                    if metric_instance.name == metric_name:
                        metric_target_score = metric_instance.target_score
                        break
            if metric_target_score is not None:
                mask = full_results_df['metric_name'] == metric_name
                full_results_df.loc[mask, 'Score_Difference_From_Target'] = \
                    abs(full_results_df.loc[mask, 'Testing Score'] - metric_target_score)
        _logger.info("Evaluation pipeline completed.")
        return full_results_df, best_models_by_metric
    def generate_reports_and_visualizations(
        self,
        full_results_df: pd.DataFrame,
        best_models_by_metric: Dict[str, pd.DataFrame]
    ):
        if full_results_df.empty:
            if self._enable_logging: _logger.info("No results to display. Skipping all reports and visualizations.")
            return
        _logger.info("Starting comprehensive report and visualization generation...")
        if self._enable_logging:
            reporter = ReportingGenerator()
            reporter.print_performance_report(full_results_df)
            reporter.print_best_performance_summary(best_models_by_metric)
        else:
            _logger.debug("Logging disabled, skipping basic reports to console.")
        if self._enable_logging:
            summary_reporter = SummaryReportGenerator(full_results_df, best_models_by_metric)
            summary_reporter.print_summary_report() # This will also call its create_spider_plot internally
        else:
            _logger.debug("Logging disabled, skipping summary reports to console.")
        if self._enable_visualizations:
            _logger.info("Generating additional visualizations...")
            viz_generator = VisualizationGenerator(full_results_df, best_models_by_metric, output_dir=os.path.join(self._output_base_dir, "plots"))
            viz_generator.generate_all_plots(target_overrides=self._target_score_overrides)
        else:
            _logger.info("Visualizations disabled.")
        _logger.info("Comprehensive report and visualization generation complete.")
