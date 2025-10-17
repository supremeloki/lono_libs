import pandas as pd
import logging
from typing import Dict, Any, Optional
import io
_logger = logging.getLogger(__name__)
class ReportingGenerator:
    _HEADER_SEP = "=" * 100
    def _format_dataframe_for_display(self, df: pd.DataFrame) -> str:
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', None,
                               'display.float_format', '{:.4f}'.format):
            return df.to_string(index=False)
    def generate_performance_report_string(self, results_df: pd.DataFrame) -> str:
        if results_df.empty:
            return "No results to report."
        report_buffer = io.StringIO()
        report_buffer.write("\nModel Performance Comparison:\n")
        report_buffer.write(self._HEADER_SEP + "\n")
        for (model_name, model_type), group in results_df.groupby(['model_name', 'model_type']):
            report_buffer.write(f"\n--- Model: {model_name} ({model_type}) ---\n")
            for _, row in group.iterrows():
                report_buffer.write(f"  Metric: {row['metric_name']}\n")
                report_buffer.write(f"    Training Score: {row['Training Score']:.4f}\n")
                report_buffer.write(f"    Testing Score:  {row['Testing Score']:.4f}\n")
                report_buffer.write(f"    Difference:     {row['Difference']:.4f}\n")
                if 'target_score' in row.index and pd.notna(row['target_score']):
                    report_buffer.write(f"    Target Score:   {row['target_score']:.4f}\n")
                if 'difference_from_target' in row.index and pd.notna(row['difference_from_target']):
                    report_buffer.write(f"    Diff from target: {row['difference_from_target']:.4f}\n")
            if 'weighted_score' in group.columns and pd.notna(group['weighted_score'].iloc[0]):
                report_buffer.write(f"  Weighted Test Score: {group['weighted_score'].iloc[0]:.4f}\n")
        report_buffer.write(self._HEADER_SEP + "\n")
        return report_buffer.getvalue()
    def print_performance_report(self, results_df: pd.DataFrame):
        _logger.info(self.generate_performance_report_string(results_df))
    def generate_best_performance_summary_string(self, best_models_by_metric: Dict[str, pd.DataFrame]) -> str:
        if not best_models_by_metric:
            return "No best models to summarize."
        summary_buffer = io.StringIO()
        summary_buffer.write("\nBest Models Summary:\n")
        summary_buffer.write(self._HEADER_SEP + "\n")
        for metric_name, best_model_df in best_models_by_metric.items():
            if not best_model_df.empty:
                summary_buffer.write(f"--- Best Model for {metric_name} ---\n")
                summary_buffer.write(self._format_dataframe_for_display(best_model_df) + "\n\n")
            else:
                summary_buffer.write(f"--- No best model found for {metric_name} ---\n\n")
        summary_buffer.write(self._HEADER_SEP + "\n")
        return summary_buffer.getvalue()
    def print_best_performance_summary(self, best_models_by_metric: Dict[str, pd.DataFrame]):
        _logger.info(self.generate_best_performance_summary_string(best_models_by_metric))
    def generate_unified_report_string(self, results_df: pd.DataFrame, best_models_by_metric: Dict[str, pd.DataFrame]) -> str:
        report_buffer = io.StringIO()
        report_buffer.write(self.generate_performance_report_string(results_df))
        report_buffer.write("\n\n")
        report_buffer.write(self.generate_best_performance_summary_string(best_models_by_metric))
        return report_buffer.getvalue()
    def print_unified_report(self, results_df: pd.DataFrame, best_models_by_metric: Dict[str, pd.DataFrame]):
        _logger.info(self.generate_unified_report_string(results_df, best_models_by_metric))
