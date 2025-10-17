import pandas as pd
import logging
import io
from typing import Dict, Any, List, Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
_logger = logging.getLogger(__name__)
class SummaryReportGenerator:
    _SECTION_SEPARATOR = "=" * 100
    _EXCELLENCE_TOLERANCE = 1e-3 # Numerical tolerance for 'closest_to_target' metrics
    def __init__(self, full_results_df: pd.DataFrame, best_models_by_metric: Dict[str, pd.DataFrame]):
        self._full_results_df = full_results_df
        self._best_models_by_metric = best_models_by_metric
    @staticmethod
    def _display_dataframe_as_string(df: pd.DataFrame) -> str:
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', None,
                               'display.float_format', '{:.4f}'.format):
            return df.to_string(index=False)
    def _generate_weighted_ranking_string(self) -> str:
        if 'weighted_score' not in self._full_results_df.columns or self._full_results_df['weighted_score'].isnull().all():
            return "No weighted scores available for ranking.\n"
        weighted_df = self._full_results_df[['model_name', 'model_type', 'weighted_score']].drop_duplicates().copy()
        if weighted_df.empty:
            return "No weighted scores to display for ranking.\n"
        weighted_df['Rank'] = weighted_df['weighted_score'].rank(ascending=False, method='min')
        weighted_df = weighted_df.sort_values(by='Rank').reset_index(drop=True)
        report_content = io.StringIO()
        report_content.write("\nWeighted Model Ranking:\n")
        report_content.write(self._SECTION_SEPARATOR + "\n")
        report_content.write(self._display_dataframe_as_string(weighted_df) + "\n")
        report_content.write(self._SECTION_SEPARATOR + "\n")
        return report_content.getvalue()
    def _generate_excellence_summary_string(self) -> str:
        excellence_data = []
        unique_models = self._full_results_df[['model_name', 'model_type']].drop_duplicates()
        for _, model_identity in unique_models.iterrows():
            model_name = model_identity['model_name']
            model_type = model_identity['model_type']
            model_metrics_subset = self._full_results_df[
                (self._full_results_df['model_name'] == model_name) &
                (self._full_results_df['model_type'] == model_type)
            ].copy()
            met_target_count = 0
            total_evaluable_metrics = 0
            if 'target_score' in model_metrics_subset.columns and 'optimization_goal' in model_metrics_subset.columns:
                for _, metric_record in model_metrics_subset.iterrows():
                    test_score = metric_record['Testing Score']
                    target_score = metric_record['target_score']
                    optimization_goal = metric_record['optimization_goal']
                    if pd.notna(target_score):
                        total_evaluable_metrics += 1
                        if optimization_goal == "maximize" and test_score >= target_score:
                            met_target_count += 1
                        elif optimization_goal == "minimize" and test_score <= target_score:
                            met_target_count += 1
                        elif optimization_goal == "closest_to_target" and abs(test_score - target_score) <= self._EXCELLENCE_TOLERANCE:
                            met_target_count += 1
            excellence_data.append({
                'Model': model_name,
                'Type': model_type,
                'Metrics Meeting Target': f"{met_target_count}/{total_evaluable_metrics}"
            })
        excellence_df = pd.DataFrame(excellence_data)
        if excellence_df.empty:
            return "No models found that meet specified target thresholds.\n"
        excellence_df['_sort_ratio'] = excellence_df['Metrics Meeting Target'].apply(lambda x: int(x.split('/')[0]) / (int(x.split('/')[1]) if int(x.split('/')[1]) > 0 else 1))
        excellence_df = excellence_df.sort_values(by='_sort_ratio', ascending=False).drop(columns=['_sort_ratio']).reset_index(drop=True)
        report_content = io.StringIO()
        report_content.write("\nModels Meeting Target Thresholds (Excellent Performance):\n")
        report_content.write(self._SECTION_SEPARATOR + "\n")
        report_content.write(self._display_dataframe_as_string(excellence_df) + "\n")
        report_content.write(self._SECTION_SEPARATOR + "\n")
        return report_content.getvalue()
    def _generate_best_per_metric_summary_string(self) -> str:
        summary_content = io.StringIO()
        summary_content.write("\nBest Overall Algorithm Summary (Per Metric):\n")
        summary_content.write(self._SECTION_SEPARATOR + "\n")
        if not self._best_models_by_metric:
            summary_content.write("No best models identified per metric.\n")
            summary_content.write(self._SECTION_SEPARATOR + "\n")
            return summary_content.getvalue()
        for metric_name, best_model_df in self._best_models_by_metric.items():
            if not best_model_df.empty:
                summary_content.write(f"--- Best for {metric_name} ---\n")
                summary_content.write(self._display_dataframe_as_string(best_model_df.head(1)) + "\n\n")
            else:
                summary_content.write(f"--- No Best Model for {metric_name} ---\n\n")
        summary_content.write(self._SECTION_SEPARATOR + "\n")
        return summary_content.getvalue()
    def _prepare_spider_plot_data(self) -> tuple[List[str], List[str], Dict[str, List[float]], List[float]]:
        unique_metrics = self._full_results_df['metric_name'].unique().tolist()
        if 'weighted_score' in self._full_results_df.columns and not self._full_results_df['weighted_score'].isnull().all():
            top_models_df = self._full_results_df[['model_name', 'weighted_score']].drop_duplicates().nlargest(3, 'weighted_score')
            top_model_names = top_models_df['model_name'].tolist()
        else:
            avg_scores = self._full_results_df.groupby('model_name')['Testing Score'].mean().nlargest(3)
            top_model_names = avg_scores.index.tolist()
        if not top_model_names or not unique_metrics:
            return [], [], {}, []
        metric_normalization_ranges = {}
        for metric in unique_metrics:
            scores = self._full_results_df[self._full_results_df['metric_name'] == metric]['Testing Score']
            if not scores.empty and scores.max() > scores.min():
                metric_normalization_ranges[metric] = (scores.min(), scores.max())
            else:
                metric_normalization_ranges[metric] = (0, 1) # Default to 0-1 range
        model_normalized_scores: Dict[str, List[float]] = {model: [] for model in top_model_names}
        for metric in unique_metrics:
            min_val, max_val = metric_normalization_ranges[metric]
            range_val = max_val - min_val if max_val > min_val else 1
            for model_name in top_model_names:
                score_record = self._full_results_df[
                    (self._full_results_df['model_name'] == model_name) &
                    (self._full_results_df['metric_name'] == metric)
                ]
                if not score_record.empty:
                    score = score_record['Testing Score'].iloc[0]
                    normalized_score = (score - min_val) / range_val
                    model_normalized_scores[model_name].append(normalized_score)
                else:
                    model_normalized_scores[model_name].append(0)
        angles = np.linspace(0, 2 * np.pi, len(unique_metrics), endpoint=False).tolist()
        angles += angles[:1] # Close the loop
        return top_model_names, unique_metrics, model_normalized_scores, angles
    def create_spider_plot(self, plot_filename: str = "top_models_spider_plot.png"):
        model_labels, metric_labels, model_data, angles = self._prepare_spider_plot_data()
        if not model_labels or not metric_labels:
            _logger.warning("Insufficient data for spider plot generation.")
            return
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.colormaps['Set2'](np.linspace(0, 1, len(model_labels)))
        for i, model in enumerate(model_labels):
            scores = model_data[model]
            if len(scores) != len(angles) - 1:
                continue
            scores_closed = scores + scores[:1]
            ax.plot(angles, scores_closed, linewidth=2, linestyle='solid', label=model, color=colors[i], marker='o', markersize=6)
            ax.fill(angles, scores_closed, color=colors[i], alpha=0.15)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10, rotation=20, ha='right', color='darkslategray')
        ax.tick_params(axis='x', which='major', pad=15)
        ax.set_title("Top Model Performance Across Key Metrics", y=1.1, fontsize=14, fontweight='bold')
        legend_patches = [
            Patch(facecolor=colors(i), edgecolor='darkslategray', label=model_labels[i])
            for i in range(len(model_labels))
        ]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Models", frameon=False)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        _logger.info(f"Spider plot saved to {plot_filename}")
    def generate_summary_report_string(self) -> str:
        report_buffer = io.StringIO()
        report_buffer.write(self._generate_weighted_ranking_string())
        report_buffer.write("\n")
        report_buffer.write(self._generate_excellence_summary_string())
        report_buffer.write("\n")
        report_buffer.write(self._generate_best_per_metric_summary_string())
        return report_buffer.getvalue()
    def print_summary_report(self):
        _logger.info(self.generate_summary_report_string())
        self.create_spider_plot()
