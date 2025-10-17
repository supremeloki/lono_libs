from click import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Literal
from matplotlib.patches import Patch
import os
_logger = logging.getLogger(__name__)
class VisualizationGenerator:
    _DEFAULT_OUTPUT_DIR = "lono_visualizations"
    def __init__(self, full_results_df: pd.DataFrame, best_models_by_metric: Dict[str, pd.DataFrame], output_dir: Optional[str] = None):
        self._full_results_df = full_results_df
        self._best_models_by_metric = best_models_by_metric
        self._output_dir = output_dir if output_dir else self._DEFAULT_OUTPUT_DIR
        os.makedirs(self._output_dir, exist_ok=True)
    def _plot_metric_comparison_bar(self, metric_name: str, target_score: Optional[float], best_model_name: Optional[str], filename: str) -> None:
        metric_df = self._full_results_df[self._full_results_df['metric_name'] == metric_name].copy()
        if metric_df.empty or 'Training Score' not in metric_df.columns or 'Testing Score' not in metric_df.columns:
            _logger.warning(f"Insufficient data for bar plot for metric: {metric_name}")
            return
        metric_df_sorted = metric_df.sort_values(by='Testing Score', ascending=False).reset_index(drop=True)
        model_names = metric_df_sorted['model_name']
        num_models = len(model_names)
        fig, ax = plt.subplots(figsize=(max(10, num_models * 0.5), 6))
        width = 0.35
        x = np.arange(num_models)
        bars_train = ax.bar(x - width / 2, metric_df_sorted['Training Score'], width, label='Train', color='skyblue')
        bars_test = ax.bar(x + width / 2, metric_df_sorted['Testing Score'], width, label='Test', color='salmon', alpha=0.8)
        if best_model_name:
            best_idx = model_names[model_names == best_model_name].index
            if not best_idx.empty:
                bars_test[best_idx[0]].set_edgecolor('gold')
                bars_test[best_idx[0]].set_linewidth(3)
        if target_score is not None:
            ax.axhline(y=target_score, color='green', linestyle='--', label=f'Target ({target_score:.4f})')
        ax.set_title(f'{metric_name} Model Performance Comparison')
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self._output_dir, filename))
        plt.close(fig)
        _logger.info(f"Generated bar plot for {metric_name} at {os.path.join(self._output_dir, filename)}")
    def _plot_train_test_difference(self, metric_name: str, filename: str) -> None:
        metric_df = self._full_results_df[self._full_results_df['metric_name'] == metric_name].copy()
        if metric_df.empty or 'Training Score' not in metric_df.columns or 'Testing Score' not in metric_df.columns:
            _logger.warning(f"Insufficient data for difference plot for metric: {metric_name}")
            return
        metric_df['Score Difference (Train - Test)'] = metric_df['Training Score'] - metric_df['Testing Score']
        metric_df_sorted = metric_df.sort_values(by='Score Difference (Train - Test)', ascending=False).reset_index(drop=True)
        model_names = metric_df_sorted['model_name']
        fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 0.5), 5))
        ax.barh(model_names, metric_df_sorted['Score Difference (Train - Test)'], color='teal')
        ax.set_xlabel('Score Difference (Train - Test)')
        ax.set_title(f'Overfitting/Underfitting Indicator ({metric_name})')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self._output_dir, filename))
        plt.close(fig)
        _logger.info(f"Generated difference plot for {metric_name} at {os.path.join(self._output_dir, filename)}")
    def _plot_all_model_scores_vs_target(self, metric_name: str, target_score: Optional[float], filename: str) -> None:
        metric_df = self._full_results_df[self._full_results_df['metric_name'] == metric_name].copy()
        if metric_df.empty or 'Testing Score' not in metric_df.columns:
            _logger.warning(f"Insufficient data for target comparison plot for metric: {metric_name}")
            return
        metric_df_sorted = metric_df.sort_values(by='Testing Score', ascending=False).reset_index(drop=True)
        model_names = metric_df_sorted['model_name']
        fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 0.5), 5))
        ax.bar(model_names, metric_df_sorted['Testing Score'], color='darkorchid')
        if target_score is not None:
            ax.axhline(y=target_score, color='red', linestyle='--', label=f'Target ({target_score:.4f})')
        ax.set_title(f'Testing Scores vs Target ({metric_name})')
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self._output_dir, filename))
        plt.close(fig)
        _logger.info(f"Generated target comparison plot for {metric_name} at {os.path.join(self._output_dir, filename)}")
    def _prepare_spider_plot_data(self) -> Tuple[List[str], List[str], Dict[str, List[float]], List[float]]:
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
                metric_normalization_ranges[metric] = (0, 1)
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
        angles += angles[:1]
        return top_model_names, unique_metrics, model_normalized_scores, angles
    def _create_spider_plot(self, filename: str = "top_models_spider_plot.png"):
        model_labels, metric_labels, model_data, angles = self._prepare_spider_plot_data()
        if not model_labels or not metric_labels:
            _logger.warning("Insufficient data for spider plot generation.")
            return
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        colors = plt.colormaps['Set2'](np.linspace(0, 1, len(model_labels)))
        for i, model in enumerate(model_labels):
            scores = model_data[model]
            if len(scores) != len(angles) - 1: continue
            scores_closed = scores + scores[:1]
            ax.plot(angles, scores_closed, linewidth=2, linestyle='solid', label=model, color=colors(i), marker='o', markersize=6)
            ax.fill(angles, scores_closed, color=colors(i), alpha=0.15)
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
        plt.savefig(os.path.join(self._output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close(fig)
        _logger.info(f"Spider plot saved to {os.path.join(self._output_dir, filename)}")
    def generate_all_plots(self, target_overrides: Optional[Dict[str, float]] = None):
        target_overrides = target_overrides if target_overrides is not None else {}
        unique_metrics = self._full_results_df['metric_name'].unique()
        for metric in unique_metrics:
            best_model_for_metric_df = self._best_models_by_metric.get(metric)
            best_model_name = best_model_for_metric_df.iloc[0]['model_name'] if best_model_for_metric_df is not None and not best_model_for_metric_df.empty else None
            target_score_from_df = self._full_results_df[self._full_results_df['metric_name'] == metric]['target_score'].dropna().iloc[0] if 'target_score' in self._full_results_df.columns and not self._full_results_df[self._full_results_df['metric_name'] == metric]['target_score'].dropna().empty else None
            target_score = target_overrides.get(metric, target_score_from_df)
            self._plot_metric_comparison_bar(metric, target_score, best_model_name, f"{metric.lower().replace(' ', '_')}_bar_plot.png")
            self._plot_train_test_difference(metric, f"{metric.lower().replace(' ', '_')}_difference_plot.png")
            self._plot_all_model_scores_vs_target(metric, target_score, f"{metric.lower().replace(' ', '_')}_target_comparison_plot.png")
        self._create_spider_plot(filename="top_models_spider_plot.png")
