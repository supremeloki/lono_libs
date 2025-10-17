import pandas as pd
from typing import Optional, Literal, Dict
class BestResultFinder:
    def __init__(self, results_df: pd.DataFrame):
        self._results_df = results_df.copy()
    def find_best_for_metric(self, metric_name: str, target: Optional[float] = None, goal: Optional[Literal["maximize", "minimize", "closest_to_target"]] = None) -> pd.DataFrame:
        metric_subset = self._results_df[self._results_df['metric_name'] == metric_name]
        if metric_subset.empty:
            return pd.DataFrame()
        current_target = target
        current_goal = goal
        if current_goal is None and 'optimization_goal' in metric_subset.columns:
            current_goal = metric_subset['optimization_goal'].iloc[0]
        if current_goal is None: # Fallback if no goal in df
            current_goal = "maximize" # Default to maximize
        if current_target is None and 'target_score' in metric_subset.columns:
            non_null_targets = metric_subset['target_score'].dropna()
            if not non_null_targets.empty:
                current_target = non_null_targets.iloc[0]
        if current_goal == "closest_to_target" and current_target is not None:
            metric_subset_copy = metric_subset.copy()
            metric_subset_copy['_proximity_score'] = abs(metric_subset_copy['Testing Score'] - current_target)
            return metric_subset_copy.nsmallest(1, '_proximity_score').drop(columns=['_proximity_score'])
        elif current_goal == "maximize":
            return metric_subset.nlargest(1, 'Testing Score')
        elif current_goal == "minimize":
            return metric_subset.nsmallest(1, 'Testing Score')
        else:
            return metric_subset.nlargest(1, 'Testing Score') # Final fallback
    def get_best(self, metric_name: str, target_score: Optional[float] = None) -> pd.DataFrame:
        return self.find_best_for_metric(metric_name, target=target_score)
    def get_all_best_models(self) -> Dict[str, pd.DataFrame]:
        best_models_map = {}
        for metric_name in self._results_df['metric_name'].unique():
            best_models_map[metric_name] = self.find_best_for_metric(metric_name)
        return best_models_map
