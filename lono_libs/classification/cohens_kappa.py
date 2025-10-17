import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split  # type: ignore
from lono_libs import UnifiedRunner
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.random.seed(42)
num_samples = 200
num_features = 5
X = pd.DataFrame(np.random.rand(num_samples, num_features) * 100, columns=[f'feature_{i}' for i in range(num_features)])
y_class = pd.Series(np.random.choice(['A', 'B', 'C'], num_samples))
y_reg = pd.Series(np.random.rand(num_samples) * 10 + X['feature_0'] * 0.5 + X['feature_1'] * 0.2 + np.random.randn(num_samples) * 2)
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.3, random_state=42
)
target_cohens_kappa_score = 1.0
preprocessing_config = {
    "add_polynomial_features": True,
    "poly_degree": 2,
    "apply_scaler": True
}
metric_target_overrides = {
    "CohensKappa": target_cohens_kappa_score
}
runner = UnifiedRunner(
    output_base_dir="lono_results_cohens_kappa_example",
    enable_logging=True,
    enable_visualizations=True,
    target_score_overrides=metric_target_overrides,
)
print("Starting LONO_LIBS UnifiedRunner pipeline execution...")
full_results_df, best_models_by_metric = runner.run_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train_class=y_train_class,
    y_test_class=y_test_class,
    y_train_reg=y_train_reg,
    y_test_reg=y_test_reg,
    preprocessing_config=preprocessing_config,
    metrics_to_evaluate=["CohensKappa"]
)
print("\n--- LONO_LIBS Pipeline Execution Complete ---")
runner.generate_reports_and_visualizations(
    full_results_df=full_results_df,
    best_models_by_metric=best_models_by_metric
)
print("\n--- Reports and Visualizations Generated (Check 'lono_results_cohens_kappa_example' directory) ---")
print("\nCohens Kappa Model Performance Comparison (from UnifiedRunner results):")
print("=" * 100)
cohens_kappa_results_df = full_results_df[full_results_df['metric_name'] == 'CohensKappa']
print(cohens_kappa_results_df[['Model', 'Type', 'Training Score', 'Testing Score', 'Difference', 'Score_Difference_From_Target']].to_string(index=False))
print("=" * 100)
if 'CohensKappa' in best_models_by_metric and not best_models_by_metric['CohensKappa'].empty:
    best_cohens_kappa_model = best_models_by_metric['CohensKappa'].iloc[0]
    print(f"\nBest Cohens Kappa Overall Model (Closest to {best_cohens_kappa_model['target_score']:.4f}):")
    print(f"Model: {best_cohens_kappa_model['Model']} ({best_cohens_kappa_model['Type']})")
    print(f"Testing Score: {best_cohens_kappa_model['Testing Score']:.4f}")
    if 'Score_Difference_From_Target' in best_cohens_kappa_model:
        print(f"Difference from target ({best_cohens_kappa_model['target_score']:.4f}): {best_cohens_kappa_model['Score_Difference_From_Target']:.4f}")
else:
    print("\nNo best Cohens Kappa model found in the results.")
print("\nExecution Finished. Check the 'lono_results_cohens_kappa_example' directory for detailed reports and plots.")
