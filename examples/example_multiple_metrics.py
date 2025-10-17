import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from lono_libs import UnifiedRunner
from lono_libs.classification import Accuracy, F1Score  # type: ignore
from lono_libs.regression import MAE  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set pandas display options for better output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)


def generate_sample_data(num_samples: int = 200, num_features: int = 5,
                        test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Generate sample classification and regression data.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        test_size: Proportion of data for testing

    Returns:
        Tuple of (X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)
    """
    np.random.seed(42)

    # Generate features
    X = pd.DataFrame(
        np.random.rand(num_samples, num_features) * 100,
        columns=[f'feature_{i}' for i in range(num_features)]
    )

    # Generate classification target
    y_class = pd.Series(np.random.choice(['A', 'B', 'C'], num_samples))

    # Generate regression target with some correlation to features
    y_reg = pd.Series(
        np.random.rand(num_samples) * 10 +
        X['feature_0'] * 0.5 + X['feature_1'] * 0.2 +
        np.random.randn(num_samples) * 2
    )

    # Split data
    try:
        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=42, stratify=y_class
        )
        return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg
    except ValueError as e:
        logger.error(f"Error splitting data: {e}")
        raise


def setup_runner_config(target_scores: Optional[Dict[str, float]] = None) -> Tuple[Dict, Dict]:
    """
    Set up preprocessing and metric target configurations.

    Args:
        target_scores: Optional dictionary of target scores for metrics

    Returns:
        Tuple of (preprocessing_config, metric_target_overrides)
    """
    preprocessing_config = {
        "add_polynomial_features": True,
        "poly_degree": 2,
        "apply_scaler": True
    }

    default_targets = {
        "Accuracy": 1.0,
        "F1Score": 0.95,
        "MAE": 1.5
    }

    metric_target_overrides = target_scores or default_targets

    return preprocessing_config, metric_target_overrides


def run_multiple_metrics_evaluation(metrics_to_evaluate: List[str]) -> None:
    """
    Run the multiple metrics evaluation pipeline.

    Args:
        metrics_to_evaluate: List of metric names to evaluate
    """
    logger.info("Starting multiple metrics evaluation...")

    # Generate sample data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = generate_sample_data()

    # Setup configurations
    preprocessing_config, metric_target_overrides = setup_runner_config()

    # Filter target overrides to only include metrics being evaluated
    filtered_overrides = {k: v for k, v in metric_target_overrides.items() if k in metrics_to_evaluate}

    # Initialize UnifiedRunner
    runner = UnifiedRunner(
        output_base_dir="multiple_metrics_results",
        enable_logging=True,
        enable_visualizations=True,
        target_score_overrides=filtered_overrides,
    )

    try:
        # Run pipeline
        full_results_df, best_models_by_metric = runner.run_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train_class=y_train_class,
            y_test_class=y_test_class,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            preprocessing_config=preprocessing_config,
            metrics_to_evaluate=metrics_to_evaluate
        )

        logger.info("Pipeline execution completed successfully.")

        # Generate reports and visualizations
        runner.generate_reports_and_visualizations(
            full_results_df=full_results_df,
            best_models_by_metric=best_models_by_metric
        )

        # Display results
        display_results(full_results_df, best_models_by_metric, metrics_to_evaluate)

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


def display_results(full_results_df: pd.DataFrame,
                   best_models_by_metric: Dict[str, pd.DataFrame],
                   metrics_evaluated: List[str]) -> None:
    """
    Display evaluation results for each metric.

    Args:
        full_results_df: Full results dataframe
        best_models_by_metric: Best models by metric
        metrics_evaluated: List of evaluated metrics
    """
    print("\nMultiple Metrics Evaluation Results:")
    print("=" * 120)

    for metric in metrics_evaluated:
        print(f"\n{metric} Results:")
        print("-" * 50)
        metric_results_df = full_results_df[full_results_df['metric_name'] == metric]

        if metric_results_df.empty:
            print(f"No results found for {metric}.")
            continue

        # Display key columns
        columns_to_display = ['Model', 'Type', 'Training Score', 'Testing Score',
                            'Difference', 'Score_Difference_From_Target']
        available_columns = [col for col in columns_to_display if col in metric_results_df.columns]
        if available_columns:
            print(metric_results_df[available_columns].to_string(index=False))
        else:
            print("No suitable columns found for display.")

        # Display best model for this metric
        if metric in best_models_by_metric and not best_models_by_metric[metric].empty:
            best_model = best_models_by_metric[metric].iloc[0]
            target_score = best_model.get('target_score', None)
            target_str = f"{target_score:.4f}" if target_score is not None else "N/A"
            print(f"\nBest {metric} Model (Target: {target_str}):")
            print(f"  Model: {best_model['Model']} ({best_model['Type']})")
            print(f"  Testing Score: {best_model['Testing Score']:.4f}")
            if 'Score_Difference_From_Target' in best_model:
                print(f"  Difference from Target: {best_model['Score_Difference_From_Target']:.4f}")
        else:
            print(f"\nNo best {metric} model found.")

    print("\n" + "=" * 120)
    print("Execution finished. Check 'multiple_metrics_results' directory for detailed reports and plots.")


if __name__ == "__main__":
    # Define metrics to evaluate (can be modified as needed)
    metrics_to_evaluate = ["Accuracy", "F1Score", "MAE"]

    try:
        run_multiple_metrics_evaluation(metrics_to_evaluate)
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        exit(1)