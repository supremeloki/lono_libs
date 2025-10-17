import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore

from lono_libs import UnifiedRunner
from lono_libs.classification import Accuracy, F1Score, CohensKappa
from lono_libs.regression import MAE, MSE, R2Score


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('evaluation.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def generate_sample_data(num_samples: int = 200, num_features: int = 5,
                        test_size: float = 0.3, random_seed: int = 42):
    """
    Generate sample classification and regression data.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features
        test_size: Proportion of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg)
    """
    np.random.seed(random_seed)

    # Generate features
    X = pd.DataFrame(
        np.random.rand(num_samples, num_features) * 100,
        columns=[f'feature_{i}' for i in range(num_features)]
    )

    # Generate classification target (multi-class)
    y_class = pd.Series(np.random.choice([0, 1, 2], num_samples))

    # Generate regression target with some correlation to features
    y_reg = pd.Series(
        np.random.rand(num_samples) * 10 +
        X['feature_0'] * 0.5 + X['feature_1'] * 0.2 +
        np.random.randn(num_samples) * 2
    )

    # Split data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=test_size, random_state=random_seed, stratify=y_class if len(y_class.unique()) > 1 else None
    )

    logger.info(f"Generated dataset: {num_samples} samples, {num_features} features")
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg


def get_default_config() -> Dict:
    """Get default configuration for preprocessing and metrics."""
    return {
        'preprocessing': {
            "add_polynomial_features": True,
            "poly_degree": 2,
            "apply_scaler": True
        },
        'target_scores': {
            "Accuracy": 1.0,
            "F1Score": 0.95,
            "LogLoss": 0.5,
            "CohensKappa": 0.8,
            "MAE": 1.5,
            "MSE": 5.0,
            "R2Score": 0.9
        }
    }


def validate_metrics(metrics: List[str]) -> List[str]:
    """Validate that requested metrics are available."""
    available_metrics = {
        'classification': ['Accuracy', 'F1Score', 'LogLoss', 'CohensKappa'],
        'regression': ['MAE', 'MSE', 'R2Score']
    }

    valid_metrics = []
    for metric in metrics:
        if metric in available_metrics['classification'] or metric in available_metrics['regression']:
            valid_metrics.append(metric)
        else:
            logger.warning(f"Metric '{metric}' is not available. Skipping.")

    if not valid_metrics:
        raise ValueError("No valid metrics specified. Available metrics: " +
                        ", ".join(available_metrics['classification'] + available_metrics['regression']))

    return valid_metrics


def run_evaluation(output_dir: str = "evaluation_results",
                  metrics: Optional[List[str]] = None,
                  config: Optional[Dict] = None) -> None:
    """
    Run the evaluation pipeline.

    Args:
        output_dir: Directory to save results
        metrics: List of metrics to evaluate
        config: Configuration dictionary
    """
    logger.info("Starting model evaluation...")

    # Get configuration
    if config is None:
        config = get_default_config()

    # Validate metrics
    if metrics is None:
        metrics = list(config['target_scores'].keys())
    metrics = validate_metrics(metrics)

    logger.info(f"Evaluating metrics: {', '.join(metrics)}")

    # Generate sample data
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = generate_sample_data()

    # Filter target scores to only include requested metrics
    target_scores = {k: v for k, v in config['target_scores'].items() if k in metrics}

    # Initialize UnifiedRunner
    runner = UnifiedRunner(
        output_base_dir=output_dir,
        enable_logging=True,
        enable_visualizations=True,
        target_score_overrides=target_scores
    )

    try:
        # Run pipeline
        logger.info("Running evaluation pipeline...")
        full_results_df, best_models_by_metric = runner.run_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train_class=y_train_class,
            y_test_class=y_test_class,
            y_train_reg=y_train_reg,
            y_test_reg=y_test_reg,
            preprocessing_config=config['preprocessing'],
            metrics_to_evaluate=metrics
        )

        logger.info("Pipeline execution completed successfully.")

        # Generate reports and visualizations
        logger.info("Generating reports and visualizations...")
        runner.generate_reports_and_visualizations(
            full_results_df=full_results_df,
            best_models_by_metric=best_models_by_metric
        )

        # Display results summary
        display_results_summary(full_results_df, best_models_by_metric, metrics, output_dir)

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def display_results_summary(full_results_df: pd.DataFrame,
                            best_models_by_metric: Dict[str, pd.DataFrame],
                            metrics_evaluated: List[str],
                            output_dir: str) -> None:
    """Display a summary of evaluation results."""
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)

    # Metrics where lower score is better
    lower_better_metrics = {"LogLoss", "MAE", "MSE"}

    for metric in metrics_evaluated:
        print(f"\n{metric} Results:")
        print("-" * 40)

        # Get metric results
        metric_results = full_results_df[full_results_df['metric_name'] == metric]
        if metric_results.empty:
            print(f"No results found for {metric}.")
            continue

        # Display top 5 models (best performing)
        if metric in lower_better_metrics:
            top_results = metric_results.nsmallest(5, 'Testing Score')[
                ['Model', 'Type', 'Training Score', 'Testing Score', 'Difference']
            ]
        else:
            top_results = metric_results.nlargest(5, 'Testing Score')[
                ['Model', 'Type', 'Training Score', 'Testing Score', 'Difference']
            ]
        print(top_results.to_string(index=False))

        # Display best model
        if metric in best_models_by_metric and not best_models_by_metric[metric].empty:
            best_model = best_models_by_metric[metric].iloc[0]
            target_score = best_model.get('target_score', 'N/A')
            print(f"\nBest {metric} Model (Target: {target_score}):")
            print(f"  Model: {best_model['Model']} ({best_model['Type']})")
            print(f"  Testing Score: {best_model['Testing Score']:.4f}")
            if 'Score_Difference_From_Target' in best_model:
                diff = best_model['Score_Difference_From_Target']
                print(f"  Difference from Target: {diff:.4f}")
        else:
            print(f"\nNo best {metric} model found.")

    print("\n" + "="*80)
    print(f"Detailed results saved to '{output_dir}' directory")
    print("Check the directory for comprehensive reports, plots, and logs.")
    print("="*80)


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive model evaluations using lono_libs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py
  python run_evaluation.py --metrics Accuracy F1Score MAE
  python run_evaluation.py --output_dir my_results --metrics Accuracy MAE
        """
    )

    parser.add_argument(
        '--metrics', '-m',
        nargs='+',
        help='Metrics to evaluate (default: all available)',
        choices=['Accuracy', 'F1Score', 'LogLoss', 'CohensKappa', 'MAE', 'MSE', 'R2Score']
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )

    args = parser.parse_args()

    try:
        run_evaluation(
            output_dir=args.output_dir,
            metrics=args.metrics
        )
        logger.info("Evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()