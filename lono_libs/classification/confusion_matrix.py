import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from typing import List
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from lono_libs import UnifiedRunner

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.random.seed(42)

num_samples = 1000
num_features = 7
base_features = np.random.randn(num_samples, 4)
X_array = np.column_stack([
    base_features,
    base_features[:, 0] * 0.5 + np.random.randn(num_samples) * 0.3,
    base_features[:, 1] + base_features[:, 2] * 0.4 + np.random.randn(num_samples) * 0.2,
    np.sin(base_features[:, 0]) + np.random.randn(num_samples) * 0.1
]) * 10 + 50
X: pd.DataFrame = pd.DataFrame(X_array, columns=[f'feature_{i}' for i in range(num_features)])

class_centers = np.array([[0, 0, 0, 0], [2, 2, 2, 2], [-1, -1, -1, -1], [1, -1, 1, -1]])
y_class_list: List[str] = []
for i in range(num_samples):
    point = base_features[i, :4]
    distances = np.sum((class_centers - point)**2, axis=1)
    class_idx = int(np.argmin(distances))
    if np.random.rand() < 0.15:
        class_idx = int(np.random.randint(4))
    y_class_list.append(['A', 'B', 'C', 'D'][class_idx])

y_class = pd.Series(y_class_list)
y_reg = pd.Series(np.random.rand(num_samples) * 10 + X.iloc[:, 0] * 0.5 + np.random.randn(num_samples) * 2)

X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class_list, y_reg, test_size=0.3, random_state=42, stratify=y_class_list
)

target_accuracy_score = 0.85
preprocessing_config = {
    "add_polynomial_features": True,
    "poly_degree": 2,
    "apply_scaler": True,
    "feature_selection": {"method": "mutual_info_classif", "k": 10}
}
metric_target_overrides = {"Accuracy": target_accuracy_score, "ConfusionMatrix": None}

runner = UnifiedRunner(
    output_base_dir="lono_results_confusion_matrix_example",
    enable_logging=True,
    enable_visualizations=True,
    target_score_overrides={"Accuracy": target_accuracy_score}
)

print("=" * 120)
print("LONO_LIBS Confusion Matrix Example - Advanced Multi-Class Classification Analysis")
print("=" * 120)
print(f"Dataset: {num_samples} samples, {num_features} features, {len(np.unique(y_class))} classes")
print(f"Class distribution: {y_class.value_counts().to_dict()}")
print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")
print(f"Target accuracy: {target_accuracy_score:.1f}")
print("=" * 120)

try:
    full_results_df, best_models_by_metric = runner.run_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train_class=y_train_class,
        y_test_class=y_test_class,
        y_train_reg=y_train_reg,
        y_test_reg=y_test_reg,
        preprocessing_config=preprocessing_config,
        metrics_to_evaluate=["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1Score", "ConfusionMatrix"]
    )
    print("\nPipeline execution complete")
except Exception as e:
    print(f"Pipeline execution failed: {str(e)}")
    raise

try:
    runner.generate_reports_and_visualizations(
        full_results_df=full_results_df,
        best_models_by_metric=best_models_by_metric
    )
    print("\nReports and visualizations generated")
    print("Check 'lono_results_confusion_matrix_example' directory for detailed outputs")
except Exception as e:
    print(f"Report generation failed: {str(e)}")
    raise

print("\n" + "=" * 120)
print("MODEL PERFORMANCE ANALYSIS - Confusion Matrix Focus")
print("=" * 120)

confusion_results = full_results_df[full_results_df['metric_name'] == 'ConfusionMatrix'].copy()
if not confusion_results.empty:
    print(f"\nFound {len(confusion_results)} confusion matrix evaluations across models")

    accuracy_results = full_results_df[full_results_df['metric_name'] == 'Accuracy']
    if not accuracy_results.empty:
        best_accuracy_model = accuracy_results.loc[accuracy_results['Testing Score'].idxmax()]

        print("BEST PERFORMING MODEL:")
        print(f"Model: {best_accuracy_model['Model']} ({best_accuracy_model['Type']})")
        print(f"Training Score: {best_accuracy_model['Training Score']:.4f}")
        print(f"Testing Score: {best_accuracy_model['Testing Score']:.4f}")
        print(f"Difference: {best_accuracy_model['Difference']:.4f}")

        best_model_confusion = confusion_results[
            (confusion_results['Model'] == best_accuracy_model['Model']) &
            (confusion_results['Type'] == best_accuracy_model['Type'])
        ]

        if not best_model_confusion.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle('Advanced Confusion Matrix Analysis', fontsize=16, fontweight='bold')

            unique_labels = np.unique(y_test_class)

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train_class)
            y_pred_test = rf_model.predict(X_test)

            cm = sklearn_confusion_matrix(y_test_class, y_pred_test, labels=unique_labels)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_labels, yticklabels=unique_labels, ax=ax1,
                       cbar_kws={'label': 'Count'})
            ax1.set_title('Confusion Matrix Heatmap', fontweight='bold')
            ax1.set_xlabel('Predicted Label')
            ax1.set_ylabel('True Label')

            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlBu_r',
                       xticklabels=unique_labels, yticklabels=unique_labels, ax=ax2,
                       cbar_kws={'label': 'Percentage'})
            ax2.set_title('Normalized Confusion Matrix (%)', fontweight='bold')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')

            precision_per_class = np.diag(cm) / np.sum(cm, axis=0)
            recall_per_class = np.diag(cm) / np.sum(cm, axis=1)
            f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)

            x = np.arange(len(unique_labels))
            width = 0.25
            ax3.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8, color='skyblue')
            ax3.bar(x, recall_per_class, width, label='Recall', alpha=0.8, color='lightcoral')
            ax3.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8, color='lightgreen')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Score')
            ax3.set_title('Per-Class Metrics', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(unique_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            errors = cm - np.eye(len(unique_labels)) * np.diag(cm)
            sns.heatmap(errors, annot=True, fmt='d', cmap='Reds',
                       xticklabels=unique_labels, yticklabels=unique_labels, ax=ax4,
                       cbar_kws={'label': 'Misclassification Count'})
            ax4.set_title('Misclassification Patterns', fontweight='bold')
            ax4.set_xlabel('Predicted Label')
            ax4.set_ylabel('True Label')

            plt.tight_layout()
            plt.savefig('lono_results_confusion_matrix_example/advanced_confusion_matrix_visualization.png',
                       dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\nConfusion Matrix Analysis for {best_accuracy_model['Model']}:")
            print("-" * 80)
            print("Raw Confusion Matrix:")
            print(cm)
            print(f"\nClass Labels: {unique_labels}")

            accuracy = np.trace(cm) / np.sum(cm)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)
            f1_macro = np.mean(f1_per_class)

            print("\nOverall Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro Precision: {precision_macro:.4f}")
            print(f"Macro Recall: {recall_macro:.4f}")
            print(f"Macro F1-Score: {f1_macro:.4f}")

            print("\nPer-Class Performance:")
            for i, label in enumerate(unique_labels):
                print(f"  {label}: Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

            max_errors = np.max(errors, axis=1)
            most_confused_idx = np.argmax(max_errors)
            print(f"\nMost confused class: {unique_labels[most_confused_idx]} "
                  f"(misclassified {max_errors[most_confused_idx]} times)")

else:
    print("\nNo confusion matrix results found in the evaluation results.")

print("\nCOMPLETE MODEL COMPARISON:")
print("=" * 120)
model_comparison = full_results_df[full_results_df['metric_name'].isin(['Accuracy', 'BalancedAccuracy', 'F1Score'])].copy()
if not model_comparison.empty:
    pivot_table = model_comparison.pivot_table(
        values='Testing Score',
        index=['Model', 'Type'],
        columns='metric_name',
        aggfunc='first'
    ).round(4)
    print(pivot_table.to_string())
else:
    print("No comparison data available.")

print("\nEXECUTION COMPLETE")
print("Outputs saved to: lono_results_confusion_matrix_example/")
print("Advanced visualization: advanced_confusion_matrix_visualization.png")
print("=" * 120)
