# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC 
# MAGIC This notebook evaluates the trained model and generates performance reports:
# MAGIC 1. Load the trained model from Unity Catalog
# MAGIC 2. Evaluate on test data
# MAGIC 3. Generate performance metrics and visualizations
# MAGIC 4. Create evaluation report

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Get parameters
# Get parameters from workflow, with fallback to default values
CATALOG_NAME = dbutils.widgets.get("catalog_name") if dbutils.widgets.get("catalog_name") else "mle_batch_catalog_2025_q2"
SCHEMA_NAME = dbutils.widgets.get("schema_name") if dbutils.widgets.get("schema_name") else "mle_shyamkumar_vn"
MODEL_NAME = dbutils.widgets.get("model_name") if dbutils.widgets.get("model_name") else "political_party_classifier"
DBFS_BASE_PATH = dbutils.widgets.get("dbfs_base_path") if dbutils.widgets.get("dbfs_base_path") else "/dbfs/FileStore/shyamkumar.vn"

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Model: {MODEL_NAME}")
print(f"DBFS Base Path: {DBFS_BASE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model and Data

# COMMAND ----------

# DBTITLE 1,Load the trained model
import mlflow
import mlflow.pyfunc

# Configure MLflow
mlflow.set_registry_uri("databricks-uc")

# Load the latest version of the model
model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}/latest"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"Model loaded from: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Load test data
# Load features from Delta table
features_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.tweet_features"
df = spark.read.table(features_table)

# Convert to pandas for evaluation
test_data = df.toPandas()

print(f"Loaded {len(test_data)} samples for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation

# COMMAND ----------

# DBTITLE 1,Prepare test data
# Extract features and labels
feature_columns = [col for col in test_data.columns if col.startswith('feature_')]
X_test = test_data[feature_columns].values
y_true = test_data['Party'].values

print(f"Test features shape: {X_test.shape}")
print(f"Test labels shape: {y_true.shape}")

# COMMAND ----------

# DBTITLE 1,Make predictions
# Make predictions using the loaded model
# Note: The model expects raw text, so we need to use the original tweets
predictions = loaded_model.predict(test_data[['Tweet']])

print(f"Generated {len(predictions)} predictions")

# COMMAND ----------

# DBTITLE 1,Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Calculate metrics
accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)

print("=== MODEL EVALUATION METRICS ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# COMMAND ----------

# DBTITLE 1,Detailed classification report
print("\n=== DETAILED CLASSIFICATION REPORT ===")
print(classification_report(y_true, predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Visualizations

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate confusion matrix
cm = confusion_matrix(y_true, predictions)

# Create confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_true), 
            yticklabels=np.unique(y_true))
plt.title('Confusion Matrix - Political Party Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save plot
confusion_matrix_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Confusion matrix saved to: {confusion_matrix_path}")

# COMMAND ----------

# DBTITLE 1,Class-wise performance
# Calculate per-class metrics
from sklearn.metrics import precision_recall_fscore_support

precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_true, predictions, average=None, labels=np.unique(y_true)
)

# Create performance comparison
class_performance = pd.DataFrame({
    'Class': np.unique(y_true),
    'Precision': precision_per_class,
    'Recall': recall_per_class,
    'F1-Score': f1_per_class,
    'Support': support_per_class
})

print("=== PER-CLASS PERFORMANCE ===")
print(class_performance)

# COMMAND ----------

# DBTITLE 1,Performance visualization
# Create bar plot of per-class F1 scores
plt.figure(figsize=(10, 6))
bars = plt.bar(class_performance['Class'], class_performance['F1-Score'], 
               color=['skyblue', 'lightcoral'])
plt.title('F1-Score by Political Party')
plt.ylabel('F1-Score')
plt.xlabel('Political Party')
plt.ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, class_performance['F1-Score']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()

# Save plot
f1_scores_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_f1_scores.png"
plt.savefig(f1_scores_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"F1 scores plot saved to: {f1_scores_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Error Analysis

# COMMAND ----------

# DBTITLE 1,Analyze misclassifications
# Find misclassified samples
misclassified_mask = y_true != predictions
misclassified_data = test_data[misclassified_mask].copy()
misclassified_data['predicted_party'] = predictions[misclassified_mask]

print(f"Total misclassifications: {len(misclassified_data)}")
print(f"Error rate: {len(misclassified_data) / len(test_data):.2%}")

# Show some misclassified examples
print("\n=== SAMPLE MISCLASSIFICATIONS ===")
sample_misclassified = misclassified_data[['Tweet', 'Party', 'predicted_party']].head(10)
for idx, row in sample_misclassified.iterrows():
    print(f"Tweet: {row['Tweet'][:100]}...")
    print(f"True: {row['Party']}, Predicted: {row['predicted_party']}")
    print("-" * 80)

# COMMAND ----------

# DBTITLE 1,Save evaluation results
# Create evaluation summary
evaluation_summary = {
    'model_name': MODEL_NAME,
    'catalog': CATALOG_NAME,
    'schema': SCHEMA_NAME,
    'evaluation_date': pd.Timestamp.now().isoformat(),
    'total_samples': len(test_data),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'error_rate': len(misclassified_data) / len(test_data),
    'confusion_matrix_path': confusion_matrix_path,
    'f1_scores_path': f1_scores_path
}

# Save evaluation summary
import json
evaluation_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_evaluation_summary.json"
with open(evaluation_path, 'w') as f:
    json.dump(evaluation_summary, f, indent=2)

print(f"Evaluation summary saved to: {evaluation_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Summary

# COMMAND ----------

# DBTITLE 1,Display final summary
print("=" * 60)
print("MODEL EVALUATION SUMMARY")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Catalog: {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"Evaluation Date: {evaluation_summary['evaluation_date']}")
print(f"Total Test Samples: {evaluation_summary['total_samples']}")
print()
print("PERFORMANCE METRICS:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Error Rate: {evaluation_summary['error_rate']:.2%}")
print()
print("ARTIFACTS GENERATED:")
print(f"  Confusion Matrix: {confusion_matrix_path}")
print(f"  F1 Scores Plot: {f1_scores_path}")
print(f"  Evaluation Summary: {evaluation_path}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Evaluation Complete!
# MAGIC 
# MAGIC Successfully completed:
# MAGIC - Model loaded from Unity Catalog
# MAGIC - Comprehensive evaluation on test data
# MAGIC - Performance metrics calculated
# MAGIC - Visualizations generated
# MAGIC - Error analysis performed
# MAGIC - Evaluation report saved 