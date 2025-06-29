# Databricks notebook source
# MAGIC %md
# MAGIC # Inference Method Comparison
# MAGIC 
# MAGIC This notebook compares the results from both inference methods:
# MAGIC 1. Serving Endpoint inference
# MAGIC 2. Direct Model inference
# MAGIC 
# MAGIC It analyzes performance, accuracy, and consistency between the two approaches.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any

from src.utils import get_widget_value, get_table_name, print_parameters

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure parameters
# Get parameters from workflow
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "DBFS Base Path": DBFS_BASE_PATH
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Inference Results

# COMMAND ----------

# DBTITLE 1,Load results from both inference methods
# Load serving endpoint results
serving_endpoint_table = get_table_name(CATALOG_NAME, SCHEMA_NAME, "inference_results")
serving_endpoint_df = spark.read.table(serving_endpoint_table)

# Filter for serving endpoint results
serving_endpoint_results = serving_endpoint_df.filter(
    serving_endpoint_df.inference_method == "serving_endpoint"
).toPandas()

print(f"Loaded {len(serving_endpoint_results)} serving endpoint results")

# Load direct model results
direct_model_results = serving_endpoint_df.filter(
    serving_endpoint_df.inference_method == "direct_model"
).toPandas()

print(f"Loaded {len(direct_model_results)} direct model results")

# COMMAND ----------

# DBTITLE 1,Load performance reports
# Load performance reports for both methods
serving_endpoint_report_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_performance_report_serving_endpoint.json"
direct_model_report_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_performance_report_direct_model.json"

serving_endpoint_performance = {}
direct_model_performance = {}

try:
    with open(serving_endpoint_report_path, 'r') as f:
        serving_endpoint_performance = json.load(f)
    print("Loaded serving endpoint performance report")
except FileNotFoundError:
    print("Serving endpoint performance report not found")

try:
    with open(direct_model_report_path, 'r') as f:
        direct_model_performance = json.load(f)
    print("Loaded direct model performance report")
except FileNotFoundError:
    print("Direct model performance report not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Comparison

# COMMAND ----------

# DBTITLE 1,Compare performance metrics
print("="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

if serving_endpoint_performance and direct_model_performance:
    serving_summary = serving_endpoint_performance.get('inference_summary', {})
    direct_summary = direct_model_performance.get('inference_summary', {})
    
    print(f"{'Metric':<25} {'Serving Endpoint':<20} {'Direct Model':<20} {'Difference':<15}")
    print("-" * 80)
    
    metrics = [
        ('Total Predictions', 'total_predictions'),
        ('Successful Predictions', 'successful_predictions'),
        ('Success Rate', 'success_rate'),
        ('Total Time (s)', 'total_time_seconds'),
        ('Avg Batch Time (s)', 'avg_batch_time_seconds'),
        ('Predictions/Second', 'predictions_per_second')
    ]
    
    for metric_name, metric_key in metrics:
        serving_val = serving_summary.get(metric_key, 0)
        direct_val = direct_summary.get(metric_key, 0)
        
        if metric_key == 'success_rate':
            serving_val = f"{serving_val:.2%}"
            direct_val = f"{direct_val:.2%}"
            diff = "N/A"
        elif metric_key in ['total_time_seconds', 'avg_batch_time_seconds']:
            diff = f"{serving_val - direct_val:+.2f}s"
        elif metric_key == 'predictions_per_second':
            diff = f"{serving_val - direct_val:+.2f}"
        else:
            diff = f"{serving_val - direct_val:+d}"
        
        print(f"{metric_name:<25} {serving_val:<20} {direct_val:<20} {diff:<15}")
    
    print("="*60)
else:
    print("Performance reports not available for comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction Consistency Analysis

# COMMAND ----------

# DBTITLE 1,Compare predictions between methods
if len(serving_endpoint_results) > 0 and len(direct_model_results) > 0:
    # Merge results on common columns for comparison
    comparison_df = serving_endpoint_results[['Tweet', 'predicted_party']].copy()
    comparison_df = comparison_df.rename(columns={'predicted_party': 'serving_endpoint_prediction'})
    
    # Add direct model predictions
    direct_predictions = direct_model_results[['Tweet', 'predicted_party']].copy()
    direct_predictions = direct_predictions.rename(columns={'predicted_party': 'direct_model_prediction'})
    
    # Merge on Tweet text
    comparison_df = comparison_df.merge(direct_predictions, on='Tweet', how='inner')
    
    print(f"Comparing {len(comparison_df)} predictions")
    
    # Calculate consistency
    consistent_predictions = (comparison_df['serving_endpoint_prediction'] == comparison_df['direct_model_prediction']).sum()
    consistency_rate = consistent_predictions / len(comparison_df)
    
    print(f"Consistent predictions: {consistent_predictions}/{len(comparison_df)} ({consistency_rate:.2%})")
    
    # Show disagreements
    disagreements = comparison_df[comparison_df['serving_endpoint_prediction'] != comparison_df['direct_model_prediction']]
    if len(disagreements) > 0:
        print(f"\nFound {len(disagreements)} disagreements:")
        display(disagreements.head(10))
    else:
        print("\nAll predictions are consistent between methods!")
        
else:
    print("Not enough data for prediction comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accuracy Comparison (if ground truth available)

# COMMAND ----------

# DBTITLE 1,Compare accuracy with ground truth
if len(serving_endpoint_results) > 0 and 'Party' in serving_endpoint_results.columns:
    print("="*60)
    print("ACCURACY COMPARISON WITH GROUND TRUTH")
    print("="*60)
    
    # Serving endpoint accuracy
    serving_accuracy = (serving_endpoint_results['Party'] == serving_endpoint_results['predicted_party']).mean()
    print(f"Serving Endpoint Accuracy: {serving_accuracy:.4f}")
    
    # Direct model accuracy
    if len(direct_model_results) > 0:
        direct_accuracy = (direct_model_results['Party'] == direct_model_results['predicted_party']).mean()
        print(f"Direct Model Accuracy: {direct_accuracy:.4f}")
        
        accuracy_diff = serving_accuracy - direct_accuracy
        print(f"Accuracy Difference: {accuracy_diff:+.4f}")
        
        if abs(accuracy_diff) < 0.001:
            print("✅ Accuracies are essentially identical")
        elif accuracy_diff > 0:
            print("✅ Serving endpoint has slightly better accuracy")
        else:
            print("✅ Direct model has slightly better accuracy")
    
    print("="*60)
else:
    print("Ground truth not available for accuracy comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization

# COMMAND ----------

# DBTITLE 1,Create comparison visualizations
# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Inference Method Comparison', fontsize=16, fontweight='bold')

# 1. Performance comparison
if serving_endpoint_performance and direct_model_performance:
    serving_summary = serving_endpoint_performance.get('inference_summary', {})
    direct_summary = direct_model_performance.get('inference_summary', {})
    
    metrics = ['total_time_seconds', 'avg_batch_time_seconds', 'predictions_per_second']
    metric_names = ['Total Time (s)', 'Avg Batch Time (s)', 'Predictions/Second']
    
    serving_values = [serving_summary.get(m, 0) for m in metrics]
    direct_values = [direct_summary.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, serving_values, width, label='Serving Endpoint', color='skyblue')
    axes[0, 0].bar(x + width/2, direct_values, width, label='Direct Model', color='lightcoral')
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

# 2. Prediction distribution comparison
if len(serving_endpoint_results) > 0 and len(direct_model_results) > 0:
    serving_dist = serving_endpoint_results['predicted_party'].value_counts()
    direct_dist = direct_model_results['predicted_party'].value_counts()
    
    # Get all unique parties
    all_parties = sorted(set(serving_dist.index) | set(direct_dist.index))
    
    serving_counts = [serving_dist.get(party, 0) for party in all_parties]
    direct_counts = [direct_dist.get(party, 0) for party in all_parties]
    
    x = np.arange(len(all_parties))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, serving_counts, width, label='Serving Endpoint', color='skyblue')
    axes[0, 1].bar(x + width/2, direct_counts, width, label='Direct Model', color='lightcoral')
    axes[0, 1].set_xlabel('Political Party')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Prediction Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_parties, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# 3. Success rate comparison
if serving_endpoint_performance and direct_model_performance:
    serving_success = serving_endpoint_performance.get('inference_summary', {}).get('success_rate', 0)
    direct_success = direct_model_performance.get('inference_summary', {}).get('success_rate', 0)
    
    methods = ['Serving Endpoint', 'Direct Model']
    success_rates = [serving_success, direct_success]
    colors = ['skyblue', 'lightcoral']
    
    bars = axes[1, 0].bar(methods, success_rates, color=colors)
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_title('Inference Success Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.2%}', ha='center', va='bottom')

# 4. Inference times distribution
if serving_endpoint_performance and direct_model_performance:
    serving_times = serving_endpoint_performance.get('inference_times', [])
    direct_times = direct_model_performance.get('inference_times', [])
    
    if serving_times and direct_times:
        axes[1, 1].hist(serving_times, alpha=0.7, label='Serving Endpoint', color='skyblue', bins=20)
        axes[1, 1].hist(direct_times, alpha=0.7, label='Direct Model', color='lightcoral', bins=20)
        axes[1, 1].set_xlabel('Inference Time (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Inference Time Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save the comparison plot
comparison_plot_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_comparison.png"
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {comparison_plot_path}")

display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Comparison Report

# COMMAND ----------

# DBTITLE 1,Create comprehensive comparison report
comparison_report = {
    "comparison_summary": {
        "timestamp": datetime.now().isoformat(),
        "serving_endpoint_samples": len(serving_endpoint_results),
        "direct_model_samples": len(direct_model_results),
        "comparison_samples": len(comparison_df) if 'comparison_df' in locals() else 0
    },
    "performance_comparison": {
        "serving_endpoint": serving_endpoint_performance.get('inference_summary', {}),
        "direct_model": direct_model_performance.get('inference_summary', {})
    },
    "prediction_consistency": {
        "consistent_predictions": consistent_predictions if 'consistent_predictions' in locals() else 0,
        "total_comparisons": len(comparison_df) if 'comparison_df' in locals() else 0,
        "consistency_rate": consistency_rate if 'consistency_rate' in locals() else 0
    },
    "accuracy_comparison": {
        "serving_endpoint_accuracy": serving_accuracy if 'serving_accuracy' in locals() else None,
        "direct_model_accuracy": direct_accuracy if 'direct_accuracy' in locals() else None,
        "accuracy_difference": accuracy_diff if 'accuracy_diff' in locals() else None
    },
    "recommendations": []
}

# Generate recommendations
if 'serving_accuracy' in locals() and 'direct_accuracy' in locals():
    if abs(accuracy_diff) < 0.001:
        comparison_report["recommendations"].append("Both methods provide similar accuracy - choose based on operational requirements")
    elif accuracy_diff > 0:
        comparison_report["recommendations"].append("Serving endpoint provides slightly better accuracy")
    else:
        comparison_report["recommendations"].append("Direct model provides slightly better accuracy")

if 'consistency_rate' in locals():
    if consistency_rate > 0.99:
        comparison_report["recommendations"].append("Excellent consistency between methods - both are reliable")
    elif consistency_rate > 0.95:
        comparison_report["recommendations"].append("Good consistency between methods - minor differences may be due to implementation details")
    else:
        comparison_report["recommendations"].append("Significant differences between methods - investigate implementation differences")

if serving_endpoint_performance and direct_model_performance:
    serving_time = serving_endpoint_performance.get('inference_summary', {}).get('total_time_seconds', 0)
    direct_time = direct_model_performance.get('inference_summary', {}).get('total_time_seconds', 0)
    
    if serving_time < direct_time:
        comparison_report["recommendations"].append("Serving endpoint is faster - better for real-time applications")
    else:
        comparison_report["recommendations"].append("Direct model is faster - better for batch processing")

# Save comparison report
comparison_report_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_comparison_report.json"
with open(comparison_report_path, 'w') as f:
    json.dump(comparison_report, f, indent=2)

print(f"Comparison report saved to: {comparison_report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Comparison Summary

# COMMAND ----------

# DBTITLE 1,Show comparison summary
print("="*80)
print("INFERENCE METHOD COMPARISON SUMMARY")
print("="*80)

print(f"Comparison Timestamp: {comparison_report['comparison_summary']['timestamp']}")
print(f"Serving Endpoint Samples: {comparison_report['comparison_summary']['serving_endpoint_samples']}")
print(f"Direct Model Samples: {comparison_report['comparison_summary']['direct_model_samples']}")
print(f"Comparison Samples: {comparison_report['comparison_summary']['comparison_samples']}")

if 'consistency_rate' in locals():
    print(f"\nPrediction Consistency: {consistency_rate:.2%}")

if 'serving_accuracy' in locals() and 'direct_accuracy' in locals():
    print(f"\nAccuracy Comparison:")
    print(f"  Serving Endpoint: {serving_accuracy:.4f}")
    print(f"  Direct Model: {direct_accuracy:.4f}")
    print(f"  Difference: {accuracy_diff:+.4f}")

print(f"\nRecommendations:")
for i, rec in enumerate(comparison_report['recommendations'], 1):
    print(f"  {i}. {rec}")

print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Comparison Complete!
# MAGIC 
# MAGIC The comparison analysis has successfully:
# MAGIC - Compared performance metrics between both inference methods
# MAGIC - Analyzed prediction consistency
# MAGIC - Evaluated accuracy differences (if ground truth available)
# MAGIC - Generated visualizations and reports
# MAGIC - Provided recommendations for method selection
# MAGIC 
# MAGIC The comparison results are now available for decision-making and optimization. 