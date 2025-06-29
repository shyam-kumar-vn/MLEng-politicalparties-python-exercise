# Databricks notebook source
# MAGIC %md
# MAGIC # Inference Reporting and Alerting
# MAGIC 
# MAGIC This notebook generates comprehensive reports and alerts for the batch inference workflow.

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
from datetime import datetime
from src.utils import get_widget_value, get_table_name, print_parameters

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure parameters
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")

params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "DBFS Base Path": DBFS_BASE_PATH
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Inference Data

# COMMAND ----------

# DBTITLE 1,Load inference results
inference_results_table = get_table_name(CATALOG_NAME, SCHEMA_NAME, "inference_results")
inference_df = spark.read.table(inference_results_table)
inference_results = inference_df.toPandas()

print(f"Loaded {len(inference_results)} inference results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Executive Summary

# COMMAND ----------

# DBTITLE 1,Create executive summary
executive_summary = {
    "report_timestamp": datetime.now().isoformat(),
    "total_inference_samples": len(inference_results),
    "inference_methods_used": inference_results['inference_method'].unique().tolist()
}

# Add prediction distribution
if 'predicted_party' in inference_results.columns:
    prediction_dist = inference_results['predicted_party'].value_counts().to_dict()
    executive_summary["prediction_distribution"] = prediction_dist

# Add accuracy metrics if ground truth available
if 'Party' in inference_results.columns and 'predicted_party' in inference_results.columns:
    overall_accuracy = (inference_results['Party'] == inference_results['predicted_party']).mean()
    executive_summary["overall_accuracy"] = overall_accuracy

print("Executive summary created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Visualizations

# COMMAND ----------

# DBTITLE 1,Create dashboard
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Inference Reporting Dashboard', fontsize=16, fontweight='bold')

# Prediction distribution
if 'predicted_party' in inference_results.columns:
    prediction_counts = inference_results['predicted_party'].value_counts()
    axes[0].pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%')
    axes[0].set_title('Prediction Distribution')

# Accuracy by method
if 'Party' in inference_results.columns and 'predicted_party' in inference_results.columns:
    accuracy_by_method = {}
    for method in inference_results['inference_method'].unique():
        method_data = inference_results[inference_results['inference_method'] == method]
        accuracy = (method_data['Party'] == method_data['predicted_party']).mean()
        accuracy_by_method[method] = accuracy
    
    methods = list(accuracy_by_method.keys())
    accuracies = list(accuracy_by_method.values())
    
    bars = axes[1].bar(methods, accuracies, color=['skyblue', 'lightcoral'])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Method')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Save dashboard
dashboard_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_dashboard.png"
plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
print(f"Dashboard saved to: {dashboard_path}")

display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Final Report

# COMMAND ----------

# DBTITLE 1,Create final report
final_report = {
    "report_metadata": {
        "generated_at": datetime.now().isoformat(),
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "total_samples": len(inference_results)
    },
    "executive_summary": executive_summary,
    "files_generated": [
        dashboard_path,
        f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_report.json"
    ]
}

# Save final report
final_report_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_report.json"
with open(final_report_path, 'w') as f:
    json.dump(final_report, f, indent=2)

print(f"Final report saved to: {final_report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Summary

# COMMAND ----------

# DBTITLE 1,Show final summary
print("="*60)
print("INFERENCE REPORTING SUMMARY")
print("="*60)
print(f"Report Generated: {final_report['report_metadata']['generated_at']}")
print(f"Total Samples: {final_report['report_metadata']['total_samples']}")
print(f"Methods Used: {', '.join(executive_summary['inference_methods_used'])}")

if 'overall_accuracy' in executive_summary:
    print(f"Overall Accuracy: {executive_summary['overall_accuracy']:.4f}")

print(f"\nFiles Generated:")
for file_path in final_report['files_generated']:
    print(f"  - {file_path}")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Reporting Complete! 