# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference Workflow
# MAGIC 
# MAGIC This notebook implements batch inference for the Political Party Classification model (XGBoost):
# MAGIC 1. Load data for inference (same data used for training)
# MAGIC 2. Choose inference method: Serving Endpoint OR Direct Model
# MAGIC 3. Perform batch predictions
# MAGIC 4. Save results to Delta table
# MAGIC 5. Generate inference reports

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries and components
import os
import json
import time
import requests
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from src.utils import get_widget_value, get_model_uri, print_parameters, get_table_name

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure MLflow and Unity Catalog
# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Get parameters from workflow, with fallback to default values
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
MODEL_NAME = get_widget_value("model_name", "political_party_classifier")
FEATURES_TABLE = get_widget_value("features_table", "tweet_features")
INFERENCE_METHOD = get_widget_value("inference_method", "serving_endpoint")  # "serving_endpoint" or "direct_model"
ENDPOINT_NAME = get_widget_value("endpoint_name", "political-party-classifier-endpoint")
PRODUCTION_ALIAS = get_widget_value("production_alias", "production")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")
BATCH_SIZE = int(get_widget_value("batch_size", "100"))

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "Model": MODEL_NAME,
    "Features Table": FEATURES_TABLE,
    "Inference Method": INFERENCE_METHOD,
    "Endpoint Name": ENDPOINT_NAME,
    "Production Alias": PRODUCTION_ALIAS,
    "DBFS Base Path": DBFS_BASE_PATH,
    "Batch Size": BATCH_SIZE
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data for Inference

# COMMAND ----------

# DBTITLE 1,Load features data from Delta table
# Load data from Delta table (same data used for training)
features_table_name = get_table_name(CATALOG_NAME, SCHEMA_NAME, FEATURES_TABLE)
df = spark.read.table(features_table_name)

# Convert to pandas
inference_data = df.toPandas()

print(f"Loaded {len(inference_data)} samples from {features_table_name}")
print(f"Data columns: {list(inference_data.columns)}")

# COMMAND ----------

# DBTITLE 1,Prepare data for inference
# For inference, we'll use all data (not just test set)
# We'll focus on the original tweets and features
inference_texts = inference_data['Tweet'].tolist()
inference_features = inference_data[[col for col in inference_data.columns if col.startswith('feature_')]].values

print(f"Prepared {len(inference_texts)} texts for inference")
print(f"Feature matrix shape: {inference_features.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Method Selection

# COMMAND ----------

# DBTITLE 1,Define inference methods
class BatchInferenceEngine:
    """Engine for performing batch inference using different methods"""
    
    def __init__(self, method: str, config: Dict[str, Any]):
        self.method = method
        self.config = config
        self.predictions = []
        self.inference_times = []
        
    def infer_serving_endpoint(self, texts: List[str]) -> List[str]:
        """Perform inference using Databricks serving endpoint"""
        logger.info(f"Using serving endpoint: {self.config['endpoint_name']}")
        
        # Get endpoint URL
        workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().get("api_url")
        endpoint_url = f"{workspace_url}/serving-endpoints/{self.config['endpoint_name']}/invocations"
        
        # Get authentication token
        token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        predictions = []
        batch_size = self.config.get('batch_size', 100)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            data = {
                "dataframe_records": [
                    {"text": text} for text in batch_texts
                ]
            }
            
            start_time = time.time()
            response = requests.post(endpoint_url, headers=headers, json=data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                batch_predictions = result.get('predictions', [])
                predictions.extend(batch_predictions)
                self.inference_times.append(end_time - start_time)
                logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_predictions)} predictions")
            else:
                logger.error(f"Endpoint request failed: {response.status_code} - {response.text}")
                # Fallback to empty predictions
                predictions.extend([None] * len(batch_texts))
        
        return predictions
    
    def infer_direct_model(self, texts: List[str]) -> List[str]:
        """Perform inference using registered model directly"""
        logger.info("Using registered model directly")
        
        # Load model from Unity Catalog
        model_uri = get_model_uri(self.config['catalog_name'], self.config['schema_name'], 
                                 self.config['model_name'], self.config['production_alias'])
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded from: {model_uri}")
        
        predictions = []
        batch_size = self.config.get('batch_size', 100)
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Prepare input data
            input_df = pd.DataFrame({'text': batch_texts})
            
            start_time = time.time()
            batch_predictions = model.predict(input_df)
            end_time = time.time()
            
            # Convert to list if needed
            if hasattr(batch_predictions, 'tolist'):
                batch_predictions = batch_predictions.tolist()
            elif not isinstance(batch_predictions, list):
                batch_predictions = list(batch_predictions)
            
            predictions.extend(batch_predictions)
            self.inference_times.append(end_time - start_time)
            logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_predictions)} predictions")
        
        return predictions
    
    def run_inference(self, texts: List[str]) -> Dict[str, Any]:
        """Run inference using the selected method"""
        logger.info(f"Starting batch inference using method: {self.method}")
        
        start_time = time.time()
        
        if self.method == "serving_endpoint":
            predictions = self.infer_serving_endpoint(texts)
        elif self.method == "direct_model":
            predictions = self.infer_direct_model(texts)
        else:
            raise ValueError(f"Unknown inference method: {self.method}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_batch_time = np.mean(self.inference_times) if self.inference_times else 0
        total_predictions = len(predictions)
        successful_predictions = sum(1 for p in predictions if p is not None)
        
        results = {
            "predictions": predictions,
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "success_rate": successful_predictions / total_predictions if total_predictions > 0 else 0,
            "inference_times": self.inference_times
        }
        
        logger.info(f"Inference completed: {successful_predictions}/{total_predictions} successful predictions")
        logger.info(f"Total time: {total_time:.2f}s, Avg batch time: {avg_batch_time:.2f}s")
        
        return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform Batch Inference

# COMMAND ----------

# DBTITLE 1,Initialize inference engine
# Create configuration for inference engine
inference_config = {
    "catalog_name": CATALOG_NAME,
    "schema_name": SCHEMA_NAME,
    "model_name": MODEL_NAME,
    "production_alias": PRODUCTION_ALIAS,
    "endpoint_name": ENDPOINT_NAME,
    "batch_size": BATCH_SIZE
}

# Initialize inference engine
inference_engine = BatchInferenceEngine(INFERENCE_METHOD, inference_config)

print(f"Inference engine initialized with method: {INFERENCE_METHOD}")

# COMMAND ----------

# DBTITLE 1,Run batch inference
# Perform batch inference
inference_results = inference_engine.run_inference(inference_texts)

print("="*60)
print("BATCH INFERENCE RESULTS")
print("="*60)
print(f"Method: {INFERENCE_METHOD}")
print(f"Total predictions: {inference_results['total_predictions']}")
print(f"Successful predictions: {inference_results['successful_predictions']}")
print(f"Success rate: {inference_results['success_rate']:.2%}")
print(f"Total time: {inference_results['total_time']:.2f} seconds")
print(f"Average batch time: {inference_results['avg_batch_time']:.2f} seconds")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Inference Results

# COMMAND ----------

# DBTITLE 1,Create results DataFrame
# Create results DataFrame
results_df = inference_data.copy()
results_df['predicted_party'] = inference_results['predictions']
results_df['inference_timestamp'] = datetime.now().isoformat()
results_df['inference_method'] = INFERENCE_METHOD
results_df['inference_success'] = results_df['predicted_party'].notna()

# Add inference metadata
results_df['total_inference_time'] = inference_results['total_time']
results_df['avg_batch_time'] = inference_results['avg_batch_time']
results_df['success_rate'] = inference_results['success_rate']

print("Results DataFrame created:")
print(f"Shape: {results_df.shape}")
print(f"Columns: {list(results_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Save results to Delta table
# Save inference results to Delta table
inference_results_table = get_table_name(CATALOG_NAME, SCHEMA_NAME, "inference_results")
spark_results_df = spark.createDataFrame(results_df)

# Save with timestamp to avoid overwriting
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
inference_results_table_with_timestamp = f"{inference_results_table}_{timestamp}"

spark_results_df.write.mode("overwrite").saveAsTable(inference_results_table_with_timestamp)

print(f"Inference results saved to: {inference_results_table_with_timestamp}")

# Also save to a fixed table name for easy access
spark_results_df.write.mode("overwrite").saveAsTable(inference_results_table)

print(f"Inference results also saved to: {inference_results_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Inference Reports

# COMMAND ----------

# DBTITLE 1,Calculate inference statistics
# Calculate prediction distribution
prediction_distribution = results_df['predicted_party'].value_counts()
print("Prediction Distribution:")
print(prediction_distribution)

# Calculate accuracy if we have ground truth
if 'Party' in results_df.columns:
    accuracy = (results_df['Party'] == results_df['predicted_party']).mean()
    print(f"\nAccuracy (if ground truth available): {accuracy:.4f}")

# Performance statistics
print(f"\nPerformance Statistics:")
print(f"Total inference time: {inference_results['total_time']:.2f} seconds")
print(f"Average batch time: {inference_results['avg_batch_time']:.2f} seconds")
print(f"Predictions per second: {inference_results['total_predictions'] / inference_results['total_time']:.2f}")
print(f"Success rate: {inference_results['success_rate']:.2%}")

# COMMAND ----------

# DBTITLE 1,Create detailed performance report
# Create detailed performance report
performance_report = {
    "inference_summary": {
        "method": INFERENCE_METHOD,
        "total_predictions": inference_results['total_predictions'],
        "successful_predictions": inference_results['successful_predictions'],
        "success_rate": inference_results['success_rate'],
        "total_time_seconds": inference_results['total_time'],
        "avg_batch_time_seconds": inference_results['avg_batch_time'],
        "predictions_per_second": inference_results['total_predictions'] / inference_results['total_time'],
        "batch_size": BATCH_SIZE
    },
    "prediction_distribution": prediction_distribution.to_dict(),
    "inference_times": inference_results['inference_times'],
    "configuration": inference_config,
    "timestamp": datetime.now().isoformat()
}

# Save performance report
performance_report_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_performance_report.json"
with open(performance_report_path, 'w') as f:
    json.dump(performance_report, f, indent=2)

print(f"Performance report saved to: {performance_report_path}")

# COMMAND ----------

# DBTITLE 1,Display sample predictions
# Display sample predictions
print("Sample Predictions:")
sample_results = results_df[['Tweet', 'Party', 'predicted_party', 'inference_success']].head(10)
display(sample_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparison with Ground Truth (if available)

# COMMAND ----------

# DBTITLE 1,Compare predictions with ground truth
if 'Party' in results_df.columns:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter for successful predictions
    valid_results = results_df[results_df['inference_success']]
    
    if len(valid_results) > 0:
        # Generate classification report
        y_true = valid_results['Party']
        y_pred = valid_results['predicted_party']
        
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=y_true.unique(), yticklabels=y_true.unique())
        plt.title('Confusion Matrix - Batch Inference Results')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_confusion_matrix.png"
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        
        display(plt.gcf())
        plt.close()
    else:
        print("No successful predictions to compare with ground truth")
else:
    print("No ground truth available for comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference Summary

# COMMAND ----------

# DBTITLE 1,Display final summary
print("="*80)
print("BATCH INFERENCE WORKFLOW SUMMARY")
print("="*80)
print(f"Inference Method: {INFERENCE_METHOD}")
print(f"Model: {CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}")
print(f"Data Source: {features_table_name}")
print(f"Total Samples: {len(inference_data)}")
print(f"Batch Size: {BATCH_SIZE}")
print()
print("RESULTS:")
print(f"  Total Predictions: {inference_results['total_predictions']}")
print(f"  Successful Predictions: {inference_results['successful_predictions']}")
print(f"  Success Rate: {inference_results['success_rate']:.2%}")
print(f"  Total Time: {inference_results['total_time']:.2f} seconds")
print(f"  Average Batch Time: {inference_results['avg_batch_time']:.2f} seconds")
print(f"  Predictions/Second: {inference_results['total_predictions'] / inference_results['total_time']:.2f}")
print()
print("OUTPUTS:")
print(f"  Results Table: {inference_results_table}")
print(f"  Timestamped Results: {inference_results_table_with_timestamp}")
print(f"  Performance Report: {performance_report_path}")
if 'Party' in results_df.columns:
    print(f"  Confusion Matrix: {DBFS_BASE_PATH}/{SCHEMA_NAME}_inference_confusion_matrix.png")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference Complete!
# MAGIC 
# MAGIC The batch inference workflow has successfully:
# MAGIC - Loaded data from the features table
# MAGIC - Performed batch inference using the selected method
# MAGIC - Saved results to Delta tables
# MAGIC - Generated performance reports
# MAGIC - Compared with ground truth (if available)
# MAGIC 
# MAGIC The results are now available for further analysis and integration. 