# Databricks notebook source
# MAGIC %md
# MAGIC # Model Promotion and A/B Testing
# MAGIC 
# MAGIC This notebook implements model promotion logic:
# MAGIC 1. Load the newly trained model and its validation metrics
# MAGIC 2. Load the current production model and evaluate it on validation set
# MAGIC 3. Compare performance and promote the better model
# MAGIC 4. Set up production aliases and deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries and components
import os
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import json
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

from src.utils import get_widget_value, get_model_uri, print_parameters, get_table_name

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure MLflow and Unity Catalog
# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set the catalog and schema for model registration
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
MODEL_NAME = get_widget_value("model_name", "political_party_classifier")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")
PRODUCTION_ALIAS = get_widget_value("production_alias", "production")
STAGING_ALIAS = get_widget_value("staging_alias", "staging")

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "Model": MODEL_NAME,
    "Production Alias": PRODUCTION_ALIAS,
    "Staging Alias": STAGING_ALIAS,
    "DBFS Base Path": DBFS_BASE_PATH
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Validation Data and New Model Metrics

# COMMAND ----------

# DBTITLE 1,Load validation data
# Load features from Delta table
features_table = get_table_name(CATALOG_NAME, SCHEMA_NAME, "tweet_features")
df = spark.read.table(features_table)

# Filter for validation data only
validation_df = df.filter(df["split"] == "validation")
validation_data = validation_df.toPandas()

print(f"Loaded {len(validation_data)} validation samples")

# COMMAND ----------

# DBTITLE 1,Load new model validation metrics
# Load validation metrics from the training run
validation_metrics_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_validation_metrics.json"

try:
    with open(validation_metrics_path, 'r') as f:
        new_model_metrics = json.load(f)
    
    print("New model validation metrics:")
    for metric, value in new_model_metrics.items():
        print(f"  {metric}: {value:.4f}")
        
except FileNotFoundError:
    print(f"Warning: Validation metrics file not found at {validation_metrics_path}")
    print("This might indicate the training run didn't complete successfully")
    new_model_metrics = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Current Production Model

# COMMAND ----------

# DBTITLE 1,Get current production model
client = MlflowClient()
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

try:
    # Check if production alias exists
    production_version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    current_production_version = production_version.version
    print(f"Current production model version: {current_production_version}")
    
except Exception as e:
    print(f"No production alias found: {e}")
    print("This might be the first model deployment")
    current_production_version = None

# COMMAND ----------

# DBTITLE 1,Evaluate current production model on validation set
if current_production_version is not None:
    # Load current production model
    current_model_uri = get_model_uri(CATALOG_NAME, SCHEMA_NAME, MODEL_NAME, str(current_production_version))
    current_model = mlflow.pyfunc.load_model(current_model_uri)
    
    print(f"Loaded current production model from: {current_model_uri}")
    
    # Prepare validation data for prediction
    feature_columns = [col for col in validation_data.columns if col.startswith('feature_')]
    X_val = validation_data[feature_columns].values
    y_val = validation_data['party_encoded'].values
    
    # Make predictions using current production model
    # Note: The model expects raw text, so we need to use the original tweets
    # Rename 'Tweet' column to 'text' to match model signature
    prediction_data = validation_data[['Tweet']].copy()
    prediction_data.columns = ['text']
    current_predictions_raw = current_model.predict(prediction_data)
    
    # Load the label encoder used during training
    encoder_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_label_encoder.pkl"
    
    try:
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Convert string predictions back to encoded integers
        current_predictions = label_encoder.transform(current_predictions_raw)
        
        print(f"Raw predictions: {current_predictions_raw[:5]}")  # Show first 5 predictions
        print(f"Encoded predictions: {current_predictions[:5]}")  # Show first 5 encoded predictions
        print(f"True labels: {y_val[:5]}")  # Show first 5 true labels
        
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        print("Using raw predictions for comparison (may cause label mismatch)")
        current_predictions = current_predictions_raw
    
    # Calculate metrics for current production model
    current_accuracy = accuracy_score(y_val, current_predictions)
    current_precision = precision_score(y_val, current_predictions, average='weighted', zero_division=0)
    current_recall = recall_score(y_val, current_predictions, average='weighted', zero_division=0)
    current_f1 = f1_score(y_val, current_predictions, average='weighted', zero_division=0)
    
    current_model_metrics = {
        'accuracy': current_accuracy,
        'precision': current_precision,
        'recall': current_recall,
        'f1_score': current_f1
    }
    
    print("Current production model validation metrics:")
    for metric, value in current_model_metrics.items():
        print(f"  {metric}: {value:.4f}")
        
else:
    print("No current production model found")
    current_model_metrics = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Comparison and Promotion Decision

# COMMAND ----------

# DBTITLE 1,Compare models and make promotion decision
if new_model_metrics is not None and current_model_metrics is not None:
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # Compare F1 score (primary metric)
    new_f1 = new_model_metrics['f1_score']
    current_f1 = current_model_metrics['f1_score']
    
    print(f"New Model F1 Score:     {new_f1:.4f}")
    print(f"Current Model F1 Score: {current_f1:.4f}")
    print(f"Improvement:            {new_f1 - current_f1:+.4f}")
    
    # Define promotion threshold (e.g., 0.01 improvement)
    PROMOTION_THRESHOLD = 0.01
    
    if new_f1 > current_f1 + PROMOTION_THRESHOLD:
        print(f"\n✅ NEW MODEL PROMOTED TO PRODUCTION")
        print(f"F1 score improvement ({new_f1 - current_f1:.4f}) exceeds threshold ({PROMOTION_THRESHOLD})")
        should_promote = True
    else:
        print(f"\n❌ NEW MODEL NOT PROMOTED")
        print(f"F1 score improvement ({new_f1 - current_f1:.4f}) below threshold ({PROMOTION_THRESHOLD})")
        should_promote = False
        
elif new_model_metrics is not None and current_model_metrics is None:
    print("=" * 60)
    print("FIRST MODEL DEPLOYMENT")
    print("=" * 60)
    print("No current production model found. Promoting new model to production.")
    should_promote = True
    
else:
    print("=" * 60)
    print("ERROR: Cannot make promotion decision")
    print("=" * 60)
    print("Missing validation metrics or model loading failed")
    should_promote = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Promotion and Alias Management

# COMMAND ----------

# DBTITLE 1,Promote model if decision is positive
if should_promote:
    try:
        # Get the latest model version (the newly trained one)
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max(model_versions, key=lambda x: x.version)
        new_version_number = latest_version.version
        
        print(f"\nPromoting model version {new_version_number} to production...")
        
        # Set production alias to new model
        client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, new_version_number)
        
        # If there was a previous production model, move it to staging
        if current_production_version is not None:
            client.set_registered_model_alias(model_name, STAGING_ALIAS, current_production_version)
            print(f"Moved previous production model (v{current_production_version}) to staging")
        
        print(f"✅ Successfully promoted model version {new_version_number} to production")
        print(f"Production alias '{PRODUCTION_ALIAS}' now points to version {new_version_number}")
        
        # Log promotion event
        promotion_log = {
            'promotion_timestamp': pd.Timestamp.now().isoformat(),
            'new_model_version': new_version_number,
            'previous_model_version': current_production_version,
            'new_model_f1_score': new_model_metrics['f1_score'] if new_model_metrics else None,
            'previous_model_f1_score': current_model_metrics['f1_score'] if current_model_metrics else None,
            'improvement': new_model_metrics['f1_score'] - current_model_metrics['f1_score'] if (new_model_metrics and current_model_metrics) else None,
            'promotion_threshold': PROMOTION_THRESHOLD
        }
        
        # Save promotion log
        promotion_log_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_promotion_log.json"
        with open(promotion_log_path, 'w') as f:
            json.dump(promotion_log, f, indent=2)
        
        print(f"Promotion log saved to: {promotion_log_path}")
        
    except Exception as e:
        print(f"❌ Error during model promotion: {e}")
        should_promote = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deployment Summary

# COMMAND ----------

# DBTITLE 1,Display deployment summary
print("=" * 60)
print("MODEL PROMOTION SUMMARY")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Catalog: {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"Promotion Decision: {'✅ PROMOTED' if should_promote else '❌ NOT PROMOTED'}")
print()

if should_promote:
    print("DEPLOYMENT STATUS:")
    print(f"  Production Alias: {PRODUCTION_ALIAS}")
    print(f"  Staging Alias: {STAGING_ALIAS}")
    print(f"  New Model Version: {new_version_number}")
    if current_production_version:
        print(f"  Previous Model Version: {current_production_version}")
    print()
    print("NEXT STEPS:")
    print("  1. Monitor production model performance")
    print("  2. Set up automated rollback if needed")
    print("  3. Update inference endpoints if applicable")
else:
    print("CURRENT STATUS:")
    print("  No changes to production deployment")
    print()
    print("NEXT STEPS:")
    print("  1. Investigate why new model didn't meet promotion criteria")
    print("  2. Consider hyperparameter tuning or feature engineering")
    print("  3. Retrain with different approach if needed")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Promotion Complete!
# MAGIC 
# MAGIC Successfully completed:
# MAGIC - Model performance comparison on validation set
# MAGIC - Automated promotion decision based on F1 score improvement
# MAGIC - Production alias management
# MAGIC - Staging alias for previous model
# MAGIC - Promotion logging and audit trail 