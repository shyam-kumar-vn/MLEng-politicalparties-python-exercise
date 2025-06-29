# Databricks notebook source
# MAGIC %md
# MAGIC # Political Party Tweet Classification - Training Pipeline
# MAGIC 
# MAGIC This notebook implements the batch training workflow using our existing components:
# MAGIC 1. Data Loading and Preprocessing using DataLoader
# MAGIC 2. Feature Engineering using DataLoader
# MAGIC 3. Model Training using our train_model function
# MAGIC 4. Model Registration to Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries and components
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.text_loader.loader import DataLoader
from src.train_model import train_model
from src.utils import get_widget_value, get_model_uri, print_parameters, get_table_name, setup_mlflow_experiment

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure MLflow and Unity Catalog
# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set the catalog and schema for model registration
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")  # Replace with your name
MODEL_NAME = get_widget_value("model_name", "political_party_classifier")
EXPERIMENT_NAME = get_widget_value("experiment_name", "/Shared/mle_shyamkumar_vn_tweet_classification")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")

# Set up MLflow experiment
setup_mlflow_experiment(EXPERIMENT_NAME)

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "Model": MODEL_NAME,
    "Experiment": EXPERIMENT_NAME,
    "DBFS Base Path": DBFS_BASE_PATH
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Loading and Preprocessing Using DataLoader

# COMMAND ----------

# DBTITLE 1,Load data from Delta table using DataLoader
# Load data from Delta table
features_table = get_table_name(CATALOG_NAME, SCHEMA_NAME, "tweet_features")
df = spark.read.table(features_table)

# Convert to pandas
data = df.toPandas()

print(f"Loaded {len(data)} samples from {features_table}")

# COMMAND ----------

# DBTITLE 1,Prepare features and labels
# Extract feature columns
feature_columns = [col for col in data.columns if col.startswith('feature_')]
X = data[feature_columns].values
y = data['party_encoded'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training Using Our train_model Function

# COMMAND ----------

# DBTITLE 1,Split data and train model
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# COMMAND ----------

# DBTITLE 1,Train model using our train_model function
# Use our existing train_model function
print("Training model using our train_model function...")
clf, metrics = train_model(X_train, y_train, X_test, y_test)

print("Model training completed!")
print(f"Model type: {type(clf).__name__}")
print(f"Accuracy: {metrics['accuracy']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registration with MLflow and Unity Catalog

# COMMAND ----------

# DBTITLE 1,Register Model to Unity Catalog
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("max_features", X.shape[1])
    
    # Log metrics from our train_model function
    mlflow.log_metric("accuracy", metrics['accuracy'])
    mlflow.log_metric("precision", metrics['precision'])
    mlflow.log_metric("recall", metrics['recall'])
    mlflow.log_metric("f1_score", metrics['f1_score'])
    
    # Log confusion matrix as artifact
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('/tmp/confusion_matrix.png')
    mlflow.log_artifact('/tmp/confusion_matrix.png')
    
    # Create a custom model with preprocessing components
    class PoliticalPartyClassifier(mlflow.pyfunc.PythonModel):
        def __init__(self, model, vectorizer, label_encoder):
            self.model = model
            self.vectorizer = vectorizer
            self.label_encoder = label_encoder
        
        def predict(self, context, model_input):
            # Clean text using DataLoader's clean_text method
            cleaned_text = model_input.iloc[:, 0].apply(self.clean_text)
            # Vectorize
            X = self.vectorizer.transform(cleaned_text)
            # Predict
            predictions = self.model.predict(X)
            # Convert back to original labels
            return self.label_encoder.inverse_transform(predictions)
        
        def clean_text(self, text):
            """Clean text using DataLoader's method."""
            if not isinstance(text, str):
                return ""
            import re
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove all non-alphabetic characters
            text = re.sub(r'[^a-zA-Z]', '', text)
            return text.strip()
    
    # Load preprocessing components
    import pickle
    vectorizer_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_tfidf_vectorizer.pkl"
    encoder_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_label_encoder.pkl"
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Create and log the custom model
    custom_model = PoliticalPartyClassifier(clf, vectorizer, label_encoder)
    
    # Register model to Unity Catalog
    model_uri = get_model_uri(CATALOG_NAME, SCHEMA_NAME, MODEL_NAME)
    
    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=custom_model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
    )
    
    print(f"Model registered successfully to: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Summary

# COMMAND ----------

# DBTITLE 1,Display training summary
print("=" * 60)
print("MODEL TRAINING SUMMARY")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Catalog: {CATALOG_NAME}.{SCHEMA_NAME}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[1]}")
print(f"Features: {X.shape[1]}")
print()
print("PERFORMANCE METRICS:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1 Score:  {metrics['f1_score']:.4f}")
print()
print("COMPONENTS USED:")
print(f"  DataLoader: {type(DataLoader()).__name__}")
print(f"  train_model function: {train_model.__name__}")
print(f"  Model: {type(clf).__name__}")
print(f"  Vectorizer: {type(vectorizer).__name__}")
print(f"  Label Encoder: {type(label_encoder).__name__}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Complete!
# MAGIC 
# MAGIC The model has been successfully:
# MAGIC - Loaded and preprocessed using DataLoader
# MAGIC - Trained using our train_model function
# MAGIC - Evaluated with comprehensive metrics
# MAGIC - Registered to Unity Catalog
# MAGIC - Ready for inference 