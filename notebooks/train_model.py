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
from datetime import datetime

from src.text_loader.loader import DataLoader
from src.train_model import train_model
from src.utils import get_widget_value, get_model_uri, print_parameters, get_table_name, setup_mlflow_experiment, create_text_classification_signature

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

# DBTITLE 1,Prepare features and labels using split column
# Extract feature columns
feature_columns = [col for col in data.columns if col.startswith('feature_')]

# Split data using the split column from data preparation
train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

# Prepare training data
X_train = train_data[feature_columns].values
y_train = train_data['party_encoded'].values

# Prepare test data
X_test = test_data[feature_columns].values
y_test = test_data['party_encoded'].values

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Feature matrix shape: {X_train.shape[1]}")
print(f"Training label distribution: {np.bincount(y_train)}")
print(f"Test label distribution: {np.bincount(y_test)}")

# Store full feature matrix for logging
X = data[feature_columns].values
y = data['party_encoded'].values

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training and Registration with MLflow and Unity Catalog

# COMMAND ----------

# DBTITLE 1,Train model and register to Unity Catalog
# Create run name with model type and timestamp
run_name = f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    # Set tags for documentation
    mlflow.set_tag("model_type", "LogisticRegression")
    mlflow.set_tag("task", "political_party_classification")
    mlflow.set_tag("data_source", "tweet_features")
    mlflow.set_tag("catalog", CATALOG_NAME)
    mlflow.set_tag("schema", SCHEMA_NAME)
    mlflow.set_tag("training_timestamp", datetime.now().isoformat())
    
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("max_features", X.shape[1])
    mlflow.log_param("data_split_method", "stratified_split_column")
    mlflow.log_param("train_split_ratio", 0.7)
    mlflow.log_param("test_split_ratio", 0.2)
    mlflow.log_param("validation_split_ratio", 0.1)
    mlflow.log_param("random_state", 42)
    
    # Train model using our train_model function
    print("Training model using our train_model function...")
    clf, metrics = train_model(X_train, y_train, X_test, y_test)
    
    print("Model training completed!")
    print(f"Model type: {type(clf).__name__}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
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
        def __init__(self, model, vectorizer, label_encoder, data_loader=None):
            self.model = model
            self.vectorizer = vectorizer
            self.label_encoder = label_encoder
            # Allow injection of a mock or custom DataLoader for testing
            self.data_loader = data_loader or DataLoader()
        
        def predict(self, context, model_input):
            # Handle different input formats
            import pandas as pd
            import numpy as np
            
            # Convert to pandas DataFrame if needed
            if isinstance(model_input, dict):
                model_input = pd.DataFrame(model_input)
            elif isinstance(model_input, np.ndarray):
                model_input = pd.DataFrame(model_input, columns=['text'])
            elif not isinstance(model_input, pd.DataFrame):
                model_input = pd.DataFrame(model_input)
            
            # Get the text column (first column)
            if 'text' in model_input.columns:
                text_column = model_input['text']
            else:
                text_column = model_input.iloc[:, 0]
            
            # Clean text using DataLoader's clean_text method
            cleaned_text = text_column.apply(self.data_loader.clean_text)
            
            # Vectorize
            X = self.vectorizer.transform(cleaned_text)
            
            # Predict
            predictions = self.model.predict(X)
            
            # Convert back to original labels
            return self.label_encoder.inverse_transform(predictions)
    
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
    
    # Create model signature for Unity Catalog
    signature = create_text_classification_signature()
    
    # Register model to Unity Catalog
    model_uri = get_model_uri(CATALOG_NAME, SCHEMA_NAME, MODEL_NAME)
    
    # Log the model with signature
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=custom_model,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}",
        signature=signature
    )
    
    print(f"Model registered successfully to: {model_uri}")
    
    # Log additional training information
    mlflow.log_param("training_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("feature_count", X.shape[1])
    mlflow.log_param("class_count", len(np.unique(y)))
    
    # Evaluate on validation set for model comparison
    validation_data = data[data['split'] == 'validation']
    X_val = validation_data[feature_columns].values
    y_val = validation_data['party_encoded'].values
    
    # Make predictions on validation set
    val_predictions = clf.predict(X_val)
    
    # Calculate validation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(y_val, val_predictions, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, val_predictions, average='weighted', zero_division=0)
    
    # Log validation metrics
    mlflow.log_metric("validation_accuracy", val_accuracy)
    mlflow.log_metric("validation_precision", val_precision)
    mlflow.log_metric("validation_recall", val_recall)
    mlflow.log_metric("validation_f1_score", val_f1)
    
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    
    # Store validation metrics for model promotion decision
    validation_metrics = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1_score': val_f1
    }
    
    # Save validation metrics to file for model promotion workflow
    import json
    validation_metrics_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_validation_metrics.json"
    with open(validation_metrics_path, 'w') as f:
        json.dump(validation_metrics, f, indent=2)
    
    print(f"Validation metrics saved to: {validation_metrics_path}")

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