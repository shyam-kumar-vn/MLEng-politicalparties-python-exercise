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
# MAGIC ## Data Loading and Preprocessing

# COMMAND ----------

# DBTITLE 1,Load and preprocess data using DataLoader
# Initialize DataLoader and load data
loader = DataLoader()
csv_path = f"{DBFS_BASE_PATH}/Tweets.csv"
loader.load_data(filepath=csv_path)

# Get preprocessed features and labels
X = loader.preprocess_tweets()
y = loader.preprocess_parties()

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")

# COMMAND ----------

# DBTITLE 1,Load features from Delta table for train/test/validation split
# Load features from Delta table (created in data_preparation notebook)
features_table_name = get_table_name(CATALOG_NAME, SCHEMA_NAME, "tweet_features")
data = spark.read.table(features_table_name).toPandas()

print(f"Loaded {len(data)} samples from features table")

# Get feature columns (all columns that start with 'feature_')
feature_columns = [col for col in data.columns if col.startswith('feature_')]
print(f"Found {len(feature_columns)} feature columns")

# Split data based on the 'split' column
train_data = data[data['split'] == 'train']
validation_data = data[data['split'] == 'validation']
test_data = data[data['split'] == 'test']

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Test samples: {len(test_data)}")

# Prepare training data
X_train = train_data[feature_columns].values
y_train = train_data['party_encoded'].values

# Prepare test data
X_test = test_data[feature_columns].values
y_test = test_data['party_encoded'].values

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training

# COMMAND ----------

# DBTITLE 1,Train the model
# Train the model using our train_model function
clf, metrics = train_model(X_train, y_train, X_test, y_test)

print("Training completed!")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test Precision: {metrics['precision']:.4f}")
print(f"Test Recall: {metrics['recall']:.4f}")
print(f"Test F1 Score: {metrics['f1_score']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Preprocessing Components

# COMMAND ----------

# DBTITLE 1,Save vectorizer and label encoder for model deployment
import pickle

# Save the vectorizer
vectorizer_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_tfidf_vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(loader.vectorizer, f)
print(f"Vectorizer saved to: {vectorizer_path}")

# Save the label encoder
encoder_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_label_encoder.pkl"
with open(encoder_path, 'wb') as f:
    pickle.dump(loader.encoder, f)
print(f"Label encoder saved to: {encoder_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registration with MLflow

# COMMAND ----------

# DBTITLE 1,Register model to Unity Catalog
# Start MLflow run for model registration
with mlflow.start_run(run_name="political_party_classifier_training") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("training_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])
    mlflow.log_param("feature_count", X_train.shape[1])
    mlflow.log_param("class_count", len(np.unique(y_train)))
    
    # Log metrics
    mlflow.log_metric("test_accuracy", metrics['accuracy'])
    mlflow.log_metric("test_precision", metrics['precision'])
    mlflow.log_metric("test_recall", metrics['recall'])
    mlflow.log_metric("test_f1_score", metrics['f1_score'])
    
    # Log confusion matrix as artifact
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    confusion_matrix_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(confusion_matrix_path)
    plt.close()
    
    # Create a custom model with embedded preprocessing components
    class PoliticalPartyClassifier(mlflow.pyfunc.PythonModel):
        def __init__(self, model, vectorizer, label_encoder):
            self.model = model
            self.vectorizer = vectorizer
            self.label_encoder = label_encoder
        
        @staticmethod
        def clean_text(text):
            """Embedded text cleaning function to avoid import issues"""
            import re
            
            # Handle non-string input
            if not isinstance(text, str):
                return ""
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove all non-alphabetic characters (including numbers and punctuation)
            text = re.sub(r'[^a-zA-Z]', '', text)
            return text.strip()
        
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
            
            # Clean text using embedded clean_text method
            cleaned_text = text_column.apply(self.clean_text)
            
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
    
    # Add tags for better organization
    mlflow.set_tag("model_type", "political_party_classifier")
    mlflow.set_tag("framework", "scikit-learn")
    mlflow.set_tag("task", "text_classification")
    mlflow.set_tag("author", "mle_shyamkumar_vn")
    mlflow.set_tag("version", "1.0.0")
    
    print("Validation metrics:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Promotion to Production

# COMMAND ----------

# DBTITLE 1,Promote model to production alias
# Get the latest model version
client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(
    f"name='{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}'"
)

if model_versions:
    latest_version = max(model_versions, key=lambda x: x.version)
    
    # Transition to production
    client.transition_model_version_stage(
        name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}",
        version=latest_version.version,
        stage="Production"
    )
    
    print(f"Model version {latest_version.version} promoted to Production")
    print(f"Model URI: {model_uri}")
else:
    print("No model versions found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Summary

# COMMAND ----------

# DBTITLE 1,Display training summary
print("="*80)
print("POLITICAL PARTY CLASSIFIER TRAINING SUMMARY")
print("="*80)
print(f"Model: {CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Run ID: {run.info.run_id}")
print()
print("DATASET:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Validation samples: {len(validation_data)}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(np.unique(y_train))}")
print()
print("PERFORMANCE:")
print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
print(f"  Test Precision: {metrics['precision']:.4f}")
print(f"  Test Recall: {metrics['recall']:.4f}")
print(f"  Test F1 Score: {metrics['f1_score']:.4f}")
print()
print("VALIDATION:")
print(f"  Validation Accuracy: {val_accuracy:.4f}")
print(f"  Validation Precision: {val_precision:.4f}")
print(f"  Validation Recall: {val_recall:.4f}")
print(f"  Validation F1 Score: {val_f1:.4f}")
print()
print("ARTIFACTS:")
print(f"  Model URI: {model_uri}")
print(f"  Vectorizer: {vectorizer_path}")
print(f"  Label Encoder: {encoder_path}")
print(f"  Confusion Matrix: {confusion_matrix_path}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Complete!
# MAGIC 
# MAGIC The political party classifier has been successfully:
# MAGIC - Trained on the tweet dataset
# MAGIC - Evaluated on test and validation sets
# MAGIC - Registered to Unity Catalog
# MAGIC - Promoted to production
# MAGIC 
# MAGIC The model is now ready for deployment and inference! 