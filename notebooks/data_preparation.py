# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation and Feature Engineering
# MAGIC 
# MAGIC This notebook combines data preparation and feature engineering:
# MAGIC 1. Load and validate tweet data using DataLoader
# MAGIC 2. Extract features using DataLoader's preprocessing methods (which include text cleaning)
# MAGIC 3. Save features to Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Get parameters
# Get parameters from workflow, with fallback to default values
from src.utils import get_widget_value, get_table_name, print_parameters

CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
FEATURES_TABLE = get_widget_value("features_table", "tweet_features")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "Features Table": FEATURES_TABLE,
    "DBFS Base Path": DBFS_BASE_PATH
}
print_parameters(params)

# COMMAND ----------

# DBTITLE 1,Create catalog and schema if they don't exist
# Create catalog if it doesn't exist
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

print(f"Catalog and schema created/verified: {CATALOG_NAME}.{SCHEMA_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Validate Data Using DataLoader

# COMMAND ----------

# DBTITLE 1,Import DataLoader
import numpy as np
import pandas as pd
from src.text_loader.loader import DataLoader

print("DataLoader imported successfully")

# COMMAND ----------

# DBTITLE 1,Load data using DataLoader
# Initialize DataLoader and load data with the CSV file path in DBFS
csv_path = f"{DBFS_BASE_PATH}/Tweets.csv"
loader = DataLoader()
loader.load_data(filepath=csv_path)

# Load the data (null tweets are automatically filtered in DataLoader)
data = loader.data

print(f"Data columns: {list(data.columns)}")
data.head()

# COMMAND ----------

# DBTITLE 1,Data validation using DataLoader
# Check for null values
print("Null value counts:")
print(data.isnull().sum())

# Check data distribution
print("\nParty distribution:")
print(data['Party'].value_counts())

# Check tweet length distribution
data['tweet_length'] = data['Tweet'].str.len()
print("\nTweet length statistics:")
print(data['tweet_length'].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Using DataLoader

# COMMAND ----------

# DBTITLE 1,Use DataLoader's preprocessing methods
# Use DataLoader's preprocess_tweets method (which includes text cleaning)
print("Extracting features using DataLoader's preprocess_tweets method...")
X = loader.preprocess_tweets()

# Use DataLoader's preprocess_parties method to get labels
print("Encoding labels using DataLoader's preprocess_parties method...")
y = loader.preprocess_parties()

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Features to Delta Table

# COMMAND ----------

# DBTITLE 1,Create features DataFrame
# Create feature names
feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# Create features DataFrame with original data and features
features_df = pd.DataFrame(X, columns=feature_names)
features_df = pd.concat([data.reset_index(drop=True), features_df], axis=1)

# Add encoded labels
features_df['party_encoded'] = y

print("Features DataFrame created:")
print(f"Shape: {features_df.shape}")
print(f"Columns: {list(features_df.columns)}")

# COMMAND ----------

# DBTITLE 1,Save to Delta table
features_table_name = get_table_name(CATALOG_NAME, SCHEMA_NAME, FEATURES_TABLE)

# Convert to Spark DataFrame
spark_features_df = spark.createDataFrame(features_df)

# Save features to Delta table
spark_features_df.write.mode("overwrite").saveAsTable(features_table_name)

print(f"Features saved to Delta table: {features_table_name}")

# Verify the saved data
saved_features = spark.read.table(features_table_name)
print(f"Saved {saved_features.count()} rows with {len(feature_names)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Statistics

# COMMAND ----------

# DBTITLE 1,Generate feature statistics
# Calculate feature statistics
feature_stats = []
for i, feature_name in enumerate(feature_names):
    feature_values = X[:, i]
    feature_stats.append({
        'feature_name': feature_name,
        'mean': float(np.mean(feature_values)),
        'std': float(np.std(feature_values)),
        'min': float(np.min(feature_values)),
        'max': float(np.max(feature_values)),
        'non_zero_count': int(np.count_nonzero(feature_values))
    })

# Display top features by non-zero count
feature_stats_df = pd.DataFrame(feature_stats)
top_features = feature_stats_df.nlargest(10, 'non_zero_count')

print("Top 10 features by non-zero count:")
print(top_features[['feature_name', 'non_zero_count', 'mean', 'std']])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Vectorizer and Encoder for Model Training

# COMMAND ----------

# DBTITLE 1,Save preprocessing components as artifacts
import pickle
import os

# Save vectorizer (from DataLoader)
vectorizer_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_tfidf_vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(loader.vectorizer, f)

# Save label encoder (from DataLoader)
encoder_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_label_encoder.pkl"
with open(encoder_path, 'wb') as f:
    pickle.dump(loader.encoder, f)

print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Label encoder saved to: {encoder_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Report

# COMMAND ----------

# DBTITLE 1,Generate data quality report
# Create a summary of the data
total_rows = len(data)
party_counts = data['Party'].value_counts()
avg_tweet_length = data['tweet_length'].mean()

print("=== DATA QUALITY REPORT ===")
print(f"Total tweets: {total_rows}")
print(f"Average tweet length: {avg_tweet_length:.1f} characters")
print("\nParty distribution:")
for party, count in party_counts.items():
    percentage = (count / total_rows) * 100
    print(f"  {party}: {count} ({percentage:.1f}%)")

# Show sample of original tweets
print("\n=== SAMPLE ORIGINAL TWEETS ===")
sample_tweets = data[['Tweet', 'Party']].head(10)
for idx, row in sample_tweets.iterrows():
    print(f"Tweet: {row['Tweet'][:80]}...")
    print(f"Party: {row['Party']}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataLoader Component Summary

# COMMAND ----------

# DBTITLE 1,Display DataLoader component information
print("=== DATALOADER COMPONENT SUMMARY ===")
print(f"Input data: {len(data)} tweets from CSV")
print(f"Vectorizer type: {type(loader.vectorizer).__name__}")
print(f"Vectorizer max features: {loader.vectorizer.max_features}")
print(f"Label encoder type: {type(loader.encoder).__name__}")
print(f"Label encoder classes: {loader.encoder.classes_}")
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")
print(f"Output: {features_table_name} with {len(feature_names)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation and Feature Engineering Complete!
# MAGIC 
# MAGIC The data has been successfully:
# MAGIC - Loaded using our DataLoader component
# MAGIC - Validated for quality
# MAGIC - Feature extracted using DataLoader's preprocessing methods (including text cleaning)
# MAGIC - Saved to Unity Catalog Delta table
# MAGIC - Ready for model training 