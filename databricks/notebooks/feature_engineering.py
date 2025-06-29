# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC 
# MAGIC This notebook performs feature engineering on the tweet data using our DataLoader:
# MAGIC 1. Load data from Delta table using DataLoader
# MAGIC 2. Use DataLoader's preprocessing methods
# MAGIC 3. Extract TF-IDF features
# MAGIC 4. Save features to Delta table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Get parameters
dbutils.widgets.text("catalog_name", "mle_batch_catalog_2025_q2", "Catalog Name")
dbutils.widgets.text("schema_name", "mle_shyamkumar_vn", "Schema Name")
dbutils.widgets.text("input_table", "tweets_data", "Input Table Name")
dbutils.widgets.text("features_table", "tweet_features", "Features Table Name")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")
INPUT_TABLE = dbutils.widgets.get("input_table")
FEATURES_TABLE = dbutils.widgets.get("features_table")

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Input Table: {INPUT_TABLE}")
print(f"Features Table: {FEATURES_TABLE}")

# COMMAND ----------

# DBTITLE 1,Import DataLoader
import sys
import os

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/src')
from text_loader.loader import DataLoader

print("DataLoader imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# DBTITLE 1,Load data from Delta table
input_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{INPUT_TABLE}"
df = spark.read.table(input_table_name)

# Convert to pandas for DataLoader processing
data = df.toPandas()

print(f"Loaded {len(data)} tweets from {input_table_name}")
print(f"Data columns: {list(data.columns)}")
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Using DataLoader

# COMMAND ----------

# DBTITLE 1,Initialize DataLoader with loaded data
# Create a DataLoader instance and set the data
loader = DataLoader()
loader.data = data

print("DataLoader initialized with Delta table data")

# COMMAND ----------

# DBTITLE 1,Use DataLoader's preprocessing methods
# Use DataLoader's preprocess_tweets method to get features
print("Extracting features using DataLoader...")
X = loader.preprocess_tweets()

# Use DataLoader's preprocess_parties method to get labels
print("Encoding labels using DataLoader...")
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
features_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{FEATURES_TABLE}"

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
import numpy as np

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
vectorizer_path = f"/dbfs/FileStore/tables/{SCHEMA_NAME}_tfidf_vectorizer.pkl"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(loader.vectorizer, f)

# Save label encoder (from DataLoader)
encoder_path = f"/dbfs/FileStore/tables/{SCHEMA_NAME}_label_encoder.pkl"
with open(encoder_path, 'wb') as f:
    pickle.dump(loader.encoder, f)

print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Label encoder saved to: {encoder_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataLoader Component Summary

# COMMAND ----------

# DBTITLE 1,Display DataLoader component information
print("=== DATALOADER COMPONENT SUMMARY ===")
print(f"Vectorizer type: {type(loader.vectorizer).__name__}")
print(f"Vectorizer max features: {loader.vectorizer.max_features}")
print(f"Label encoder type: {type(loader.encoder).__name__}")
print(f"Label encoder classes: {loader.encoder.classes_}")
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Complete!
# MAGIC 
# MAGIC Successfully completed using DataLoader components:
# MAGIC - Data loaded using DataLoader
# MAGIC - Text preprocessing using DataLoader's clean_text method
# MAGIC - Feature extraction using DataLoader's vectorize_text method
# MAGIC - Label encoding using DataLoader's label_encoder method
# MAGIC - Features saved to Delta table
# MAGIC - Vectorizer and encoder saved for model training
# MAGIC - Feature statistics generated 