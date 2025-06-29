# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC This notebook prepares the tweet data for training by:
# MAGIC 1. Loading the CSV data using our DataLoader
# MAGIC 2. Validating data quality
# MAGIC 3. Saving to Delta table in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# DBTITLE 1,Get parameters
dbutils.widgets.text("catalog_name", "mle_batch_catalog_2025_q2", "Catalog Name")
dbutils.widgets.text("schema_name", "mle_shyamkumar_vn", "Schema Name")
dbutils.widgets.text("table_name", "tweets_data", "Table Name")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")
TABLE_NAME = dbutils.widgets.get("table_name")

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Table: {TABLE_NAME}")

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
import sys
import os

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/src')
from text_loader.loader import DataLoader

print("DataLoader imported successfully")

# COMMAND ----------

# DBTITLE 1,Load data using DataLoader
# Initialize DataLoader with the CSV file path in DBFS
csv_path = "/dbfs/FileStore/tables/Tweets.csv"
loader = DataLoader(filepath=csv_path)

# Load the data
data = loader.data

print(f"Loaded {len(data)} tweets using DataLoader")
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

# DBTITLE 1,Data cleaning using DataLoader methods
# Use the clean_text method from DataLoader
data['Tweet_cleaned'] = data['Tweet'].apply(loader.clean_text)

# Remove rows with empty cleaned text
data_cleaned = data[data['Tweet_cleaned'].str.len() > 0]

print(f"Original data: {len(data)} rows")
print(f"After cleaning: {len(data_cleaned)} rows")
print(f"Removed {len(data) - len(data_cleaned)} rows with empty cleaned text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table

# COMMAND ----------

# DBTITLE 1,Save to Unity Catalog Delta table
full_table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"

# Convert pandas DataFrame to Spark DataFrame
df_cleaned = spark.createDataFrame(data_cleaned)

# Save to Delta table
df_cleaned.write.mode("overwrite").saveAsTable(full_table_name)

print(f"Data saved to Delta table: {full_table_name}")

# Verify the table
print("Verifying saved data:")
saved_df = spark.read.table(full_table_name)
print(f"Saved {saved_df.count()} rows")
saved_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Report

# COMMAND ----------

# DBTITLE 1,Generate data quality report
# Create a summary of the data
total_rows = len(data_cleaned)
party_counts = data_cleaned['Party'].value_counts()
avg_tweet_length = data_cleaned['tweet_length'].mean()

print("=== DATA QUALITY REPORT ===")
print(f"Total tweets: {total_rows}")
print(f"Average tweet length: {avg_tweet_length:.1f} characters")
print("\nParty distribution:")
for party, count in party_counts.items():
    percentage = (count / total_rows) * 100
    print(f"  {party}: {count} ({percentage:.1f}%)")

# Show sample of cleaned tweets
print("\n=== SAMPLE CLEANED TWEETS ===")
sample_tweets = data_cleaned[['Tweet', 'Tweet_cleaned', 'Party']].head(10)
for idx, row in sample_tweets.iterrows():
    print(f"Original: {row['Tweet'][:80]}...")
    print(f"Cleaned:  {row['Tweet_cleaned']}")
    print(f"Party:    {row['Party']}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation Complete!
# MAGIC 
# MAGIC The data has been successfully:
# MAGIC - Loaded using our DataLoader component
# MAGIC - Validated for quality
# MAGIC - Cleaned using DataLoader's text cleaning methods
# MAGIC - Saved to Unity Catalog Delta table
# MAGIC - Ready for feature engineering 