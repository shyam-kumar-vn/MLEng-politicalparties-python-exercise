# Databricks Batch Training Workflow

This directory contains the Databricks integration for the Political Party Tweet Classification project.

## Overview

The batch training workflow consists of four main tasks that reuse our existing components:
1. **Data Preparation** - Load and validate tweet data using DataLoader
2. **Feature Engineering** - Extract features using DataLoader's preprocessing methods
3. **Model Training** - Train the classification model using our train_model function
4. **Model Evaluation** - Evaluate model performance and generate reports

## Component Reuse

The Databricks notebooks are designed to reuse the existing components from the `src/` directory:

- **DataLoader** (`src/text_loader/loader.py`) - Used for data loading and preprocessing
- **train_model function** (`src/train_model.py`) - Used for model training and evaluation

This ensures consistency between local development and Databricks execution.

## Directory Structure

```
databricks/
├── notebooks/
│   ├── data_preparation.py      # Data loading using DataLoader
│   ├── feature_engineering.py   # Feature extraction using DataLoader
│   ├── train_model.py          # Model training using train_model function
│   └── model_evaluation.py     # Model evaluation and reporting
├── workflows/
│   └── training_workflow.json   # Workflow configuration
├── deploy_workflow.py          # Deployment script (uploads src/ directory)
└── README.md                   # This file
```

## Prerequisites

### 1. Databricks Setup
Follow the setup instructions in `env_setup.md`:
- Login to Databricks workspace
- Create compute cluster using "Min Cluster Policy MLE"
- Create schema under catalog: `mle_batch_catalog_2025_q2`

### 2. Databricks CLI
Install and configure Databricks CLI:
```bash
pip install databricks-cli
databricks configure --token
```

### 3. Data Upload
Upload the `data/Tweets.csv` file to Databricks FileStore:
```bash
databricks fs cp data/Tweets.csv dbfs:/FileStore/tables/Tweets.csv
```

## Configuration

### Catalog and Schema
- **Catalog**: `mle_batch_catalog_2025_q2`
- **Schema**: `mle_shyamkumar_vn` (replace with your name)
- **Model Name**: `political_party_classifier`

### Workflow Parameters
The workflow uses the following naming convention as per `env_setup.md`:
- Workflow name: `mle_shyamkumar_vn_training_workflow`
- Experiment name: `/Shared/mle_shyamkumar_vn_tweet_classification`

## Deployment

### Option 1: Using the Deployment Script (Recommended)
```bash
cd databricks
python deploy_workflow.py
```

The script will:
1. Check Databricks CLI installation
2. Upload the entire `src/` directory to Databricks workspace
3. Upload notebooks to workspace
4. Deploy the workflow
5. Return the Job ID

### Option 2: Manual Deployment

#### 1. Upload Source Code
```bash
# Upload the src directory
databricks workspace import --language PYTHON --overwrite \
  src /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/src

# Upload individual source files
databricks workspace import --language PYTHON --overwrite \
  src/text_loader/loader.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/src/text_loader/loader

databricks workspace import --language PYTHON --overwrite \
  src/train_model.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/src/train_model
```

#### 2. Upload Notebooks
```bash
# Upload each notebook
databricks workspace import --language PYTHON --overwrite \
  databricks/notebooks/data_preparation.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/databricks/notebooks/data_preparation

databricks workspace import --language PYTHON --overwrite \
  databricks/notebooks/feature_engineering.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/databricks/notebooks/feature_engineering

databricks workspace import --language PYTHON --overwrite \
  databricks/notebooks/train_model.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/databricks/notebooks/train_model

databricks workspace import --language PYTHON --overwrite \
  databricks/notebooks/model_evaluation.py \
  /Repos/mle_shyamkumar_vn/political-parties-tweet-classifier/databricks/notebooks/model_evaluation
```

#### 3. Create Workflow
```bash
# Update cluster_id in workflow JSON
# Then create the job
databricks jobs create --json-file databricks/workflows/training_workflow.json
```

## Workflow Details

### Task Dependencies
```
data_preparation → feature_engineering → model_training → model_evaluation
```

### Task Descriptions

#### 1. Data Preparation
- Uses DataLoader to load CSV data from FileStore
- Validates data quality (nulls, distribution)
- Uses DataLoader's clean_text method for text cleaning
- Saves to Delta table: `{catalog}.{schema}.tweets_data`

#### 2. Feature Engineering
- Loads data from Delta table
- Uses DataLoader's preprocessing methods (preprocess_tweets, preprocess_parties)
- Extracts TF-IDF features using DataLoader's vectorizer
- Saves features to Delta table: `{catalog}.{schema}.tweet_features`
- Saves vectorizer and encoder for model training

#### 3. Model Training
- Loads features from Delta table
- Uses our train_model function for training and evaluation
- Registers model to Unity Catalog
- Logs experiment with MLflow

#### 4. Model Evaluation
- Loads trained model from Unity Catalog
- Evaluates on test data
- Generates performance metrics
- Creates visualizations (confusion matrix, F1 scores)
- Saves evaluation report

## Component Integration

### DataLoader Integration
- **Import**: `from text_loader.loader import DataLoader`
- **Usage**: All data loading and preprocessing uses DataLoader methods
- **Consistency**: Same text cleaning and feature extraction as local development

### train_model Function Integration
- **Import**: `from train_model import train_model`
- **Usage**: Model training uses the same function as local development
- **Benefits**: Consistent training logic and evaluation metrics

## Outputs

### Delta Tables
- `{catalog}.{schema}.tweets_data` - Cleaned tweet data
- `{catalog}.{schema}.tweet_features` - TF-IDF features

### Model
- Registered in Unity Catalog: `{catalog}.{schema}.political_party_classifier`
- MLflow experiment: `/Shared/mle_shyamkumar_vn_tweet_classification`

### Artifacts
- Confusion matrix plot
- F1 scores visualization
- Evaluation summary JSON
- TF-IDF vectorizer pickle file
- Label encoder pickle file

## Monitoring

### Workflow Monitoring
- Monitor workflow runs in Databricks Jobs UI
- Email notifications on start/success/failure
- Logs available in each task

### Model Monitoring
- MLflow experiment tracking
- Model versioning in Unity Catalog
- Performance metrics logged

## Troubleshooting

### Common Issues

1. **Import errors for DataLoader or train_model**
   - Ensure src directory is uploaded to Databricks workspace
   - Check Python path in notebooks
   - Verify file structure in Databricks workspace

2. **Cluster not found**
   - Verify cluster ID in workflow configuration
   - Ensure cluster is running

3. **Permission errors**
   - Check Unity Catalog permissions
   - Verify schema creation permissions

4. **Data not found**
   - Ensure CSV file is uploaded to FileStore
   - Check file path in notebooks

5. **MLflow errors**
   - Verify MLflow is enabled on cluster
   - Check Unity Catalog model registry permissions

### Debugging
- Check task logs in Databricks Jobs UI
- Review notebook outputs for error messages
- Verify parameter values in workflow configuration
- Check that src directory is properly uploaded

## Next Steps

After successful workflow deployment:
1. Run the workflow manually or schedule it
2. Monitor the training process
3. Review model performance in MLflow
4. Proceed to model serving endpoint creation
5. Set up batch inference workflow

## Support

For issues with:
- **Databricks setup**: Refer to `env_setup.md`
- **Workflow configuration**: Check workflow JSON syntax
- **Notebook errors**: Review notebook logs and outputs
- **Model registration**: Verify Unity Catalog permissions
- **Component imports**: Ensure src directory is uploaded correctly 