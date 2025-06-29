# Databricks Batch Training Workflow

This directory contains the Databricks integration for the Political Party Tweet Classification project.

## Overview

The batch training workflow consists of three main tasks that reuse our existing components:
1. **Data Preparation and Feature Engineering** - Load data and extract features using DataLoader's preprocessing methods
2. **Model Training** - Train the classification model using our train_model function
3. **Model Evaluation** - Evaluate model performance and generate reports

## Component Reuse

The Databricks notebooks are designed to reuse the existing components from the `src/` directory:

- **DataLoader** (`src/text_loader/loader.py`) - Used for data loading and feature preprocessing (includes text cleaning)
- **train_model function** (`src/train_model.py`) - Used for model training and evaluation

This ensures consistency between local development and Databricks execution.

## Directory Structure

```
databricks/
├── notebooks/
│   ├── data_preparation.py      # Combined data loading and feature extraction
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

### 2. GitHub Repository Access
- Access to the GitHub repository containing this project
- Ability to clone or download the repository

## Setup Instructions (Without Databricks CLI)

### Step 1: Clone/Download the Repository

#### Option A: Clone from GitHub (if you have access)
```bash
git clone <repository-url>
cd MLEng-politicalparties-python-exercise-1
```

#### Option B: Download from GitHub
1. Go to the GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file to your local machine
4. Navigate to the extracted directory

### Step 2: Prepare the Project Files

1. **Verify the project structure**:
   ```
   MLEng-politicalparties-python-exercise-1/
   ├── data/
   │   └── Tweets.csv
   ├── src/
   │   ├── text_loader/
   │   │   └── loader.py
   │   └── train_model.py
   ├── databricks/
   │   ├── notebooks/
   │   │   ├── data_preparation.py
   │   │   ├── train_model.py
   │   │   └── model_evaluation.py
   │   └── workflows/
   │       └── training_workflow.json
   └── README.md
   ```

2. **Ensure you have the required files**:
   - `data/Tweets.csv` - The input dataset
   - `src/text_loader/loader.py` - DataLoader component
   - `src/train_model.py` - Training function
   - `databricks/notebooks/*.py` - Databricks notebooks

### Step 3: Upload to Databricks Workspace

#### 3.1 Upload Source Code
1. In Databricks workspace, go to **Workspace** → **Users** → **shyamkumar.vn@thoughtworks.com**
2. Navigate to the `MLEng-politicalparties-python-exercise-fork` folder
3. Upload the `src/` directory:
   - Create folder: `src/text_loader/`
   - Upload `src/text_loader/loader.py` to `src/text_loader/`
   - Upload `src/train_model.py` to `src/`
   - Upload `src/text_loader/__init__.py` if it exists

#### 3.2 Upload Notebooks
1. Create folder: `databricks/notebooks/` in your workspace
2. Upload each notebook:
   - `databricks/notebooks/data_preparation.py`
   - `databricks/notebooks/train_model.py`
   - `databricks/notebooks/model_evaluation.py`

#### 3.3 Upload Data
1. Go to **Data** → **Add Data** → **Upload File**
2. Upload `data/Tweets.csv`
3. Note the DBFS path (e.g., `/FileStore/tables/Tweets.csv`)
4. **Important**: Move the file to your personal directory:
   ```python
   # Run this in a notebook or use FileStore UI
   dbutils.fs.mv("/FileStore/tables/Tweets.csv", "/FileStore/shyamkumar.vn/Tweets.csv")
   ```

### Step 4: Create Unity Catalog Structure

1. **Create Catalog** (if not exists):
   ```sql
   CREATE CATALOG IF NOT EXISTS mle_batch_catalog_2025_q2;
   ```

2. **Create Schema** (if not exists):
   ```sql
   CREATE SCHEMA IF NOT EXISTS mle_batch_catalog_2025_q2.mle_shyamkumar_vn;
   ```

### Step 5: Create and Configure the Workflow

#### 5.1 Create a New Job
1. Go to **Workflows** → **Jobs**
2. Click **Create Job**
3. Name: `mle_shyamkumar_vn_training_workflow`

#### 5.2 Add Tasks

**Task 1: Data Preparation and Feature Engineering**
1. Click **Add task** → **Notebook**
2. **Task name**: `data_preparation_and_feature_engineering`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/data_preparation`
4. **Cluster**: Select your cluster
5. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `features_table`: `tweet_features`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`

**Task 2: Model Training**
1. Click **Add task** → **Notebook**
2. **Task name**: `model_training`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/train_model`
4. **Cluster**: Select your cluster
5. **Dependencies**: Add dependency on `data_preparation_and_feature_engineering`
6. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `model_name`: `political_party_classifier`
   - `experiment_name`: `/Shared/mle_shyamkumar_vn_tweet_classification`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`

**Task 3: Model Evaluation**
1. Click **Add task** → **Notebook**
2. **Task name**: `model_evaluation`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/model_evaluation`
4. **Cluster**: Select your cluster
5. **Dependencies**: Add dependency on `model_training`
6. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `model_name`: `political_party_classifier`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`

#### 5.3 Configure Job Settings
1. **Email notifications**: Add your email for start/success/failure notifications
2. **Timeout**: Set appropriate timeouts (3600s for data prep, 7200s for training, 1800s for evaluation)
3. **Retry policy**: Configure as needed

### Step 6: Run the Workflow

#### 6.1 Manual Run
1. Go to your job in **Workflows** → **Jobs**
2. Click **Run now**
3. Monitor the progress in the job details page

#### 6.2 Schedule Run (Optional)
1. In job settings, configure **Schedule**
2. Set frequency (daily, weekly, etc.)
3. Set start time and timezone

### Step 7: Monitor and Verify

#### 7.1 Check Job Status
1. Go to **Workflows** → **Jobs**
2. Click on your job to see run history
3. Check task status and logs

#### 7.2 Verify Outputs
1. **Delta Tables**: Check `mle_batch_catalog_2025_q2.mle_shyamkumar_vn.tweet_features`
2. **Model**: Check Unity Catalog → `mle_batch_catalog_2025_q2.mle_shyamkumar_vn.political_party_classifier`
3. **Artifacts**: Check `/dbfs/FileStore/shyamkumar.vn/` for:
   - `mle_shyamkumar_vn_tfidf_vectorizer.pkl`
   - `mle_shyamkumar_vn_label_encoder.pkl`
   - `mle_shyamkumar_vn_confusion_matrix.png`
   - `mle_shyamkumar_vn_f1_scores.png`
   - `mle_shyamkumar_vn_evaluation_summary.json`

#### 7.3 MLflow Experiment
1. Go to **MLflow** → **Experiments**
2. Find `/Shared/mle_shyamkumar_vn_tweet_classification`
3. View metrics, parameters, and artifacts

## Configuration

### Workflow Parameters
The workflow uses configurable parameters that are passed to all notebooks:

- **catalog_name**: `mle_batch_catalog_2025_q2` (Unity Catalog)
- **schema_name**: `mle_shyamkumar_vn` (replace with your name)
- **dbfs_base_path**: `/dbfs/FileStore/shyamkumar.vn` (DBFS base path for artifacts)
- **model_name**: `political_party_classifier`
- **experiment_name**: `/Shared/mle_shyamkumar_vn_tweet_classification`

### Workflow Naming Convention
The workflow uses the following naming convention as per `env_setup.md`:
- Workflow name: `mle_shyamkumar_vn_training_workflow`

## Alternative: Using Databricks CLI (If Available)

If you have access to Databricks CLI, you can use the automated deployment:

### 1. Install Databricks CLI
```bash
pip install databricks-cli
databricks configure --token
```

### 2. Run Deployment Script
```bash
cd databricks
python deploy_workflow.py
```

## Workflow Details

### Task Dependencies
```
data_preparation_and_feature_engineering → model_training → model_evaluation
```

### Task Descriptions

#### 1. Data Preparation and Feature Engineering
- Uses DataLoader to load CSV data from FileStore
- Validates data quality (nulls, distribution)
- Uses DataLoader's preprocessing methods (preprocess_tweets, preprocess_parties) which include text cleaning
- Saves features to Delta table: `{catalog}.{schema}.tweet_features`
- Saves vectorizer and encoder from DataLoader for model training

#### 2. Model Training
- Loads features from Delta table
- Uses our train_model function for training and evaluation
- Registers model to Unity Catalog
- Logs experiment with MLflow

#### 3. Model Evaluation
- Loads trained model from Unity Catalog
- Evaluates on test data
- Generates performance metrics
- Creates visualizations (confusion matrix, F1 scores)
- Saves evaluation report

## Component Integration

### DataLoader Integration
- **Import**: `from text_loader.loader import DataLoader`
- **Usage**: 
  - Data loading using DataLoader
  - Feature extraction using DataLoader's preprocessing methods (includes text cleaning)
- **Consistency**: Same text cleaning and feature extraction logic as local development

### train_model Function Integration
- **Import**: `from train_model import train_model`
- **Usage**: Model training uses the same function as local development
- **Benefits**: Consistent training logic and evaluation metrics

## Data Flow

### Step 1: Data Preparation and Feature Engineering
- Input: `Tweets.csv` from FileStore
- Processing: 
  - DataLoader loads data
  - DataLoader's `preprocess_tweets()` for TF-IDF features (includes text cleaning)
  - DataLoader's `preprocess_parties()` for label encoding
- Output: Delta table with original data + feature columns + `party_encoded`

### Step 2: Model Training
- Input: Features Delta table
- Processing: Uses train_model function for training and evaluation
- Output: Registered model in Unity Catalog

### Step 3: Model Evaluation
- Input: Trained model + test data
- Processing: Performance evaluation and visualization
- Output: Evaluation reports and visualizations

## Outputs

### Delta Tables
- `{catalog}.{schema}.tweet_features` - TF-IDF features

### Model
- Registered in Unity Catalog: `{catalog}.{schema}.political_party_classifier`
- MLflow experiment: `/Shared/mle_shyamkumar_vn_tweet_classification`

### Artifacts (saved to {dbfs_base_path})
- Confusion matrix plot: `{schema}_confusion_matrix.png`
- F1 scores visualization: `{schema}_f1_scores.png`
- Evaluation summary JSON: `{schema}_evaluation_summary.json`
- TF-IDF vectorizer pickle file: `{schema}_tfidf_vectorizer.pkl`
- Label encoder pickle file: `{schema}_label_encoder.pkl`

## Monitoring

### Workflow Monitoring
- Monitor workflow runs in Databricks Jobs UI
- Email notifications on start/success/failure
- Logs available in each task

### Parameter Customization
To customize the workflow for different environments or users:
1. Update the task parameters in the Databricks Jobs UI
2. Modify `catalog_name`, `schema_name`, and `dbfs_base_path` as needed
3. Ensure the corresponding DBFS directory exists and contains the input data

## Troubleshooting

### Common Issues
- **Data not found**: Ensure `Tweets.csv` is uploaded to the correct DBFS path
- **Permission errors**: Check Unity Catalog permissions for the specified catalog/schema
- **Model registration**: Verify Unity Catalog permissions
- **Component imports**: Ensure src directory is uploaded correctly
- **Cluster issues**: Verify cluster is running and has sufficient resources
- **Path issues**: Check that all file paths match your Databricks workspace structure

### Debugging Steps
1. **Check task logs**: Click on failed task to view detailed logs
2. **Verify file paths**: Ensure all paths match your workspace structure
3. **Test individual notebooks**: Run notebooks manually to isolate issues
4. **Check permissions**: Verify Unity Catalog and DBFS permissions
5. **Validate data**: Ensure input data is properly formatted and accessible

### Getting Help
- Check Databricks documentation for workspace-specific issues
- Review notebook outputs for specific error messages
- Verify all prerequisites are met before running the workflow

print(f"\n✅ Deployment completed successfully!")
print(f"Job ID: {job_id}")
print(f"View workflow at: {workspace_url}/#job/{job_id}")
print(f"\n📝 Next steps:")
print(f"1. Upload your data: databricks fs cp data/Tweets.csv dbfs:/FileStore/shyamkumar.vn/Tweets.csv")
print(f"2. Run the workflow manually or schedule it")
print(f"3. Monitor the training process in the Databricks Jobs UI") 