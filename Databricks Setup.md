# Databricks Batch Training Workflow

This directory contains the Databricks integration for the Political Party Tweet Classification project.

## Overview

The batch training workflow consists of five main tasks that reuse our existing components:
1. **Data Preparation and Feature Engineering** - Load data and extract features using DataLoader's preprocessing methods with train/test/validation splits (70/20/10)
2. **Model Training** - Train the classification model using our train_model function and evaluate on validation set
3. **Model Promotion** - Compare new model with production model and promote if better (using validation performance)
4. **Email Notification** - Send email notification when model is promoted to production
5. **Model Evaluation** - Evaluate production model performance and generate reports

## Component Reuse

The Databricks notebooks are designed to reuse the existing components from the `src/` directory:

- **DataLoader** (`src/text_loader/loader.py`) - Used for data loading and feature preprocessing (includes text cleaning)
- **train_model function** (`src/train_model.py`) - Used for model training and evaluation
- **Utils** (`src/utils.py`) - Common utilities for widget handling, model URI generation, and MLflow setup

This ensures consistency between local development and Databricks execution.

## Directory Structure

```
databricks/
├── notebooks/
│   ├── data_preparation.py      # Combined data loading and feature extraction with splits
│   ├── train_model.py          # Model training using train_model function with validation
│   ├── model_promotion.py      # Model comparison and promotion logic
│   ├── email_notification.py   # Email notification when model is promoted
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

### 3. Email Configuration
- Configure email settings for notifications (SMTP server, credentials)
- Set up email recipients for model promotion notifications

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
   │   │   ├── model_promotion.py
   │   │   ├── email_notification.py
   │   │   └── model_evaluation.py
   │   └── workflows/
   │       └── training_workflow.json
   └── README.md
   ```

2. **Ensure you have the required files**:
   - `data/Tweets.csv` - The input dataset
   - `src/text_loader/loader.py` - DataLoader component
   - `src/train_model.py` - Training function
   - `src/utils.py` - Utils component
   - `databricks/notebooks/*.py` - Databricks notebooks

### Step 3: Upload to Databricks Workspace

#### 3.1 Upload Source Code
1. In Databricks workspace, go to **Workspace** → **Users** → **shyamkumar.vn@thoughtworks.com**
2. Navigate to the `MLEng-politicalparties-python-exercise-fork` folder
3. Upload the `src/` directory:
   - Create folder: `src/text_loader/`
   - Upload `src/text_loader/loader.py` to `src/text_loader/`
   - Upload `src/train_model.py` to `src/`
   - Upload `src/utils.py` if it exists
   - Upload `src/text_loader/__init__.py` if it exists

#### 3.2 Upload Notebooks
1. Create folder: `databricks/notebooks/` in your workspace
2. Upload each notebook:
   - `databricks/notebooks/data_preparation.py`
   - `databricks/notebooks/train_model.py`
   - `databricks/notebooks/model_promotion.py`
   - `databricks/notebooks/email_notification.py`
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

**Task 3: Model Promotion**
1. Click **Add task** → **Notebook**
2. **Task name**: `model_promotion`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/model_promotion`
4. **Cluster**: Select your cluster
5. **Dependencies**: Add dependency on `model_training`
6. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `model_name`: `political_party_classifier`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`

**Task 4: Email Notification (Conditional)**
1. Click **Add task** → **Notebook**
2. **Task name**: `email_notification`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/email_notification`
4. **Cluster**: Select your cluster
5. **Dependencies**: Add dependency on `model_promotion`
6. **Condition**: Set to run only if `model_promotion` succeeds
7. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `model_name`: `political_party_classifier`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`
   - `notification_email`: `shyamkumar.vn@thoughtworks.com`
   - `smtp_server`: `smtp.gmail.com` (or your SMTP server)
   - `smtp_port`: `587`

**Task 5: Model Evaluation**
1. Click **Add task** → **Notebook**
2. **Task name**: `model_evaluation`
3. **Notebook path**: `/Workspace/Users/shyamkumar.vn@thoughtworks.com/MLEng-politicalparties-python-exercise-fork/databricks/notebooks/model_evaluation`
4. **Cluster**: Select your cluster
5. **Dependencies**: Add dependency on `email_notification`
6. **Parameters**:
   - `catalog_name`: `mle_batch_catalog_2025_q2`
   - `schema_name`: `mle_shyamkumar_vn`
   - `model_name`: `political_party_classifier`
   - `dbfs_base_path`: `/dbfs/FileStore/shyamkumar.vn`

#### 5.3 Configure Job Settings
1. **Email notifications**: Add your email for start/success/failure notifications
2. **Timeout**: Set appropriate timeouts (3600s for data prep, 7200s for training, 1800s for evaluation)
3. **Retry policy**: Configure as needed

### Step 6: Configure Email Settings

#### 6.1 Email Configuration for Notifications
The email notification task requires SMTP configuration. You can set this up in several ways:

**Option A: Using Databricks Secrets (Recommended)**
1. Go to **Settings** → **Secrets**
2. Create a new secret scope: `email-notifications`
3. Add secrets:
   - `smtp-username`: Your email username
   - `smtp-password`: Your email password or app password
   - `smtp-server`: SMTP server (e.g., `smtp.gmail.com`)
   - `smtp-port`: SMTP port (e.g., `587`)

**Option B: Using Environment Variables**
Set these in your cluster environment variables:
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_SERVER`
- `SMTP_PORT`

**Option C: Using Notebook Parameters**
Pass email credentials as notebook parameters (less secure, not recommended for production)

#### 6.2 Email Recipients
Configure the following email recipients:
- **Primary**: `shyamkumar.vn@thoughtworks.com`
- **Backup**: Add additional team members as needed
- **Stakeholders**: Add relevant stakeholders for model promotion notifications

### Step 7: Run the Workflow

#### 7.1 Manual Run
1. Go to your job in **Workflows** → **Jobs**
2. Click **Run now**
3. Monitor the execution in the job runs tab

#### 7.2 Scheduled Run
1. In your job settings, configure a schedule:
   - **Type**: Cron schedule
   - **Cron expression**: `0 0 * * 0` (weekly on Sunday at midnight)
   - **Timezone**: Your local timezone

### Step 8: Monitor and Troubleshoot

#### 8.1 Monitor Job Execution
1. **Job Runs**: Check the job runs tab for execution status
2. **Logs**: Review logs for each task
3. **Metrics**: Monitor MLflow experiments for model performance

#### 8.2 Email Notification Monitoring
1. **Check Email Delivery**: Verify emails are being sent successfully
2. **Review Email Content**: Ensure email content is accurate and informative
3. **Troubleshoot SMTP Issues**: Check SMTP configuration if emails fail

#### 8.3 Common Issues and Solutions

**Issue: Email notifications not sending**
- **Solution**: Check SMTP configuration and credentials
- **Solution**: Verify network connectivity to SMTP server
- **Solution**: Check Databricks secrets configuration

**Issue: Model promotion not triggering emails**
- **Solution**: Verify model promotion logic is working correctly
- **Solution**: Check email notification task dependencies
- **Solution**: Review email notification notebook logs

**Issue: Workflow failing at email task**
- **Solution**: Check SMTP server availability
- **Solution**: Verify email credentials are correct
- **Solution**: Review email notification notebook for errors

## Email Notification Details

### When Emails Are Sent
- **Model Promotion**: Email sent when a new model is promoted to production
- **Promotion Failure**: Email sent if model promotion fails
- **Workflow Completion**: Summary email when entire workflow completes

### Email Content
The email notification includes:
- **Model Information**: Model name, version, and performance metrics
- **Promotion Details**: Old vs new model comparison
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Next Steps**: Actions required after promotion
- **Contact Information**: Who to contact for questions

### Email Configuration
```python
# Example email configuration in the notification notebook
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'ml-notifications@yourcompany.com',
    'recipients': ['shyamkumar.vn@thoughtworks.com'],
    'subject_prefix': '[ML Model Promotion]'
}
```

## Security Considerations

### Email Security
- Use app passwords instead of regular passwords for SMTP
- Store credentials in Databricks secrets, not in code
- Use TLS/SSL for email transmission
- Limit email access to necessary personnel only

### Model Security
- Ensure only authorized users can promote models
- Log all model promotion activities
- Implement approval workflows for critical model changes
- Regular security audits of the ML pipeline

## Troubleshooting Guide

### Email Issues
1. **SMTP Connection Failed**
   - Check SMTP server and port configuration
   - Verify network connectivity
   - Check firewall settings

2. **Authentication Failed**
   - Verify username and password
   - Check if app password is required
   - Ensure account is not locked

3. **Email Not Delivered**
   - Check recipient email address
   - Verify sender email is not blocked
   - Check spam/junk folders

### Workflow Issues
1. **Task Dependencies**
   - Verify task dependencies are correctly set
   - Check if previous tasks completed successfully
   - Review task execution order

2. **Resource Issues**
   - Check cluster availability
   - Monitor cluster resources (CPU, memory)
   - Verify timeout settings are appropriate

3. **Data Issues**
   - Check data file availability
   - Verify data format and quality
   - Ensure sufficient storage space

## Best Practices

### Email Notifications
- Keep email content concise and actionable
- Include relevant metrics and context
- Use clear subject lines for easy identification
- Test email functionality before production deployment

### Model Promotion
- Always validate models before promotion
- Maintain clear promotion criteria
- Document promotion decisions
- Have rollback procedures ready

### Monitoring
- Set up alerts for workflow failures
- Monitor model performance after promotion
- Track email delivery success rates
- Regular review of notification recipients

## Support and Maintenance

### Regular Maintenance
- Update email recipient lists as needed
- Review and update SMTP configuration
- Monitor email delivery success rates
- Update notification content and format

### Support Contacts
- **Technical Issues**: Contact your Databricks administrator
- **Email Configuration**: Contact your IT team
- **Model Performance**: Contact the ML team
- **Workflow Issues**: Contact the DevOps team

This comprehensive setup ensures that your ML pipeline includes proper notification mechanisms for model promotions, keeping stakeholders informed of important changes in your production models.

print(f"\n✅ Deployment completed successfully!")
print(f"Job ID: {job_id}")
print(f"View workflow at: {workspace_url}/#job/{job_id}")
print(f"\n📝 Next steps:")
print(f"1. Upload your data: databricks fs cp data/Tweets.csv dbfs:/FileStore/shyamkumar.vn/Tweets.csv")
print(f"2. Run the workflow manually or schedule it")
print(f"3. Monitor the training process in the Databricks Jobs UI") 