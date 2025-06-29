# Databricks Model Inference Endpoint Deployment

This directory contains the deployment setup for the Political Party Classification Model Inference Endpoint using Databricks serving endpoints.

## Overview

The Political Party Classification model is trained and registered in Unity Catalog with the following details:
- **Catalog**: `mle_batch_catalog_2025_q2`
- **Schema**: `mle_shyamkumar_vn`
- **Model Name**: `political_party_classifier`
- **Production Alias**: `production`

## Deployment Options

### Option 1: UI Deployment (Recommended for Beginners)

Follow these steps to deploy the model using the Databricks UI:

#### Step 1: Access Machine Learning Workspace
1. Open your Databricks workspace
2. Navigate to **Machine Learning** in the left sidebar
3. Click on **Serving Endpoints**

#### Step 2: Create New Serving Endpoint
1. Click **Create Serving Endpoint**
2. Enter the following details:
   - **Name**: `political-party-classifier-endpoint`
   - **Description**: `Political Party Classification Model Inference Endpoint`

#### Step 3: Configure Model
1. Click **Add Model**
2. Select **Unity Catalog** as the model source
3. Choose your model: `mle_batch_catalog_2025_q2.mle_shyamkumar_vn.political_party_classifier`
4. Select **Latest** version or specific version
5. Set **Workload Size**: `Small` (for development) or `Medium` (for production)
6. Enable **Scale to Zero** for cost optimization

#### Step 4: Set Environment Variables
Add the following environment variables:
```
CATALOG_NAME=mle_batch_catalog_2025_q2
SCHEMA_NAME=mle_shyamkumar_vn
MODEL_NAME=political_party_classifier
PRODUCTION_ALIAS=production
```

#### Step 5: Deploy
1. Click **Create Endpoint**
2. Wait for the endpoint to be ready (status should show "Ready")
3. Note the endpoint URL for API access

#### Step 6: Test the Endpoint
1. Click on your endpoint name
2. Go to the **Query Endpoint** tab
3. Test with sample data:
```json
{
  "dataframe_records": [
    {
      "text": "Great economic policies from our party! #politics #economy"
    }
  ]
}
```

### Option 2: Programmatic Deployment

Use the provided Python scripts for automated deployment:

#### Prerequisites
1. Databricks CLI configured with appropriate permissions
2. Python environment with required packages
3. Access to Unity Catalog

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Configure Environment
Set environment variables or use the configuration file:
```bash
export DATABRICKS_HOST="your-workspace-url"
export DATABRICKS_TOKEN="your-access-token"
export CATALOG_NAME="mle_batch_catalog_2025_q2"
export SCHEMA_NAME="mle_shyamkumar_vn"
export MODEL_NAME="political_party_classifier"
export PRODUCTION_ALIAS="production"
```

#### Step 3: Deploy Endpoint
```bash
python deploy_endpoint.py
```

#### Step 4: Test Endpoint
```bash
python test_endpoint.py
```

## API Usage

### Endpoint Information
- **Base URL**: `https://your-workspace-url/serving-endpoints/political-party-classifier-endpoint/invocations`
- **Authentication**: Bearer token required
- **Content-Type**: `application/json`

### Single Prediction
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "text": "Great economic policies from our party!"
      }
    ]
  }' \
  https://your-workspace-url/serving-endpoints/political-party-classifier-endpoint/invocations
```

### Batch Prediction
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "text": "Great economic policies from our party!"
      },
      {
        "text": "We need better healthcare for all citizens"
      },
      {
        "text": "Supporting local businesses and job creation"
      }
    ]
  }' \
  https://your-workspace-url/serving-endpoints/political-party-classifier-endpoint/invocations
```

### Python Client Example
```python
import requests
import json

# Configuration
endpoint_url = "https://your-workspace-url/serving-endpoints/political-party-classifier-endpoint/invocations"
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}

# Single prediction
data = {
    "dataframe_records": [
        {"text": "Great economic policies from our party!"}
    ]
}

response = requests.post(endpoint_url, headers=headers, json=data)
result = response.json()
print(f"Prediction: {result['predictions'][0]}")

# Batch prediction
batch_data = {
    "dataframe_records": [
        {"text": "Great economic policies from our party!"},
        {"text": "We need better healthcare for all citizens"},
        {"text": "Supporting local businesses and job creation"}
    ]
}

response = requests.post(endpoint_url, headers=headers, json=batch_data)
result = response.json()
print(f"Predictions: {result['predictions']}")
```

## Model Details

### Input Schema
The model expects input in the following format:
```json
{
  "dataframe_records": [
    {
      "text": "string"  // Raw tweet text
    }
  ]
}
```

### Output Schema
The model returns predictions in the following format:
```json
{
  "predictions": ["string"]  // Political party labels
}
```

### Supported Political Parties
The model can classify tweets into the following political parties:
- Democratic Party
- Republican Party
- Independent
- Other

## Monitoring and Management

### Endpoint Monitoring
1. Go to **Machine Learning** > **Serving Endpoints**
2. Click on your endpoint name
3. Monitor:
   - **Traffic**: Number of requests
   - **Latency**: Response times
   - **Errors**: Failed requests
   - **Cost**: Compute usage

### Model Version Management
1. **Production Alias**: Points to the current production model
2. **Staging Alias**: Points to the model being tested
3. **Version History**: Track all model versions

### Scaling
- **Auto-scaling**: Automatically scales based on traffic
- **Scale to Zero**: Stops compute when no traffic (cost optimization)
- **Manual Scaling**: Adjust workload size as needed

## Troubleshooting

### Common Issues

1. **Endpoint Not Ready**
   - Check if the model exists in Unity Catalog
   - Verify permissions to access the model
   - Check cluster availability

2. **Authentication Errors**
   - Verify your access token is valid
   - Check token permissions for serving endpoints
   - Ensure proper Authorization header format

3. **Model Loading Errors**
   - Verify model is registered in Unity Catalog
   - Check model version exists
   - Ensure model signature matches expected input

4. **Prediction Errors**
   - Verify input format matches model signature
   - Check text preprocessing requirements
   - Review model logs for detailed error messages

### Debug Steps

1. **Check Endpoint Status**
   ```bash
   databricks serving-endpoints list
   ```

2. **View Endpoint Logs**
   - Go to endpoint details in UI
   - Check **Logs** tab for error messages

3. **Test Model Locally**
   ```python
   import mlflow
   model = mlflow.pyfunc.load_model("models:/your-model-uri")
   result = model.predict({"text": "test tweet"})
   ```

## Cost Optimization

1. **Scale to Zero**: Enable to stop compute when idle
2. **Workload Size**: Choose appropriate size for your traffic
3. **Auto-scaling**: Let Databricks handle scaling automatically
4. **Monitoring**: Track usage and adjust as needed

## Security Considerations

1. **Access Control**: Use appropriate permissions for endpoint access
2. **Token Management**: Rotate access tokens regularly
3. **Network Security**: Use VPC endpoints if required
4. **Data Privacy**: Ensure no sensitive data in logs

## Next Steps

After successful deployment:

1. **Integration**: Integrate the endpoint with your applications
2. **Monitoring**: Set up monitoring and alerting
3. **Testing**: Perform load testing and validation
4. **Documentation**: Document API usage for your team
5. **CI/CD**: Set up automated deployment pipelines

## Support

For issues or questions:
1. Check Databricks documentation
2. Review endpoint logs
3. Contact your Databricks administrator
4. Open a support ticket if needed 