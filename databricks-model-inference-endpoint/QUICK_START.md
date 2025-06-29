# Quick Start Guide - Model Inference Endpoint Deployment

This guide provides quick instructions for deploying the Political Party Classification model inference endpoint.

## Prerequisites

1. **Databricks Workspace Access**: You need access to a Databricks workspace
2. **Model Trained**: The model should be trained and registered in Unity Catalog
3. **Authentication**: Databricks CLI configured or access token available

## Option 1: UI Deployment (Easiest)

1. Open your Databricks workspace
2. Go to **Machine Learning** > **Serving Endpoints**
3. Click **Create Serving Endpoint**
4. Configure:
   - **Name**: `political-party-classifier-endpoint`
   - **Model**: `mle_batch_catalog_2025_q2.mle_shyamkumar_vn.political_party_classifier`
   - **Version**: Latest
   - **Workload Size**: Small
   - **Scale to Zero**: Enabled
5. Click **Create Endpoint**

## Option 2: Programmatic Deployment

### Using the Shell Script (Recommended)

```bash
# Make script executable
chmod +x deploy.sh

# Deploy with default configuration
./deploy.sh deploy

# Deploy and test
./deploy.sh full

# Test only
./deploy.sh test quick
```

### Using Python Scripts Directly

```bash
# Install dependencies
pip install -r requirements.txt

# Deploy endpoint
python deploy_endpoint.py

# Test endpoint
python test_endpoint.py --test-mode quick
```

### Using Custom Configuration

```bash
# Deploy with custom config
./deploy.sh deploy custom_config.json

# Or with command line arguments
python deploy_endpoint.py \
  --catalog-name "your_catalog" \
  --schema-name "your_schema" \
  --model-name "your_model" \
  --endpoint-name "your_endpoint"
```

## Testing the Endpoint

### Quick Test
```bash
./deploy.sh test quick
```

### Full Test Suite
```bash
./deploy.sh test full
```

### Performance Test
```bash
./deploy.sh test performance
```

## API Usage

Once deployed, you can use the endpoint:

```python
import requests

# Configuration
endpoint_url = "https://your-workspace/serving-endpoints/political-party-classifier-endpoint/invocations"
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
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   export DATABRICKS_TOKEN="your-token"
   ```

2. **Model Not Found**
   - Ensure model is trained and registered
   - Check Unity Catalog permissions

3. **Endpoint Not Ready**
   - Wait for deployment to complete
   - Check endpoint status in UI

### Getting Help

```bash
# Show help
./deploy.sh help

# Verbose output
./deploy.sh deploy --verbose
```

## Files Overview

- `README.md` - Comprehensive documentation
- `deploy.sh` - Easy deployment script
- `deploy_endpoint.py` - Python deployment script
- `test_endpoint.py` - Python testing script
- `config.json` - Configuration template
- `requirements.txt` - Python dependencies

## Next Steps

After successful deployment:

1. **Integration**: Use the endpoint in your applications
2. **Monitoring**: Set up monitoring and alerting
3. **Scaling**: Adjust workload size based on traffic
4. **Updates**: Use the same scripts to update the endpoint 