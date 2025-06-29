# Political Party Classification Model Inference Endpoint

This directory contains the model inference endpoint for the Political Party Classification system. The endpoint provides a REST API for making predictions using the trained model stored in Unity Catalog.

## Features

- **Single Prediction**: Classify individual tweet texts
- **Batch Prediction**: Classify multiple tweet texts at once
- **Health Monitoring**: Health check and model information endpoints
- **Model Reload**: Ability to reload the model without restarting the service
- **Docker Support**: Containerized deployment
- **Databricks Integration**: Designed to work with Databricks serving endpoints

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the endpoint and model loading status.

### Model Information
```
GET /model-info
```
Returns information about the loaded model including catalog, schema, and model details.

### Single Prediction
```
POST /predict
```
**Request Body:**
```json
{
    "text": "Your tweet text here"
}
```

**Response:**
```json
{
    "prediction": "PartyName",
    "confidence": 0.95
}
```

### Batch Prediction
```
POST /predict-batch
```
**Request Body:**
```json
{
    "texts": [
        "First tweet text",
        "Second tweet text",
        "Third tweet text"
    ]
}
```

**Response:**
```json
{
    "predictions": ["Party1", "Party2", "Party3"],
    "confidences": [0.95, 0.87, 0.92]
}
```

### Model Reload
```
POST /reload-model
```
Reloads the model from Unity Catalog (useful for model updates).

## Environment Variables

The following environment variables can be configured:

- `CATALOG_NAME`: Unity Catalog name (default: "mle_batch_catalog_2025_q2")
- `SCHEMA_NAME`: Unity Catalog schema name (default: "mle_shyamkumar_vn")
- `MODEL_NAME`: Model name in Unity Catalog (default: "political_party_classifier")
- `PRODUCTION_ALIAS`: Production model alias (default: "production")

## Local Development

### Prerequisites

1. Python 3.9+
2. MLflow 2.22.0
3. Databricks CLI configured with appropriate permissions

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export CATALOG_NAME="your_catalog"
export SCHEMA_NAME="your_schema"
export MODEL_NAME="your_model"
export PRODUCTION_ALIAS="production"
```

3. Run the endpoint:
```bash
python main.py
```

The endpoint will be available at `http://localhost:8000`

### Testing

Run the test suite:
```bash
python ../test_inference_endpoint.py
```

## Docker Deployment

### Build the Docker Image

```bash
docker build -t political-party-classifier-endpoint .
```

### Run the Container

```bash
docker run -p 8000:8000 \
  -e CATALOG_NAME="your_catalog" \
  -e SCHEMA_NAME="your_schema" \
  -e MODEL_NAME="your_model" \
  -e PRODUCTION_ALIAS="production" \
  political-party-classifier-endpoint
```

## Databricks Deployment

### Using Databricks Serving Endpoints

1. **Via UI:**
   - Go to Machine Learning > Serving Endpoints
   - Click "Create Serving Endpoint"
   - Select your model from Unity Catalog
   - Configure the endpoint settings
   - Deploy

2. **Via API/SDK:**
   ```bash
   python ../deploy_inference_endpoint.py
   ```

### Using Custom Container Service

For custom container deployment:

1. Build and push the Docker image to your container registry
2. Configure the serving endpoint to use the custom container
3. Set the appropriate environment variables

## Usage Examples

### Python Client

```python
import requests
import json

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Great economic policies from our party!"}
)
result = response.json()
print(f"Prediction: {result['prediction']}")

# Batch prediction
texts = [
    "Great economic policies from our party!",
    "We need better healthcare for all citizens",
    "Supporting local businesses and job creation"
]

response = requests.post(
    "http://localhost:8000/predict-batch",
    json={"texts": texts}
)
result = response.json()
print(f"Predictions: {result['predictions']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great economic policies from our party!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'
```

## Monitoring and Logging

The endpoint includes comprehensive logging:

- Application startup and model loading
- Prediction requests and responses
- Error handling and debugging information
- Health check monitoring

Logs are written to stdout/stderr and can be collected by your logging infrastructure.

## Error Handling

The endpoint handles various error scenarios:

- Model loading failures
- Invalid input data
- Prediction errors
- Network connectivity issues

All errors return appropriate HTTP status codes and error messages.

## Security Considerations

- The endpoint runs as a non-root user in Docker
- Input validation is performed on all requests
- Environment variables should be properly secured
- Consider adding authentication/authorization for production use

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   - Check Unity Catalog permissions
   - Verify model exists and is accessible
   - Check environment variables

2. **Prediction Errors**
   - Verify input data format
   - Check model signature compatibility
   - Review logs for detailed error messages

3. **Docker Issues**
   - Ensure Docker has sufficient resources
   - Check port availability
   - Verify image build process

### Debug Mode

To enable debug logging, set the log level:

```bash
export LOG_LEVEL=DEBUG
```

## Performance Considerations

- The endpoint uses model caching for better performance
- Batch predictions are more efficient than multiple single predictions
- Consider scaling based on your workload requirements
- Monitor memory usage, especially for large models

## Contributing

When contributing to this endpoint:

1. Follow the existing code style
2. Add appropriate tests for new features
3. Update documentation as needed
4. Test thoroughly before deployment 