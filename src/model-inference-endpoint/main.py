from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Model configuration - these would typically come from environment variables
CATALOG_NAME = os.getenv("CATALOG_NAME", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = os.getenv("SCHEMA_NAME", "mle_shyamkumar_vn")
MODEL_NAME = os.getenv("MODEL_NAME", "political_party_classifier")
PRODUCTION_ALIAS = os.getenv("PRODUCTION_ALIAS", "production")

# Global variable to store the loaded model
loaded_model = None

class InputText(BaseModel):
    text: str

class BatchInputText(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = None

class BatchPredictionResponse(BaseModel):
    predictions: List[str]
    confidences: List[float] = None

app = FastAPI(
    title="Political Party Classification API",
    description="API for classifying political parties from tweet text",
    version="1.0.0"
)

def load_model():
    """Load the production model from Unity Catalog"""
    global loaded_model
    
    try:
        # Get the production model using alias
        model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}@{PRODUCTION_ALIAS}"
        logger.info(f"Loading model from: {model_uri}")
        
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fallback: try to load latest version
        try:
            model_uri = f"models:/{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
            logger.info(f"Trying fallback model URI: {model_uri}")
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Fallback model loaded successfully")
        except Exception as fallback_error:
            logger.error(f"Fallback model loading failed: {fallback_error}")
            raise HTTPException(status_code=500, detail="Model loading failed")

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    logger.info("Starting up the inference endpoint...")
    load_model()
    logger.info("Inference endpoint ready!")

@app.get("/health")
def get_health():
    """Health check endpoint"""
    return {
        "status": "OK",
        "model_loaded": loaded_model is not None,
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "model": MODEL_NAME
    }

@app.get("/model-info")
def get_model_info():
    """Get information about the loaded model"""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(loaded_model).__name__,
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "model_name": MODEL_NAME,
        "production_alias": PRODUCTION_ALIAS
    }

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(input_data: InputText):
    """Get prediction for a single text input"""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_df = pd.DataFrame({'text': [input_data.text]})
        
        # Make prediction
        predictions = loaded_model.predict(input_df)
        
        # Extract prediction
        prediction = predictions[0] if len(predictions) > 0 else "Unknown"
        
        logger.info(f"Prediction for text '{input_data.text[:50]}...': {prediction}")
        
        return PredictionResponse(prediction=prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
def get_batch_prediction(input_data: BatchInputText):
    """Get predictions for multiple text inputs"""
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_df = pd.DataFrame({'text': input_data.texts})
        
        # Make predictions
        predictions = loaded_model.predict(input_df)
        
        # Convert to list
        prediction_list = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        logger.info(f"Batch prediction for {len(input_data.texts)} texts completed")
        
        return BatchPredictionResponse(predictions=prediction_list)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/reload-model")
def reload_model():
    """Reload the model (useful for model updates)"""
    try:
        load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)