
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import yaml
from typing import List
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictionRequest, 
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse
)


# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="MLOps-ready API for detecting fake news articles",
    version=config['model']['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
preprocessor = None
feature_extractor = None
model = None
model_loaded = False


@app.on_event("startup")
async def load_model():
    """Load model artifacts on startup."""
    global preprocessor, feature_extractor, model, model_loaded
    
    try:
        print("Loading model artifacts...")
        preprocessor = joblib.load('models/preprocessor.pkl')
        feature_extractor = joblib.load('models/feature_extractor.pkl')
        model = joblib.load('models/model.pkl')
        model_loaded = True
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        model_loaded = False


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Fake News Detection API",
        "version": config['model']['version'],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=config['model']['version']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict if a news article is real or fake.
    
    Args:
        request: PredictionRequest with text to classify
    
    Returns:
        PredictionResponse with prediction and confidence
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess text
        cleaned_text = preprocessor.clean_text(request.text)
        
        # Extract features
        features = feature_extractor.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        # Map prediction to label
        prediction_label = "REAL" if prediction == 1 else "FAKE"
        
        return PredictionResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            prediction=prediction_label,
            confidence=confidence,
            model_version=config['model']['version']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict multiple news articles at once.
    
    Args:
        request: BatchPredictionRequest with list of texts
    
    Returns:
        BatchPredictionResponse with list of predictions
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for text in request.texts:
            # Preprocess text
            cleaned_text = preprocessor.clean_text(text)
            
            # Extract features
            features = feature_extractor.transform([cleaned_text])
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            
            # Map prediction to label
            prediction_label = "REAL" if prediction == 1 else "FAKE"
            
            predictions.append(
                PredictionResponse(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    prediction=prediction_label,
                    confidence=confidence,
                    model_version=config['model']['version']
                )
            )
        
        return BatchPredictionResponse(predictions=predictions)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config['api']['host'], 
        port=config['api']['port']
    )
 