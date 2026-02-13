from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    text: str = Field(..., example="Breaking: Scientists discover new planet")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    texts: List[str] = Field(..., example=[
        "Breaking: Scientists discover new planet",
        "Local team wins championship"
    ])


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    text: str
    prediction: str  # "REAL" or "FAKE"
    confidence: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str
    model_loaded: bool
    model_version: str
 