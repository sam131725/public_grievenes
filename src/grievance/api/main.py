
"""
REST API for classifying citizen complaints.

Endpoints:
    GET  /health         - Check API and model health
    GET  /model          - Get model information
    POST /predict        - Classify a single complaint
    POST /predict/batch  - Classify multiple complaints

Run:
    uvicorn grievance.api.main:app --app-dir src --reload

Set GRIEVANCE_MODEL_DIR to serve the fine-tuned transformer
instead of the baseline model.
"""

from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..models.predictor import load_predictor


# Configuration
MAX_BATCH = 500


# Initialize FastAPI application
app = FastAPI(
    title="Grievance Classifier & Router",
    version="0.2.0",
    description="Classifies citizen complaints and routes them to a department.",
)

predictor = load_predictor()


# Validation helpers
def _validate_non_empty_text(value: str) -> str:
    """Ensure complaint text contains non-whitespace characters."""
    if not value.strip():
        raise ValueError("text must not be empty")

    return value


# Request schemas
class Complaint(BaseModel):
    """Schema for a single complaint."""

    text: str = Field(
        min_length=1,
        examples=["No water supply in our street for three days"],
    )

    _check = field_validator("text")(_validate_non_empty_text)


class Batch(BaseModel):
    """Schema for batch complaint classification."""

    complaints: List[str] = Field(
        min_length=1,
        max_length=MAX_BATCH,
    )

    @field_validator("complaints")
    @classmethod
    def _validate_complaints(cls, values: List[str]) -> List[str]:
        """Validate each complaint in the batch."""
        for text in values:
            _validate_non_empty_text(text)

        return values


# Error handling helper
def _guard(fn, *args) -> Any:
    """Handle prediction errors and convert them into HTTP responses."""
    try:
        return fn(*args)

    except RuntimeError as error:
        # No trained model available
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )


# API endpoints
@app.get("/health")
def health():
    """Return API health and model availability."""
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
    }


@app.get("/model")
def model_info():
    """Return information about the loaded model."""
    return predictor.info()


@app.post("/predict")
def predict(complaint: Complaint):
    """Classify a single complaint."""
    return _guard(
        predictor.predict,
        complaint.text,
    )


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    """Classify multiple complaints in a single request."""
    return _guard(
        predictor.predict_batch,
        batch.complaints,
    )
