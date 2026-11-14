"""REST API: classify complaints -> department, urgency, sentiment, routing target.

Run:  uvicorn grievance.api.main:app --app-dir src --reload
Set GRIEVANCE_MODEL_DIR to serve the fine-tuned transformer instead of the baseline.
"""
from typing import Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..models.predictor import load_predictor

MAX_BATCH = 500

app = FastAPI(
    title="Grievance Classifier & Router",
    version="0.2.0",
    description="Classifies citizen complaints and routes them to a department.",
)
predictor = load_predictor()


def _non_empty(v: str) -> str:
    if not v.strip():
        raise ValueError("text must not be empty")
    return v


class Complaint(BaseModel):
    text: str = Field(min_length=1, examples=["No water supply in our street for three days"])

    _check = field_validator("text")(_non_empty)


class Batch(BaseModel):
    complaints: List[str] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("complaints")
    @classmethod
    def _each(cls, v: List[str]) -> List[str]:
        for t in v:
            _non_empty(t)
        return v


def _guard(fn, *args) -> Any:
    try:
        return fn(*args)
    except RuntimeError as e:  # no trained model available
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.get("/model")
def model_info():
    return predictor.info()


@app.post("/predict")
def predict(c: Complaint):
    return _guard(predictor.predict, c.text)


@app.post("/predict/batch")
def predict_batch(b: Batch):
    return _guard(predictor.predict_batch, b.complaints)
