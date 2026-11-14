from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ..models.predictor import Predictor

app = FastAPI(title="Grievance Classifier & Router", version="0.1.0")
predictor = Predictor()

class Complaint(BaseModel):
    text: str = Field(min_length=1)

class Batch(BaseModel):
    complaints: List[str]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor.model is not None}

@app.post("/predict")
def predict(c: Complaint):
    return predictor.predict(c.text)

@app.post("/predict/batch")
def predict_batch(b: Batch):
    return [predictor.predict(t) for t in b.complaints]
