import os
from pathlib import Path
import pickle
from typing import Any

from grievance.data.urgency_rules import weak_sentiment, weak_urgency
from grievance.labels import ROUTING


class Predictor:
    """Load the trained baseline and combine it with weak-label task rules."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        default_path = (
            Path(__file__).resolve().parents[3]
            / "reports"
            / "baseline"
            / "model.pkl"
        )
        path = Path(model_path) if model_path is not None else default_path
        self.model: Any = None
        if path.exists():
            with path.open("rb") as file:
                self.model = pickle.load(file)

    def predict(self, text: str) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Complaint text must not be empty.")
        if self.model is None:
            raise RuntimeError(
                "No trained model is available. Run scripts/train_baseline.py first."
            )

        department = str(self.model.predict([text])[0])
        return {
            "department": department,
            "urgency": weak_urgency(text),
            "sentiment": weak_sentiment(text),
            "route": ROUTING[department],
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, str]]:
        return [self.predict(t) for t in texts]

    def info(self) -> dict[str, Any]:
        return {"backend": "tfidf-logreg-baseline", "loaded": self.model is not None}


def load_predictor():
    """Return the best available predictor.

    If GRIEVANCE_MODEL_DIR points to a directory written by
    scripts/train_transformer.py, use the fine-tuned multitask transformer;
    otherwise fall back to the TF-IDF baseline.
    """
    model_dir = os.environ.get("GRIEVANCE_MODEL_DIR")
    if model_dir:
        from grievance.models.transformer_predictor import TransformerPredictor

        return TransformerPredictor(model_dir)
    return Predictor()
