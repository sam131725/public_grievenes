"""Inference for the fine-tuned multitask transformer (needs torch + transformers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from grievance.labels import DEPARTMENTS, ROUTING, SENTIMENT, URGENCY
from grievance.models.multitask import MultiTaskClassifier

_TASKS = {"department": DEPARTMENTS, "urgency": URGENCY, "sentiment": SENTIMENT}


class TransformerPredictor:
    """Loads a directory produced by scripts/train_transformer.py:
    config.json, model.pt, tokenizer files and the encoder config."""

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.dir = Path(model_dir)
        cfg_path = self.dir / "config.json"
        weights = self.dir / "model.pt"
        if not cfg_path.exists() or not weights.exists():
            raise FileNotFoundError(f"{self.dir} must contain config.json and model.pt")
        self.cfg = json.loads(cfg_path.read_text())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        model = MultiTaskClassifier(str(self.dir), load_pretrained=False)
        model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        if any(not t.strip() for t in texts):
            raise ValueError("Complaint text must not be empty.")
        results: list[dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk, padding=True, truncation=True,
                max_length=self.cfg.get("max_len", 128), return_tensors="pt",
            ).to(self.device)
            out = self.model(enc["input_ids"], enc["attention_mask"])
            probs = {k: torch.softmax(v, dim=-1).cpu() for k, v in out.items()}
            for j in range(len(chunk)):
                row: dict[str, Any] = {}
                for task, labels in _TASKS.items():
                    conf, idx = probs[task][j].max(dim=-1)
                    row[task] = labels[int(idx)]
                    row[f"{task}_confidence"] = round(float(conf), 4)
                row["route"] = ROUTING[row["department"]]
                results.append(row)
        return results

    def predict(self, text: str) -> dict[str, Any]:
        return self.predict_batch([text])[0]

    def info(self) -> dict[str, Any]:
        return {"backend": "multitask-transformer", "base_model": self.cfg.get("model_name"),
                "max_len": self.cfg.get("max_len"), "device": self.device, "loaded": True}
