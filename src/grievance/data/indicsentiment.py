"""Load AI4Bharat IndicSentiment (product/service reviews, 13 Indic languages).

Not civic complaints: use it only to train/evaluate the *sentiment* head across
languages. It has validation + test splits only, and labels are positive/negative
(no neutral). Rows without a label are dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LANGS = {"as": "Assamese", "bd": "Bodo", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
         "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu"}


def load_indicsentiment(data_dir) -> pd.DataFrame:
    """`data_dir` is the folder containing validation/ and test/ (…/indicsentiment/data)."""
    data_dir = Path(data_dir)
    rows = []
    for split in ("validation", "test"):
        for f in sorted((data_dir / split).glob("*.json")):
            lang = f.stem
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                label, text = d.get("LABEL"), d.get("INDIC REVIEW")
                if not label or not text or not str(text).strip():
                    continue
                rows.append({"text": str(text).strip(), "language": lang,
                             "sentiment": str(label).strip().lower(), "split": split})
    if not rows:
        raise FileNotFoundError(f"No IndicSentiment json files under {data_dir}")
    return pd.DataFrame(rows)
