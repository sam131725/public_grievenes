"""Turn raw complaint CSVs into the processed dataset used by every model.

Processed schema (data/processed/grievances.csv):
    text, language, department, urgency, sentiment, label_source
`label_source` is "gold" when urgency/sentiment came from the raw file and
"weak" when they were filled by keyword rules (see urgency_rules.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..labels import DEPARTMENTS, SENTIMENT, URGENCY
from .preprocess import clean_text
from .urgency_rules import weak_sentiment, weak_urgency

TEXT_ALIASES = ["text", "complaint", "description", "grievance", "body"]
MIN_CHARS = 10

_DEPT_KEYWORDS = {
    "water": ["water", "pipeline", "tap ", "sewage", "sewer", "पानी", "जल", "नल", "पाणी"],
    "roads": ["road", "pothole", "footpath", "bridge", "highway", "सड़क", "गड्ढ", "रस्ता", "रस्त्य", "खड्ड"],
    "power": ["electric", "power", "transformer", "street light", "streetlight", "voltage", "बिजली", "विद्युत", "वीज"],
    "sanitation": ["garbage", "waste", "sweep", "toilet", "sanitation", "dustbin", "कचरा", "सफाई", "स्वच्छ", "घाण"],
}

# (unicode range start, end, language code)
_SCRIPTS = [
    (0x0980, 0x09FF, "bn"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
]
_DEVANAGARI = (0x0900, 0x097F)
_MARATHI_HINTS = ["आहे", "नाही", "आमच्या", "येत नाही", "रस्त्य"]


def guess_language(text: str) -> str:
    """Cheap, dependency-free language guess based on script (+ a Marathi hint list)."""
    counts: dict[str, int] = {}
    deva = 0
    for ch in text:
        cp = ord(ch)
        if _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            deva += 1
            continue
        for lo, hi, code in _SCRIPTS:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if deva and deva >= max(counts.values(), default=0):
        return "mr" if any(h in text for h in _MARATHI_HINTS) else "hi"
    if counts:
        return max(counts, key=counts.get)
    return "en"


def infer_department(text: str) -> str:
    """Keyword department guess. Only used for bootstrapping; always flag as weak."""
    t = text.lower()
    scores = {d: sum(k in t for k in kws) for d, kws in _DEPT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def _find_text_col(df: pd.DataFrame, text_col: str | None) -> str:
    if text_col:
        if text_col not in df.columns:
            raise ValueError(f"--text-col '{text_col}' not in columns {list(df.columns)}")
        return text_col
    for c in TEXT_ALIASES:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Expected one of {TEXT_ALIASES}; got {list(df.columns)}")


def load_raw(paths: list[Path], text_col: str | None = None) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        col = _find_text_col(df, text_col)
        df = df.rename(columns={col: "text"})
        df["source_file"] = p.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No raw CSV files found.")
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    raw: pd.DataFrame,
    infer_missing_department: bool = False,
    min_chars: int = MIN_CHARS,
) -> tuple[pd.DataFrame, dict]:
    """Clean, label-fill, validate and dedupe. Returns (dataset, report)."""
    report: dict = {"rows_in": int(len(raw))}
    df = raw.copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() >= min_chars]
    report["dropped_too_short"] = report["rows_in"] - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    report["dropped_duplicates"] = before - len(df)

    if "language" not in df.columns:
        df["language"] = df["text"].map(guess_language)
    else:
        df["language"] = df["language"].fillna("").astype(str).str.strip().str.lower()
        blank = df["language"] == ""
        df.loc[blank, "language"] = df.loc[blank, "text"].map(guess_language)

    if "department" not in df.columns:
        df["department"] = None
    df["department"] = df["department"].astype("object").map(
        lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else None
    )
    missing_dept = df["department"].isna()
    if infer_missing_department:
        df.loc[missing_dept, "department"] = df.loc[missing_dept, "text"].map(infer_department)
        report["departments_inferred"] = int(missing_dept.sum())
    else:
        df = df[~missing_dept]
        report["dropped_missing_department"] = int(missing_dept.sum())

    unknown = ~df["department"].isin(DEPARTMENTS)
    df.loc[unknown, "department"] = "other"
    report["unknown_department_mapped_to_other"] = int(unknown.sum())

    gold = pd.Series(True, index=df.index)
    for col, labels, fn in (("urgency", URGENCY, weak_urgency), ("sentiment", SENTIMENT, weak_sentiment)):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("object").map(
            lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else None
        )
        bad = df[col].isna() | ~df[col].isin(labels)
        df.loc[bad, col] = df.loc[bad, "text"].map(fn)
        gold &= ~bad
        report[f"{col}_weak_labelled"] = int(bad.sum())
    df["label_source"] = gold.map({True: "gold", False: "weak"})

    out = df[["text", "language", "department", "urgency", "sentiment", "label_source"]].reset_index(drop=True)
    report["rows_out"] = int(len(out))
    report["by_language"] = out["language"].value_counts().to_dict()
    report["by_department"] = out["department"].value_counts().to_dict()
    report["warnings"] = _warnings(out)
    return out, report


def _warnings(df: pd.DataFrame, min_per_class: int = 10) -> list[str]:
    msgs = []
    for d in DEPARTMENTS:
        n = int((df["department"] == d).sum())
        if n < min_per_class:
            msgs.append(f"department '{d}' has only {n} rows (<{min_per_class}); stratified split may fail")
    small_langs = [l for l, n in df["language"].value_counts().items() if n < min_per_class]
    if small_langs:
        msgs.append(f"languages with <{min_per_class} rows (per-language scores unreliable): {small_langs}")
    weak = float((df["label_source"] == "weak").mean()) if len(df) else 0.0
    if weak > 0:
        msgs.append(f"{weak:.0%} of rows have weak urgency/sentiment labels — validate on a hand-labelled subset")
    return msgs


def write_outputs(df: pd.DataFrame, report: dict, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    card = out_csv.with_name("data_card.json")
    card.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return card
