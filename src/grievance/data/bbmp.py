"""Import the BBMP (Bengaluru) citizen-grievance export as a *seed* dataset.

The export has ~1.2M rows but no free-text complaint: `Sub Category` is a short
label (e.g. "Street Light Not Working") with only ~190 distinct values. So this
yields a small set of distinct phrases + a Category -> department mapping, not a
large training corpus. Staff columns (names/designations) are never read.
"""
from __future__ import annotations

import pandas as pd

# BBMP category -> our department. Anything not listed becomes "other".
CATEGORY_TO_DEPARTMENT = {
    "electrical": "power",
    "solid waste (garbage) related": "sanitation",
    "sanitation": "sanitation",
    "road maintenance(engg)": "roads",
    "road infrastructure": "roads",
    "storm  water drain(swd)": "water",
    "storm water drain(swd)": "water",
    "water crisis": "water",
}

_USECOLS = ["Category", "Sub Category"]  # deliberately excludes staff name/designation columns


def map_department(category: str) -> str:
    key = " ".join(str(category).lower().split())
    key = CATEGORY_TO_DEPARTMENT.get(key) or CATEGORY_TO_DEPARTMENT.get(str(category).strip().lower())
    return key or "other"


def load_bbmp_seed(path, min_chars: int = 4) -> pd.DataFrame:
    """Return distinct phrases: text, language, department, count. Majority category wins
    when the same phrase appears under several categories."""
    df = pd.read_csv(path, usecols=_USECOLS, low_memory=False).dropna(subset=["Sub Category"])
    df["text"] = df["Sub Category"].astype(str).str.strip()
    df = df[df["text"].str.len() >= min_chars]
    df["department"] = df["Category"].map(map_department)
    counts = df.groupby(["text", "department"]).size().rename("count").reset_index()
    best = counts.sort_values("count", ascending=False).drop_duplicates("text")
    totals = counts.groupby("text")["count"].sum().rename("count")
    out = best.drop(columns="count").merge(totals, on="text")
    out["language"] = "en"
    return out[["text", "language", "department", "count"]].sort_values("count", ascending=False).reset_index(drop=True)
