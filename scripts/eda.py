"""Exploratory data analysis. Writes tables (+ PNG charts if matplotlib is installed) to reports/eda/."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data", type=Path, default=None)
ap.add_argument("--out", type=Path, default=ROOT / "reports/eda")
a = ap.parse_args()

path = a.data or ROOT / "data/processed/grievances.csv"
if not path.exists():
    path = ROOT / "data/samples/grievances_sample.csv"
    print("Using demo data (repeated rows): charts are illustrative only.")
df = pd.read_csv(path)
a.out.mkdir(parents=True, exist_ok=True)

tables = {
    "language_counts": df["language"].value_counts(),
    "department_counts": df["department"].value_counts(),
    "urgency_counts": df["urgency"].value_counts(),
    "sentiment_counts": df["sentiment"].value_counts(),
    "language_x_department": pd.crosstab(df["language"], df["department"]),
    "language_x_urgency": pd.crosstab(df["language"], df["urgency"]),
}
df["n_chars"] = df["text"].str.len()
df["n_words"] = df["text"].str.split().str.len()
tables["text_length_by_language"] = df.groupby("language")[["n_chars", "n_words"]].describe().round(1)
tables["duplicate_texts"] = pd.Series({"duplicates": int(df["text"].duplicated().sum()), "rows": len(df)})

for name, t in tables.items():
    t.to_csv(a.out / f"{name}.csv")
    print(f"\n== {name} ==\n{t.to_string()}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for axis, col in zip(ax, ["department", "language", "urgency"]):
        df[col].value_counts().plot.bar(ax=axis, title=col)
    fig.tight_layout()
    fig.savefig(a.out / "label_distributions.png", dpi=120)
    print("\nchart:", a.out / "label_distributions.png")
except ImportError:
    print("matplotlib not installed; skipped charts")
