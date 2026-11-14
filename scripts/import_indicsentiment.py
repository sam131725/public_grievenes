"""IndicSentiment json -> data/raw/indicsentiment/sentiment.csv (text, language, sentiment, split).

Kept inside its own folder on purpose: prepare_data.py only reads loose CSVs in data/raw/,
and these are product reviews with no department label.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.indicsentiment import load_indicsentiment

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--src", type=Path, default=ROOT / "data/raw/indicsentiment/data")
ap.add_argument("--out", type=Path, default=ROOT / "data/raw/indicsentiment/sentiment.csv")
a = ap.parse_args()
df = load_indicsentiment(a.src)
df.to_csv(a.out, index=False)
print(f"{len(df)} rows -> {a.out}")
print(df.groupby(["split", "sentiment"]).size().to_string())
print("languages:", sorted(df["language"].unique()))
