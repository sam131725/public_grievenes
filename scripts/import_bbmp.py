"""Turn data/raw/bbmp/citizen-grievances.csv into a small seed CSV (data/raw/bbmp_seed.csv).

Run before prepare_data.py if you want these phrases included. Note: the source has
no free-text complaints, so this adds ~190 distinct short phrases only.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.bbmp import load_bbmp_seed

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--src", type=Path, default=ROOT / "data/raw/bbmp/citizen-grievances.csv")
ap.add_argument("--out", type=Path, default=ROOT / "data/raw/bbmp_seed.csv")
a = ap.parse_args()
seed = load_bbmp_seed(a.src)
seed.to_csv(a.out, index=False)
print(f"{len(seed)} distinct phrases -> {a.out}")
print(seed.groupby("department")["text"].count().to_string())
