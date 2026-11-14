"""Write data/raw/generated_complaints.csv (English, Hindi, Marathi; rows tagged label_source=generated).

These rows are rule-assembled from templates. Report them as team-generated data and keep a real
held-out set for evaluation (see docs/DATA.md).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.generate import generate

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n-en", type=int, default=420)
ap.add_argument("--n-hi", type=int, default=300)
ap.add_argument("--n-mr", type=int, default=280)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", type=Path, default=ROOT / "data/raw/generated_complaints.csv")
a = ap.parse_args()
df = generate(a.n_en, a.n_hi, a.n_mr, a.seed)
a.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"{len(df)} rows -> {a.out}")
print(df.groupby("language").size().to_string())
print(pd_crosstab := df.groupby(["department", "urgency"]).size().unstack(fill_value=0).to_string())
