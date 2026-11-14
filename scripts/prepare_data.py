"""Build data/processed/grievances.csv from raw CSVs in data/raw/.

Examples:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --raw-dir data/raw --infer-department
  python scripts/prepare_data.py --raw data/raw/cpgrams.csv data/raw/municipal.csv --text-col complaint
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.build import build_dataset, load_raw, write_outputs

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
ap.add_argument("--raw", type=Path, nargs="*", help="explicit CSV files (overrides --raw-dir)")
ap.add_argument("--text-col", help="name of the text column if not auto-detected")
ap.add_argument("--out", type=Path, default=ROOT / "data/processed/grievances.csv")
ap.add_argument("--infer-department", action="store_true",
                help="keyword-guess department for rows that have none (marks nothing as gold)")
a = ap.parse_args()

paths = a.raw if a.raw else sorted(a.raw_dir.glob("*.csv"))
if not paths:
    sys.exit(f"No CSV files in {a.raw_dir}. Put raw data there (see docs/DATA.md).")
df, report = build_dataset(load_raw(paths, a.text_col), infer_missing_department=a.infer_department)
card = write_outputs(df, report, a.out)
print(f"{report['rows_in']} raw rows -> {report['rows_out']} processed rows -> {a.out}")
for w in report["warnings"]:
    print("WARNING:", w)
print("data card:", card)
