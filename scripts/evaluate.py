"""Evaluate a saved baseline and export department metrics and error analysis."""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grievance.data.preprocess import load_and_split
from grievance.models.evaluate import (
    department_confusion_matrix,
    misclassified_complaints,
    per_language_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--data", type=Path,
        help="Full dataset to split using the training split procedure.",
    )
    data.add_argument(
        "--test-data", type=Path,
        help="Already held-out CSV; evaluate every row without splitting.",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "reports/baseline/model.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Split seed; must match training (default: 42).",
    )
    args = parser.parse_args()

    if args.test_data is not None:
        test = pd.read_csv(args.test_data)
    else:
        path = args.data
        if path is None:
            path = ROOT / "data/processed/grievances.csv"
            if not path.exists():
                path = ROOT / "data/samples/grievances_sample.csv"
                print(
                    "Using repeated demo data: results are a smoke test, "
                    "not evidence of generalization."
                )
        _, _, test = load_and_split(path, seed=args.seed)

    required = {"text", "language", "department"}
    missing = required - set(test.columns)
    if missing:
        parser.error(f"Missing required columns: {', '.join(sorted(missing))}")
    if test.empty:
        parser.error("The evaluation dataset has no rows.")
    for column in sorted(required):
        valid = test[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            parser.error(
                f"Column '{column}' must contain nonempty strings in every evaluation row."
            )

    with args.model.open("rb") as file:
        model = pickle.load(file)
    test = test.assign(pred=model.predict(test["text"]))
    matrix = department_confusion_matrix(test, "department", "pred")
    errors = misclassified_complaints(test, "department", "pred")
    languages = per_language_report(test, "department", "pred")

    print(classification_report(test["department"], test["pred"], zero_division=0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages.to_csv(args.output_dir / "per_language_report.csv", index=False)
    matrix.to_csv(args.output_dir / "confusion_matrix.csv")
    errors.to_csv(args.output_dir / "misclassified_complaints.csv", index=False)
    print(
        f"Evaluated {len(test)} complaints; {len(errors)} misclassified. "
        f"Reports: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
