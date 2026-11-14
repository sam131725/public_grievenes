import pickle
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from grievance.labels import DEPARTMENTS
from grievance.models.evaluate import (
    department_confusion_matrix,
    misclassified_complaints,
    per_language_report,
)


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "text": ["No water", "Pothole", "बिजली नहीं है", "कचरा नहीं उठा"],
            "language": ["en", "en", "hi", "hi"],
            "department": ["water", "roads", "power", "sanitation"],
            "pred": ["water", "water", "power", "roads"],
        },
        index=[3, 8, 12, 19],
    )


def test_confusion_matrix_counts_and_orientation(predictions):
    matrix = department_confusion_matrix(predictions, "department", "pred")

    assert list(matrix.index) == list(matrix.columns) == DEPARTMENTS
    assert matrix.index.name == "actual_department"
    assert matrix.columns.name == "predicted_department"
    assert matrix.to_numpy().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


@pytest.mark.parametrize("column", ["department", "pred"])
@pytest.mark.parametrize("invalid", ["unknown", None])
def test_confusion_matrix_rejects_invalid_labels(predictions, column, invalid):
    predictions.loc[3, column] = invalid
    with pytest.raises(ValueError, match="valid department labels"):
        department_confusion_matrix(predictions, "department", "pred")


def test_errors_preserve_source_indices_and_multilingual_text(predictions):
    original = predictions.copy(deep=True)
    errors = misclassified_complaints(predictions, "department", "pred")

    assert errors.to_dict("records") == [
        {"source_index": 8, "text": "Pothole", "language": "en",
         "actual_department": "roads", "predicted_department": "water"},
        {"source_index": 19, "text": "कचरा नहीं उठा", "language": "hi",
         "actual_department": "sanitation", "predicted_department": "roads"},
    ]
    pd.testing.assert_frame_equal(predictions, original)


def test_perfect_predictions_export_error_headers(predictions):
    predictions["pred"] = predictions["department"]
    errors = misclassified_complaints(predictions, "department", "pred")
    assert errors.empty
    assert errors.to_csv(index=False).strip() == (
        "source_index,text,language,actual_department,predicted_department"
    )


def test_per_language_metrics(predictions):
    report = per_language_report(predictions, "department", "pred").set_index("language")
    assert report["n"].to_dict() == {"en": 2, "hi": 2}
    assert report["accuracy"].to_dict() == {"en": 0.5, "hi": 0.5}
    assert report.loc["en", "macro_f1"] == pytest.approx(1 / 3)
    assert report.loc["hi", "macro_f1"] == pytest.approx(1 / 3)


def test_evaluation_cli_exports_held_out_rows(tmp_path, predictions):
    dataset = tmp_path / "test.csv"
    predictions.drop(columns="pred").to_csv(dataset, index=False)
    model = DummyClassifier(strategy="constant", constant="water")
    model.fit(predictions["text"], predictions["department"])
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    output = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"

    result = subprocess.run(
        [sys.executable, str(script), "--test-data", str(dataset),
         "--model", str(model_path), "--output-dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )

    assert "Evaluated 4 complaints; 3 misclassified" in result.stdout
    errors = pd.read_csv(output / "misclassified_complaints.csv")
    assert errors["source_index"].tolist() == [1, 2, 3]
    assert errors["text"].tolist() == predictions["text"].tolist()[1:]
    matrix = pd.read_csv(output / "confusion_matrix.csv", index_col="actual_department")
    assert matrix["water"].tolist() == [1, 1, 1, 1, 0]
    assert matrix.to_numpy().sum() == 4
    languages = pd.read_csv(output / "per_language_report.csv")
    assert languages["n"].sum() == 4


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("text,department\nPothole,roads\n", "Missing required columns: language"),
        ("text,language,department\n", "The evaluation dataset has no rows"),
        ("text,language,department\n   ,en,roads\n", "Column 'text' must contain nonempty strings"),
    ],
)
def test_evaluation_cli_rejects_invalid_test_data(tmp_path, contents, message):
    dataset = tmp_path / "invalid.csv"
    dataset.write_text(contents, encoding="utf-8")
    output = tmp_path / "reports"
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"

    result = subprocess.run(
        [sys.executable, str(script), "--test-data", str(dataset),
         "--model", str(tmp_path / "unused-model.pkl"), "--output-dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()
