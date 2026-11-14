import pytest
import pandas as pd

from grievance.data.preprocess import clean_text, load_and_split


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Please check https://example.com/ticket?id=4 soon", "Please check soon"),
        ("Report at WWW.example.org/path", "Report at"),
        ("  Water   is\nnot   available  ", "Water is not available"),
    ],
)
def test_clean_text_removes_links_and_normalizes_whitespace(text, expected):
    assert clean_text(text) == expected


def test_load_and_split_drops_rows_without_usable_text(tmp_path):
    rows = [
        {"text": f"Complaint {index}", "department": department}
        for department in ("water", "roads")
        for index in range(20)
    ]
    rows.append({"text": "https://example.org", "department": "water"})
    dataset = tmp_path / "complaints.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)

    train, validation, test = load_and_split(dataset)

    assert sum(map(len, (train, validation, test))) == 40
    assert all(part["text"].str.strip().ne("").all() for part in (train, validation, test))


def test_load_and_split_reports_missing_stratification_column(tmp_path):
    dataset = tmp_path / "complaints.csv"
    pd.DataFrame({"text": ["No water"]}).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="Missing stratification column: department"):
        load_and_split(dataset)


def test_load_and_split_reports_empty_cleaned_dataset(tmp_path):
    dataset = tmp_path / "complaints.csv"
    pd.DataFrame({"text": ["https://example.org"], "department": ["water"]}).to_csv(
        dataset, index=False
    )

    with pytest.raises(ValueError, match="No complaints with nonempty text"):
        load_and_split(dataset)
