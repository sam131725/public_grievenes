import pandas as pd
import pytest

from grievance.data.build import build_dataset, guess_language, infer_department, load_raw


def test_guess_language_scripts():
    assert guess_language("No water since two days") == "en"
    assert guess_language("हमारे इलाके में पानी नहीं आ रहा") == "hi"
    assert guess_language("आमच्या भागात पाणी येत नाही") == "mr"
    assert guess_language("தண்ணீர் வரவில்லை") == "ta"
    assert guess_language("নলকূপে জল নেই") == "bn"


def test_infer_department():
    assert infer_department("big pothole on the road") == "roads"
    assert infer_department("garbage not collected") == "sanitation"
    assert infer_department("transformer blew, no electricity") == "power"
    assert infer_department("please help with my pension") == "other"


def _raw():
    return pd.DataFrame(
        {
            "complaint": [
                "No water supply in our street for three days",
                "No water supply in our street for three days",  # duplicate
                "short",  # too short
                "Pothole on the main road near the school",
                "Garbage is piling up near the market",
            ],
            "department": ["Water", "water", "water", "ROADS", None],
            "urgency": ["high", "high", "low", None, None],
        }
    )


def test_build_dataset_cleans_and_labels(tmp_path):
    f = tmp_path / "raw.csv"
    _raw().to_csv(f, index=False)
    df, report = build_dataset(load_raw([f]))
    assert report["dropped_too_short"] == 1
    assert report["dropped_duplicates"] == 1
    assert report["dropped_missing_department"] == 1
    assert list(df.columns) == ["text", "language", "department", "urgency", "sentiment", "label_source"]
    assert set(df["department"]) == {"water", "roads"}
    assert df["urgency"].notna().all() and df["sentiment"].notna().all()
    # row 1 had gold urgency but no sentiment -> weak overall
    assert (df["label_source"] == "weak").all()
    assert (df["language"] == "en").all()


def test_infer_missing_department_keeps_row():
    df, report = build_dataset(load_raw_from_df(_raw()), infer_missing_department=True)
    assert "sanitation" in set(df["department"])
    assert report["departments_inferred"] == 1


def load_raw_from_df(frame):
    frame = frame.rename(columns={"complaint": "text"})
    return frame


def test_missing_text_column_raises(tmp_path):
    f = tmp_path / "bad.csv"
    pd.DataFrame({"foo": ["x"]}).to_csv(f, index=False)
    with pytest.raises(ValueError):
        load_raw([f])
