import pandas as pd

from grievance.data.bbmp import load_bbmp_seed, map_department


def test_map_department():
    assert map_department("Electrical") == "power"
    assert map_department("Solid Waste (Garbage) Related") == "sanitation"
    assert map_department("Road Maintenance(Engg)") == "roads"
    assert map_department("Storm  Water Drain(SWD)") == "water"
    assert map_department("Forest") == "other"


def test_seed_dedupes_and_never_reads_staff_columns(tmp_path):
    f = tmp_path / "b.csv"
    pd.DataFrame({
        "Category": ["Electrical", "Electrical", "Others", "Forest"],
        "Sub Category": ["Street Light Not Working"] * 3 + ["Tree fall"],
        "Staff Name": ["A", "B", "C", "D"],
    }).to_csv(f, index=False)
    out = load_bbmp_seed(f)
    assert list(out.columns) == ["text", "language", "department", "count"]
    row = out[out.text == "Street Light Not Working"].iloc[0]
    assert row.department == "power" and row["count"] == 3  # majority category wins
    assert len(out) == 2
