import json

import pytest

from grievance.data.indicsentiment import load_indicsentiment


def _write(d, split, lang, recs):
    p = d / split
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{lang}.json").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")


def test_load_drops_unlabelled_and_lowercases(tmp_path):
    _write(tmp_path, "test", "hi", [
        {"INDIC REVIEW": "बहुत अच्छा", "LABEL": "Positive"},
        {"INDIC REVIEW": "खराब सेवा", "LABEL": "Negative"},
        {"INDIC REVIEW": "बिना लेबल", "LABEL": None},
    ])
    _write(tmp_path, "validation", "bn", [{"INDIC REVIEW": "ভালো", "LABEL": "Positive"}])
    df = load_indicsentiment(tmp_path)
    assert len(df) == 3
    assert set(df["sentiment"]) == {"positive", "negative"}
    assert set(df["split"]) == {"test", "validation"}


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_indicsentiment(tmp_path)
