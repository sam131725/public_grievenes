import pandas as pd

from grievance.data.build import build_dataset
from grievance.data.generate import generate
from grievance.labels import DEPARTMENTS, SENTIMENT, URGENCY


def test_generate_is_valid_unique_and_deterministic():
    a = generate(60, 40, 40, seed=1)
    b = generate(60, 40, 40, seed=1)
    assert a.equals(b)
    assert a["text"].is_unique
    assert set(a["department"]) <= set(DEPARTMENTS)
    assert set(a["urgency"]) <= set(URGENCY)
    assert set(a["sentiment"]) <= set(SENTIMENT)
    assert set(a["language"]) == {"en", "hi", "mr"}
    assert (a["label_source"] == "generated").all()


def test_scripts_match_languages():
    df = generate(30, 30, 30, seed=2)
    deva = df[df.language != "en"]["text"].str.contains("[ऀ-ॿ]", regex=True)
    assert deva.all()
    assert not df[df.language == "en"]["text"].str.contains("[ऀ-ॿ]", regex=True).any()


def test_severe_issues_get_higher_urgency():
    df = generate(300, 0, 0, seed=3)
    crit = df[df["text"].str.contains("live electric wire|open manhole|Sewage is mixing", regex=True)]
    assert len(crit) and (crit["urgency"] == "critical").all()


def test_build_keeps_generated_label_source():
    raw = generate(40, 20, 20, seed=4)
    out, _ = build_dataset(raw)
    assert (out["label_source"] == "generated").all()
