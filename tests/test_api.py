import pickle

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from grievance.api import main
from grievance.models.baseline import build_baseline
from grievance.models.predictor import Predictor


@pytest.fixture
def trained(tmp_path, monkeypatch):
    m = build_baseline().fit(["no water", "pothole road", "power cut", "garbage waste"],
                             ["water", "roads", "power", "sanitation"])
    path = tmp_path / "m.pkl"; path.write_bytes(pickle.dumps(m))
    monkeypatch.setattr(main, "predictor", Predictor(path))


def test_health_and_model(trained):
    assert main.health() == {"status": "ok", "model_loaded": True}
    assert main.model_info()["backend"]


def test_predict(trained):
    r = main.predict(main.Complaint(text="no water in our area"))
    assert r["department"] == "water" and r["route"] and r["urgency"] and r["sentiment"]


def test_predict_batch(trained):
    r = main.predict_batch(main.Batch(complaints=["pothole on road", "garbage waste pile"]))
    assert [x["department"] for x in r] == ["roads", "sanitation"]


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_text_rejected(bad):
    with pytest.raises(ValidationError):
        main.Complaint(text=bad)


def test_batch_validation():
    with pytest.raises(ValidationError):
        main.Batch(complaints=[])
    with pytest.raises(ValidationError):
        main.Batch(complaints=["ok complaint", " "])
    with pytest.raises(ValidationError):
        main.Batch(complaints=["x"] * (main.MAX_BATCH + 1))


def test_no_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "predictor", Predictor(tmp_path / "missing.pkl"))
    assert main.health()["model_loaded"] is False
    with pytest.raises(HTTPException) as e:
        main.predict(main.Complaint(text="no water"))
    assert e.value.status_code == 503
