import numpy as np
import pytest

from grievance.models.imbalance import balanced_class_weights
from grievance.models.predictor import Predictor, load_predictor


def test_balanced_weights_favour_rare_classes():
    w = balanced_class_weights([0, 0, 0, 0, 1], 3)
    assert w[1] > w[0] > 0
    assert w[2] == 0  # absent class


def test_balanced_weights_uniform_when_balanced():
    w = balanced_class_weights([0, 1, 2, 0, 1, 2], 3)
    assert np.allclose(w, 1.0)


def test_load_predictor_defaults_to_baseline(monkeypatch):
    monkeypatch.delenv("GRIEVANCE_MODEL_DIR", raising=False)
    assert isinstance(load_predictor(), Predictor)


def test_predictor_batch_and_info(tmp_path):
    import pickle
    from grievance.models.baseline import build_baseline

    m = build_baseline().fit(["no water", "bad road pothole", "power cut", "garbage"],
                             ["water", "roads", "power", "sanitation"])
    path = tmp_path / "m.pkl"; path.write_bytes(pickle.dumps(m))
    p = Predictor(path)
    out = p.predict_batch(["no water here", "pothole on road"])
    assert [o["department"] for o in out] == ["water", "roads"]
    assert p.info()["loaded"] is True
    with pytest.raises(ValueError):
        p.predict("   ")


def test_transformer_modules_compile():
    # torch isn't required in CI; just make sure the files are syntactically valid
    import py_compile, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for f in ["src/grievance/models/transformer_predictor.py", "scripts/train_transformer.py", "scripts/compare_models.py"]:
        py_compile.compile(str(root / f), doraise=True)
