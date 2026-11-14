from grievance.data.urgency_rules import weak_urgency
from grievance.models.baseline import build_baseline

def test_urgency_critical():
    assert weak_urgency("live wire fallen on road") == "critical"

def test_baseline_fits():
    m = build_baseline().fit(["no water", "pothole road"], ["water", "roads"])
    assert m.predict(["water not coming"])[0] in {"water", "roads"}
