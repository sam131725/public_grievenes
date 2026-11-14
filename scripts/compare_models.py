"""Collect reports/*/metrics.json into one results table (reports/results_table.md).

Use for the Week 8/10 'new model vs baseline' and architecture comparison.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((ROOT / "reports").glob("*/metrics.json")):
    m = json.loads(f.read_text()); t = m["test"]
    cell = lambda task, k: f"{t[task][k]:.3f}" if task in t else "-"
    rows.append((m["model"], cell("department", "accuracy"), cell("department", "macro_f1"),
                 cell("urgency", "macro_f1"), cell("sentiment", "macro_f1"), f.parent.name))
if not rows:
    raise SystemExit("No reports/*/metrics.json found. Train a model first.")
lines = ["| Model | Dept acc | Dept macro-F1 | Urgency macro-F1 | Sentiment macro-F1 | Run dir |",
         "|---|---|---|---|---|---|"] + ["| " + " | ".join(r) + " |" for r in rows]
out = "\n".join(lines)
(ROOT / "reports/results_table.md").write_text(out + "\n")
print(out)
