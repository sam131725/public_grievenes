# Multilingual Public Grievance Classifier and Router (P8)

CS474 Deep Learning — Major Project (Dr. Mehul Kumar)

Automatically classify citizen complaints (water, roads, power, sanitation), score
**urgency** and **sentiment**, and **route** each complaint to the right department.
Supports English + Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu, ...).

**Deliverable:** API + dashboard — paste/upload complaints → department, urgency, sentiment;
plus a per-language accuracy report.

## Layout
```
configs/            training configs (YAML)
data/raw|processed  datasets (git-ignored; see docs/DATA.md)
data/samples/       tiny demo CSV committed for tests/demo
docs/               proposal, milestones, rubric, work split, experiment log
notebooks/          EDA + experiments (Colab/Kaggle friendly)
scripts/            train / evaluate / run helpers
src/grievance/      data, models, api (FastAPI), dashboard (Streamlit)
tests/              pytest
reports/            metrics, per-language report, figures
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_sample_data.py                 # builds data/samples/grievances_sample.csv
python scripts/train_baseline.py                   # TF-IDF + LogReg baseline (Week 6)
python scripts/train_transformer.py --config configs/muril.yaml   # fine-tune (use Colab/Kaggle GPU)
uvicorn grievance.api.main:app --reload --app-dir src   # API  -> http://127.0.0.1:8000/docs
streamlit run src/grievance/dashboard/app.py            # dashboard
```

## Tasks (multi-task heads)
| Task | Labels |
|------|--------|
| Department | water, roads, power, sanitation, other |
| Urgency | low, medium, high, critical |
| Sentiment | negative, neutral, positive |

## Team workflow
Every member commits weekly (commit history is graded). See `docs/WORKPLAN.md`.
