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
data/raw|processed  datasets (git-ignored)
data/samples/       tiny demo CSV committed for tests/demo
docs/               proposal, milestones, rubric, work split, experiment log
notebooks/          EDA + experiments (Colab/Kaggle friendly)
scripts/            train / evaluate / run helpers
src/grievance/      data, models, api (FastAPI), dashboard (Streamlit)
tests/              pytest
reports/            metrics, per-language report, figures
```

## Quickstart

### 1. Install dependencies
**With `uv` (recommended):**
```bash
uv sync
```

**otherwise with `pip`:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the application:
```bash
python scripts/make_sample_data.py                 # builds data/samples/grievances_sample.csv
python scripts/train_baseline.py                   # TF-IDF + LogReg baseline (Week 6)
python scripts/train_transformer.py --config configs/muril.yaml   # fine-tune (use Colab/Kaggle GPU)
uvicorn grievance.api.main:app --reload --app-dir src   # API  -> http://127.0.0.1:8000/docs
streamlit run src/grievance/dashboard/app.py            # dashboard
```

## Baseline evaluation and error analysis

After training the baseline, generate reports with:

```bash
uv run python scripts/evaluate.py
```

This uses the processed dataset when available, otherwise the demo CSV, and
recreates the test split with seed 42. For a full dataset, pass `--data PATH` and
use the same data and `--seed` as training. For an already held-out CSV, evaluate
every row without splitting it again:

```bash
uv run python scripts/evaluate.py --test-data data/processed/test.csv --model reports/baseline/model.pkl --output-dir reports/evaluation
```

Held-out files must contain nonempty `text`, `language`, and `department` columns.
Keep these records and duplicate complaints out of the model's training data.
The script exports three files to `--output-dir` (default: `reports/`), replacing
existing files with the same names:

- `per_language_report.csv`: sample count, accuracy, and macro-F1 per language.
  Macro-F1 averages the labels present in the actual or predicted values within
  each language, matching the existing evaluation behavior.
- `confusion_matrix.csv`: counts with actual departments as rows and predicted
  departments as columns, including all five departments even when absent.
- `misclassified_complaints.csv`: incorrect predictions with text, language,
  actual/predicted departments, and `source_index` (the zero-based input record
  index before splitting, not a physical CSV line number). Perfect predictions
  produce a file containing only column headers.

The repeated demo complaints support pipeline smoke tests only; their scores do
not measure generalization. These reports evaluate department classification;
urgency and sentiment evaluation remain separate work.

## Tasks (multi-task heads)
| Task | Labels |
|------|--------|
| Department | water, roads, power, sanitation, other |
| Urgency | low, medium, high, critical |
| Sentiment | negative, neutral, positive |

## Team workflow
Every member commits weekly (commit history is graded). See `docs/WORKPLAN.md`.
