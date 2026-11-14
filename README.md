# Multilingual Public Grievance Classifier and Router

> CS474 Deep Learning — Major Project **P8** · IIIT Bhagalpur · Autumn 2026 · Dr. Mehul Kumar

Citizens file complaints in many languages. This system reads a complaint, decides which
**department** it belongs to (water, roads, power, sanitation, other), scores its **urgency**
and **sentiment**, and **routes** it to the responsible authority — through a REST API and a
dashboard, with a **per-language accuracy report** so we can see where the model is weak.

```
complaint text ──► clean ──► encoder (MuRIL / IndicBERT / mBERT) ──┬─► department head ─► route-to
 (en, hi, mr, …)                  or TF-IDF + LogReg baseline       ├─► urgency head
                                                                    └─► sentiment head
```

| Task | Labels |
|------|--------|
| Department | `water`, `roads`, `power`, `sanitation`, `other` |
| Urgency | `low`, `medium`, `high`, `critical` |
| Sentiment | `negative`, `neutral`, `positive` |

## Features
- **Two model families:** a fast TF-IDF + logistic-regression baseline, and a multilingual transformer with three task heads trained jointly.
- **Class imbalance:** inverse-frequency loss weights (safe when a class is missing from a split).
- **Multilingual evaluation:** accuracy and macro-F1 per language, department confusion matrix, and a list of misclassified complaints.
- **Data pipeline:** merges raw CSVs, cleans and de-duplicates, detects language by script, fills missing urgency/sentiment with rules (flagged as `weak`), and writes a data card.
- **Serving:** FastAPI (`/predict`, `/predict/batch`) and a Streamlit dashboard (paste, CSV upload, report tab). Docker image for the baseline.

## Quickstart

```bash
# 1. Install (uv recommended; pip also works)
uv sync                                    # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Try it on the bundled demo data (60 repeated rows — a smoke test, not a benchmark)
python scripts/make_sample_data.py
python scripts/train_baseline.py           # trains baseline -> reports/baseline/
python scripts/evaluate.py                 # per-language report, confusion matrix, errors

# 3. Serve
uvicorn grievance.api.main:app --app-dir src --reload     # http://127.0.0.1:8000/docs
streamlit run src/grievance/dashboard/app.py              # dashboard
```
Shortcuts are in the [Makefile](Makefile) (`make baseline`, `make test`, `make api`, …).

## Using your real data

0. No real data yet? `python scripts/generate_complaints.py` writes a labelled English/Hindi/Marathi set to `data/raw/` (rows are tagged `label_source=generated` — see Limitations).
1. Drop raw CSVs into `data/raw/`. A `text` column is required (`complaint`, `description`, `grievance` are auto-detected; or pass `--text-col`). Optional columns: `language`, `department`, `urgency`, `sentiment`.
2. Build the processed dataset:
   ```bash
   python scripts/prepare_data.py                      # add --infer-department to keyword-guess missing departments
   python scripts/eda.py                               # tables + charts -> reports/eda/
   ```
   This writes `data/processed/grievances.csv` and `data/processed/data_card.json` (row counts, dropped rows, per-language/department counts, warnings).
3. Processed schema: `text, language, department, urgency, sentiment, label_source` where `label_source` is `gold` (from the raw file) or `weak` (keyword rules — validate on a hand-labelled sample before reporting urgency/sentiment results).

## Training

```bash
python scripts/train_baseline.py                                   # baseline
python scripts/train_transformer.py --config configs/muril.yaml    # MuRIL
python scripts/train_transformer.py --config configs/indicbert.yaml
python scripts/train_transformer.py --config configs/mbert.yaml
python scripts/compare_models.py                                   # -> reports/results_table.md
```
Transformer runs need a GPU — use Colab or Kaggle and pool the team's quotas. The one-click way is `notebooks/02_train_colab.ipynb` (clone, train all three models, compare, download a results zip). Each run writes `model.pt`, tokenizer, `metrics.json`, `per_language_report.csv` and `confusion_matrix.csv` to its `output_dir` (`reports/muril`, …). The best epoch (by validation department macro-F1) is kept. Hyper-parameters live in `configs/*.yaml`.

Serve a fine-tuned model by pointing the API/dashboard at its folder:
```bash
GRIEVANCE_MODEL_DIR=reports/muril uvicorn grievance.api.main:app --app-dir src
```
Without that variable the TF-IDF baseline is served.

## Evaluation and error analysis

```bash
python scripts/evaluate.py                                   # uses processed data if present, else the demo CSV; seed 42
python scripts/evaluate.py --test-data data/processed/test.csv --model reports/baseline/model.pkl --output-dir reports/evaluation
```
Outputs (existing files with the same names are replaced):
- `per_language_report.csv` — sample count, accuracy and macro-F1 per language
- `confusion_matrix.csv` — actual departments as rows, predicted as columns (all five always present)
- `misclassified_complaints.csv` — wrong predictions with `source_index` (zero-based record index before splitting)

Use the same data and `--seed` as training, and keep held-out rows and duplicates out of training. These reports cover **department** classification; urgency/sentiment are reported by the transformer's `metrics.json`.

## API

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "No water supply in our street for three days"}'
```
```json
{"department": "water", "urgency": "high", "sentiment": "negative",
 "route": "Water Supply & Sewerage Board"}
```
The transformer backend also returns `department_confidence`, `urgency_confidence`, `sentiment_confidence`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | liveness + whether a model is loaded |
| `GET /model` | which backend is serving |
| `POST /predict` | one complaint `{"text": "..."}` |
| `POST /predict/batch` | `{"complaints": ["...", "..."]}` (1–500 items) |

Empty text → `422`; no trained model available → `503`. Interactive docs at `/docs`.

**Docker (baseline):** `make docker` builds and runs the API on port 8000.

## Results

_Baseline row is from the generated test split (*) — see [docs/BASELINE_METRICS.md](docs/BASELINE_METRICS.md) for why this overstates real accuracy (41% on real BBMP phrases). † keyword rules, no trained head yet. Fill in the transformer rows after training (`python scripts/compare_models.py` builds the table)._

| Model | Dept acc | Dept macro-F1 | Urgency macro-F1 | Sentiment macro-F1 |
|-------|----------|---------------|------------------|--------------------|
| TF-IDF + LogReg | 0.967* | 0.969* | 0.471† | 0.156† |
| MuRIL | | | | |
| IndicBERT | | | | |
| mBERT | | | | |

## Repository layout

```
configs/            training configs (YAML) per model
data/raw/           raw CSVs (git-ignored)      data/processed/  built dataset (git-ignored)
data/samples/       tiny demo CSV
docs/               brief, milestones + rubric, work split, data plan, experiment log
notebooks/          EDA and Colab/Kaggle notebooks
reports/            baseline + per-model outputs
scripts/            prepare_data, eda, train_baseline, train_transformer, evaluate, compare_models
src/grievance/
  data/             build (pipeline), preprocess (split), urgency_rules (weak labels)
  models/           baseline, multitask, imbalance, evaluate, predictor, transformer_predictor
  api/              FastAPI app          dashboard/  Streamlit app
tests/              pytest suite (run: make test)
```

## Testing

```bash
make test        # or: PYTHONPATH=src python -m pytest -q
```
Covers the data pipeline, evaluation reports, model helpers/predictors and API validation. Transformer training/inference needs `torch` and a model download, so it is exercised by running a training job rather than by the unit tests.

## Limitations and ethics

- **Public, anonymized data only.** Complaint text can contain names, phone numbers or addresses; do not commit or publish personal data, and respect each site's terms when scraping.
- **Decision support, not a decision-maker.** Misrouted or under-prioritised complaints affect real people; a human should be able to review and override routing, especially for `critical` urgency.
- **Weak labels.** Where urgency/sentiment come from keyword rules, scores on those tasks reflect the rules, not human judgement.
- **Language coverage is uneven.** Low-resource languages and code-mixed or transliterated text will score lower — check the per-language report before trusting a language.
- **Demo data is synthetic and repeated**, so its metrics only prove the pipeline runs.
- **Generated complaints are templated** (`label_source=generated`; see [docs/DATA.md](docs/DATA.md)). High scores on a split from the same data overstate real-world accuracy — check results on real text such as the BBMP phrases, and have native speakers review the Hindi/Marathi rows.

## Team workflow

Roles and weekly ownership are in [docs/WORKPLAN.md](docs/WORKPLAN.md); milestones and the rubric in [docs/MILESTONES.md](docs/MILESTONES.md). Every member commits weekly on a feature branch and opens a pull request — commit history is graded.

## References

- Khanuja et al., *MuRIL: Multilingual Representations for Indian Languages*, 2021 (arXiv:2103.10730).
- Kakwani et al., *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages*, Findings of EMNLP 2020 (IndicBERT).
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2019 (multilingual BERT).
- Datasets: CPGRAMS-style public grievance samples, scraped municipal complaints, IndicNLP corpora — see [docs/DATA.md](docs/DATA.md).
