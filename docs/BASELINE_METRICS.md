# Baseline metrics (Week 6)

**Model:** TF-IDF (word 1–2-grams, sublinear TF) + logistic regression, `class_weight="balanced"`, `max_iter=1000`, seed 42.
**Task:** department classification (water, roads, power, sanitation, other).
**Data:** 1,000 team-generated English/Hindi/Marathi complaints (`label_source=generated`, see [DATA.md](DATA.md)); stratified split 700 / 150 / 150 (train / validation / test).
**Reproduce:** `python scripts/generate_complaints.py && python scripts/prepare_data.py && python scripts/train_baseline.py && python scripts/evaluate.py` (full walk-through in `notebooks/01_eda.ipynb`).

> **Read this first.** The generated text is templated, so the test split shares its phrasing with the training split. The
> scores below show the pipeline works; they are *not* an estimate of accuracy on real complaints. The real-text check at the
> end is the number to quote when judging generalisation.

## Department classification — generated test split (n = 150)

| Metric | Value |
|---|---|
| Validation accuracy | 0.993 |
| Test accuracy | 0.967 |
| Test macro-F1 | 0.969 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| water | 0.88 | 1.00 | 0.94 | 30 |
| roads | 1.00 | 0.97 | 0.99 | 34 |
| power | 1.00 | 1.00 | 1.00 | 30 |
| sanitation | 0.97 | 0.88 | 0.92 | 33 |
| other | 1.00 | 1.00 | 1.00 | 23 |

Confusion matrix (rows = actual, columns = predicted):

| | water | roads | power | sanitation | other |
|---|---|---|---|---|---|
| **water** | 30 | 0 | 0 | 0 | 0 |
| **roads** | 0 | 33 | 0 | 1 | 0 |
| **power** | 0 | 0 | 30 | 0 | 0 |
| **sanitation** | 4 | 0 | 0 | 29 | 0 |
| **other** | 0 | 0 | 0 | 0 | 23 |

The only systematic confusion is sanitation → water (4 of 33), e.g. drain and sewer overflow wording that overlaps with water complaints.

### Per language

| Language | n | Accuracy | Macro-F1 |
|---|---|---|---|
| en | 51 | 1.000 | 1.000 |
| hi | 62 | 0.919 | 0.929 |
| mr | 37 | 1.000 | 1.000 |

Hindi is the weakest language. Each language has only a few dozen test rows, so these differences are noisy.

## Urgency and sentiment — keyword rules (no trained model yet)

The API currently fills these two tasks with the rules in `src/grievance/data/urgency_rules.py`. Scored against the dataset labels on the same test split:

| Task | Accuracy | Macro-F1 |
|---|---|---|
| Urgency (4 levels) | 0.473 | 0.471 |
| Sentiment (3 levels) | 0.233 | 0.156 |

The sentiment rules can only output *negative* or *neutral*, so they never predict *positive*. These two numbers are the bar the transformer heads must beat.

## Real-text check — BBMP complaint phrases

The same baseline (trained only on the generated rows) scored on the 186 distinct real phrases from the BBMP (Bengaluru) grievance export, with BBMP categories mapped to our departments:

| Subset | n phrases | Accuracy |
|---|---|---|
| All phrases | 186 | 0.41 |
| Excluding "other" | 38 | 0.55 |
| Excluding "other", weighted by how often each phrase occurs | 38 | 0.91 |

Caveats: most "other" phrases are topics the generated data does not cover (forest, veterinary, khata services), which drives the low overall figure; and the BBMP category mapping is itself noisy (for example "water stagnation" sits under *Road Maintenance*, so it counts as roads). Most frequent mistakes: "Debris Removal / Construction Material" (roads → other), "Dead animal(s)" (sanitation → power), "Requirement For New Street Lights" (power → sanitation).

## What the baseline tells us
- The pipeline (clean → split → train → per-language report → confusion matrix) works end to end.
- On real phrasing the baseline drops sharply, so the next gains come from real data and from the multilingual transformers (MuRIL / IndicBERT / mBERT), not from tuning this model.
- Urgency and sentiment need trained heads; the rule baseline is weak.
