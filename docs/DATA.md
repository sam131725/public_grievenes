# Data plan

Processed schema (`data/processed/grievances.csv`, built by `scripts/prepare_data.py`):
`text, language, department, urgency, sentiment, label_source`.

`label_source` says where the labels came from: `gold` (raw file), `weak` (keyword rules filled in urgency/sentiment),
or `generated` (rule-assembled complaints, see below).

## Sources and status
| Source | Role | Notes |
|--------|------|-------|
| Generated complaints (en/hi/mr) | training base | `scripts/generate_complaints.py` — see limitations below |
| BBMP grievance export (Bengaluru) | real-world check + EDA | ~1.2M rows, ~190 distinct phrases, no free text |
| IndicSentiment (AI4Bharat, Hugging Face `ai4bharat/IndicSentiment`) | sentiment only, 13 languages | downloaded from Hugging Face and converted to `data/raw/indicsentiment/sentiment.csv` (15,002 rows, product reviews, validation+test splits); not yet merged into training |
| BanglaPSG (Mendeley, CC BY 4.0) | Bengali + severity labels | not used: Bangla is outside our three languages (en/hi/mr) |
| CivicNLP-608 (IEEE DataPort) | English/Marathi complaints | not used: needs an IEEE DataPort subscription; DOI 10.21227/13nz-rt85 |
| CPGRAMS | background statistics only | data.gov.in has counts, no complaint text |

Issues to handle: class imbalance (class weights), code-mixed / transliterated text, low-resource languages, label noise,
weak labels for urgency/sentiment (`src/grievance/data/urgency_rules.py`).

## Why part of the data is generated
The project needs complaint text in several Indian languages labelled with a department. We searched for such a corpus and found none that is
public, full-text and multilingual: BBMP has only category labels as text (about 190 distinct phrases), IndicSentiment has reviews with sentiment
but no departments, CivicNLP-608 needs a paid IEEE DataPort subscription, and BanglaPSG is Bangla. To have enough labelled text to build and
compare models, the team generated complaints from hand-written templates in English, Hindi and Marathi. This lets us test the full pipeline,
but scores on generated text are inflated, so we also check on real BBMP phrases and plan a small real Hindi/Marathi test set.

## Generated complaints (English, Hindi, Marathi)
`scripts/generate_complaints.py` writes `data/raw/generated_complaints.csv` (1,000 rows by default: 420 en, 300 hi, 280 mr)
covering all five departments, four urgency levels and three sentiments. Complaints are assembled from hand-written
sentences (issue + optional location + duration + closing line); urgency follows the issue's severity, raised by long
duration or an urgent closing line. Every row carries `label_source=generated`.

Limitations to state in the report:
- Wording is templated, so a model scores very high on a split drawn from the same data. This is **not** evidence of real-world accuracy.
- Evaluate on real text as well. The BBMP phrases (`data/raw/bbmp/bbmp_seed.csv`, from `scripts/import_bbmp.py`) are a small real set: a baseline trained only on generated rows scored 41% over all 186 distinct BBMP phrases and 55% on the 38 non-"other" phrases (91% weighted by how often each phrase occurs).
- Hindi and Marathi sentences were written without a native-speaker review; have team members who speak these languages read a sample before reporting per-language results.
- Hinglish / romanized and code-mixed complaints are not covered.

## BBMP grievance export (Bengaluru)
`data/raw/bbmp/citizen-grievances.csv` (download: https://github.com/Vonter/bbmp-citizen-grievances) has ~1.2M rows but only ~190
distinct complaint phrases (a sub-category label, no free text). Use it for EDA and as a real-world check, not as a training
corpus. Staff-name columns are never loaded.

## Ethics
Use only public, anonymized data. Do not commit raw downloads (`data/raw/` is git-ignored). Complaint text can contain names,
phone numbers or addresses — strip them before sharing. This is a decision-support prototype.
