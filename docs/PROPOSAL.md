# Project Proposal — Multilingual Public Grievance Classifier and Router
CS474 Deep Learning · Project P8 · Team: *(fill in names)* · Date: 14 Nov 2026

## 1. Problem statement
Municipal bodies and portals such as CPGRAMS receive complaints in many languages and spend effort reading, categorising and
forwarding each one. We will build a system that reads a complaint and (a) classifies its **department** (water, roads, power,
sanitation, other), (b) scores its **urgency** (low, medium, high, critical) and **sentiment** (negative, neutral, positive), and
(c) **routes** it to the responsible authority. The deliverable is a REST API and a dashboard where users paste or upload
complaints, plus a per-language accuracy report. It is a decision-support prototype: a person can always review and override the routing.

## 2. Data and verification status
| Source | Use | Status (14 Nov 2026) |
|---|---|---|
| BBMP citizen grievances, Bengaluru (GitHub, Vonter/bbmp-citizen-grievances) | Real-world check, department taxonomy, EDA | **Downloaded and inspected**: 1,230,890 rows, 32 categories; only 187 distinct complaint phrases (no free text) |
| IndicSentiment (AI4Bharat, Hugging Face) | Sentiment head across 13 Indic languages | **Downloaded**: 15,002 labelled reviews (validation + test splits; positive/negative only) |
| BanglaPSG (Mendeley Data, CC BY 4.0) | Bengali complaints with category and severity | Page located; **download not yet verified** |
| CivicNLP-608 (IEEE DataPort, CC BY 4.0) | English–Marathi complaints, 8 categories | Needs an IEEE DataPort subscription; not available to the team |
| CPGRAMS statistics (data.gov.in) | Background only | Aggregate counts, no complaint text |
| Team-generated complaints (en/hi/mr) | Training base while real text is scarce | Built by `scripts/generate_complaints.py`; 1,000 rows tagged `label_source=generated` |

**Risk and mitigation.** No free dataset provides full-text, multilingual, department-labelled complaints. We therefore combine
generated complaints with real sources, report generated and real results separately, ask native speakers to review Hindi and
Marathi rows, and add team-collected public complaints as they become available. A baseline trained on generated text scored
97% on a generated test split but only 41% on the real BBMP phrases (55% excluding the "other" class), which is why a real
held-out set is part of the evaluation.

## 3. Related work
1. Khanuja et al., *MuRIL: Multilingual Representations for Indian Languages*, arXiv:2103.10730, 2021 — BERT model pre-trained on 17 Indian languages including transliterated text; our main candidate encoder.
2. Kakwani et al., *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages*, Findings of EMNLP 2020 — IndicCorp and the first IndicBERT.
3. Doddapaneni et al., *Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages*, ACL 2023 (arXiv:2212.05409) — IndicCorp v2, IndicBERT v2 and the IndicXTREME benchmark, including IndicSentiment.
4. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019 — multilingual BERT is our general-purpose baseline encoder.
5. Singh et al., *Identifying Complaints from Product Reviews: A Case Study on Hindi*, ICON 2020 — complaint detection in Hindi with class-imbalance handling (SMOTE); closest prior work on Indic complaint text.
6. Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)*, ACL 2020 — strong multilingual reference model if time permits.

## 4. Approach
- **Baseline (done):** TF-IDF word 1–2-grams with class-weighted logistic regression for departments; keyword rules for urgency and sentiment.
- **Main models:** fine-tune MuRIL, IndicBERT and mBERT with one shared encoder and three classification heads trained jointly (`scripts/train_transformer.py`), using inverse-frequency class weights for imbalance.
- **Ablations (Weeks 10–13):** encoder choice, single-task vs multitask, class weighting on/off, generated-only vs generated + real data, with vs without the sentiment-only IndicSentiment data.
- **Evaluation:** macro-F1 and accuracy per task and **per language**, department confusion matrix, error analysis of misrouted complaints, and a held-out set of real text kept out of training.
- **Serving:** FastAPI (`/predict`, `/predict/batch`) and a Streamlit dashboard with a per-language report tab; Docker image for the API.

## 5. Work split (10 members)
| Role | Members | Responsibility |
|---|---|---|
| Data (2) | *TBD* | Collect public complaints, labelling guide, cleaning, native-speaker review of generated rows |
| Modelling (3) | *TBD* | Baselines, MuRIL / IndicBERT / mBERT fine-tuning, multitask heads, imbalance handling |
| Evaluation (2) | *TBD* | Metrics, per-language report, ablations, error analysis |
| Deployment (2) | *TBD* | API, dashboard, hosting, demo |
| Coordination (1) | *TBD* | Repo hygiene, weekly commit check, report and presentation |

Every member commits weekly; history is graded.

## 6. Timeline
Week 4 proposal · Week 6 cleaned data, EDA notebook and baseline metrics (done in the repo) · Week 8 first transformer model vs
baseline · Week 10 ablations and architecture comparison · Week 12 working demo · Week 13 IEEE report and repository · Week 14 presentation and viva.

## 7. Ethics and limitations
Only public, anonymized data are used and personal details are stripped before sharing. Generated rows are labelled as such in the
data and in every report. Routing errors affect real people, so critical complaints should always be reviewable by a human. Language
coverage is uneven: Hindi, Marathi and English first; Bengali and other languages depend on data we can obtain.
