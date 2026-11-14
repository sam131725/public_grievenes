# P8 — Multilingual Public Grievance Classifier and Router
**Course:** CS474 Deep Learning | **Instructor:** Dr. Mehul Kumar

**Problem:** Automatically classify citizen complaints (water, roads, power, sanitation), score urgency/sentiment and route to the right department.

**Datasets:** CPGRAMS-style public grievance samples; scraped municipal complaint texts; IndicNLP corpora for pre-training context.

**Core techniques (syllabus link):** Fine-tuning IndicBERT / mBERT / MuRIL (Modules 3, 5); class imbalance; multilingual evaluation.

**Final deliverable:** API + dashboard: paste/upload complaints → department, urgency, sentiment; per-language accuracy report.

## Course rules
- 12 teams x 10 students. Roster + 3 preferences in Week 2; allotment by preference + lottery.
- Team roles: data / modelling / evaluation / deployment / coordination.
- Git repo with weekly commits from ALL members is mandatory (history is graded).
- Compute: Colab, Kaggle (30 GPU hrs/week/account), Lightning AI free tier; GitHub Student Pack credits for hosting.
- Frameworks: PyTorch (recommended), Hugging Face Transformers/Datasets. Report in IEEE format.
