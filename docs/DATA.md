# Data plan
Schema (`data/processed/grievances.csv`): `text, language, department, urgency, sentiment`.

Sources (verify each is downloadable before the Week 4 proposal):
1. CPGRAMS-style public grievance samples
2. Scraped municipal complaint texts (respect robots.txt / ToS; no personal data)
3. IndicNLP corpora (pre-training context / augmentation)

Issues to handle: class imbalance (class weights, focal loss, oversampling), code-mixed / transliterated text, low-resource languages, label noise, weak labels for urgency/sentiment (see `src/grievance/data/urgency_rules.py`).
