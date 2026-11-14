# Experiment log
| Date | Who | Model | Config | Dept macro-F1 | Urgency F1 | Sentiment F1 | Notes |
|------|-----|-------|--------|---------------|-----------|--------------|-------|
| 2026-11-14 | team | TF-IDF + LogReg | word 1-2-grams, class_weight=balanced, seed 42; 1,000 generated rows, 700/150/150 | 0.969 (acc 0.967) | 0.471 (keyword rules) | 0.156 (keyword rules) | Templated data, so inflated; see docs/BASELINE_METRICS.md |
| 2026-11-14 | team | same model, real text | scored on 186 distinct BBMP phrases | acc 0.41 all / 0.55 non-"other" / 0.91 frequency-weighted | - | - | Real-text sanity check; BBMP mapping is noisy |

