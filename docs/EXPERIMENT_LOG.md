# Experiment log
| Date | Who | Model | Config | Dept macro-F1 | Urgency F1 | Sentiment F1 | Notes |
|------|-----|-------|--------|---------------|-----------|--------------|-------|
| 2026-11-14 | team | TF-IDF + LogReg | word 1-2-grams, class_weight=balanced, seed 42; 1,000 generated rows, 700/150/150 | 0.969 (acc 0.967) | 0.471 (keyword rules) | 0.156 (keyword rules) | Templated data, so inflated; see docs/BASELINE_METRICS.md |
| 2026-11-14 | team | same model, real text | scored on 186 distinct BBMP phrases | acc 0.41 all / 0.55 non-"other" / 0.91 frequency-weighted | - | - | Real-text sanity check; BBMP mapping is noisy |

| 2026-11-14 | team | IndicBERT (ai4bharat/IndicBERTv2-MLM-only) | configs/indicbert.yaml, 4 epochs, bs16, lr 3e-5, class weights on, T4 | 0.994 (acc 0.993) | 0.889 | 1.000 | Best run; en/mr 1.00, hi 0.985. Generated test split, so inflated |
| 2026-11-14 | team | mBERT | configs/mbert.yaml, same settings | 0.969 (acc 0.967) | 0.907 | 0.984 | Hindi weakest (0.93); en/mr 1.00. Ties the TF-IDF baseline on department |
| 2026-11-14 | team | MuRIL | configs/muril.yaml, same settings | 0.694 (acc 0.687) | 0.656 | 0.787 | Undertrained: val dept F1 0.07, 0.28, 0.55, 0.69 over 4 epochs and still rising. Marathi 0.54, Hindi 0.71, English 0.75. Rerun with more epochs / higher lr |
| 2026-11-14 | team | MuRIL, class_weighting=false (ablation) | same as MuRIL, only class_weighting changed | 0.074 (acc 0.227) | 0.153 | 0.309 | Never left the initial state in 4 epochs (val F1 flat at 0.072). Not a clean weighting ablation because MuRIL itself is undertrained; repeat on IndicBERT |
