# Improving Model Performance: Full Picture

**Date:** 2026-04-09
**Status:** All 6 levers now implemented as of Run 009. See experiment records 007-009 for results.

## Current metrics (Run 7, before improvements)

| Model | accuracy | macro_f1 |
|-------|----------|----------|
| LR | 0.6301 | 0.6265 |
| RF | 0.6955 | 0.6696 |
| XGBoost | 0.6979 | 0.6736 |

## Why are they "low"?

For a 3-class problem with 53%/29%/18% imbalance on a noisy Kaggle dataset with no tuning, 0.67 macro_f1 is actually reasonable. But there's clear room for improvement.

## Five levers for improvement (ordered by expected impact)

### 1. Hyperparameter tuning (HIGH impact)
All models use hardcoded params. Even basic RandomizedSearchCV could push macro_f1 up 2-5pp.
- See: `lab/analysis/hyperparameter-tuning-approach.md`

### 2. Better data cleaning at source (HIGH impact)
Noise survives preprocessing because the spec node lacks data visibility. Enriching the column-transform-spec payload (top-10 values, min/max/std) would catch garbage earlier.
- See: `lab/analysis/noisy-data-responsibility-chain.md`

### 3. XGBoost class imbalance handling (MEDIUM impact)
LR and RF use `class_weight='balanced'`. XGBoost doesn't. Fix: compute `sample_weight` from class frequencies in the training node.
```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', state.train_target)
model.fit(train_frame, train_target, sample_weight=sample_weights)
```
This requires the training node to pass `sample_weight` to `.fit()` for models that support it.

### 4. Serious feature engineering (MEDIUM impact)
Current FE only added 2 ratios. Improvements:
- Pass richer context (correlation matrix summary, skewness values)
- Require minimum transforms ("apply at least 3 of 5 heuristic rules")
- Add FE quality review (like preprocessing audit, but for features)
- Consider: polynomial features for top-5 important features, binning continuous variables

### 5. Missing evaluation metrics (LOW impact on model, HIGH impact on rubric)
- **Confusion matrix** — rubric explicitly requires it, we don't have it
- **AUC-ROC** (one-vs-rest) — better than accuracy for imbalanced classes
- **Per-class metrics logging** — we compute them but only log summary numbers

## What NOT to do

- **Don't over-tune.** The rubric cares about fair comparison and interpretation, not winning a Kaggle competition.
- **Don't add 10 models.** 3 models from 3 families is the sweet spot for the report narrative.
- **Don't do model stacking/ensembling.** Adds complexity without improving interpretability — the opposite of what the rubric rewards.

## Priority order for implementation

1. XGBoost class imbalance (quick fix, training node only)
2. Confusion matrix + per-class logging (quick fix, evaluation node)
3. SHAP for XAI (architectural change, high rubric value)
4. Richer column-transform-spec payload (reduces repair rounds)
5. Feature engineering improvements (more transforms, quality review)
6. Hyperparameter tuning (optional, highest performance impact but time-consuming)
