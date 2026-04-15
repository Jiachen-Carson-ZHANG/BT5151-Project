# Experiment 010: XAI Overhaul First Run (pre-optimization)

**Date**: 2026-04-14
**Status**: KILLED — ran on old code before training/SHAP optimizations
**Baseline**: Run 009 (XGB macro_f1=0.802, 188 min total, RF alone 2h26m)

## What changed since 009

This was the first run after the XAI overhaul (5 new nodes: generate-eda-hypotheses, training-diagnostics, global-xai, local-xai, package-analysis-bundle). However, it ran on code **before** the training speed and SHAP optimization fixes were applied.

## Pipeline timing breakdown

| Stage | Start | End | Duration |
|-------|-------|-----|----------|
| dataset-policy-spec | 09:55:35 | 09:55:40 | 5s |
| EDA | 09:55:40 | 09:55:50 | 10s |
| **generate-eda-hypotheses** (NEW) | 09:55:50 | 09:56:13 | 23s |
| column-transform-spec | 09:56:13 | 09:56:51 | 38s |
| generate-preprocessing-code | 09:56:51 | 09:57:14 | 23s |
| inspect + execute + validate (attempt 1) | 09:57:14 | 09:57:38 | 24s |
| repair + re-execute + validate (attempt 2) | 09:57:38 | 09:58:34 | 56s |
| generate-feature-engineering-code | 09:58:34 | 09:59:47 | 73s |
| **train-models** | 09:59:47 | 13:33:58 | **3h 34m** |
| evaluate-models | 13:33:58 | 13:34:12 | 14s |
| **training-diagnostics** (NEW) | 13:34:12 | 13:34:50 | 38s |
| **select-model (SHAP)** | 13:34:50 | 15:01:20 | **1h 26m** |
| **global-xai** (NEW, SHAP again) | 15:01:20 | 15:02:50 | ~1.5 min |
| (killed after global-xai) | | | |
| **Total before kill** | 09:55 | ~15:03 | **~5h 8m** |

### Training time breakdown
- LR tuning: 10:02:15 → 10:06:03 = **3m 48s** (15 trials, fast)
- RF tuning: 10:06:05 → 10:39:01 = **33 min** (15 trials, n_estimators=200-600, max_depth=10-30, no cap)
- RF final retrain: 10:39:01 → 10:40:25 = 1m 24s
- XGB tuning: 10:40:25 → 13:31:33 = **2h 51m** (15 trials, n_estimators=1000 ceiling, max_depth=6-15, no early stopping patience reduction)
- XGB final retrain + learning curve: 13:31:33 → 13:33:58 = 2m 25s

### SHAP time breakdown
- select-model SHAP (RF, metric-best): 13:34:50 → 14:59:25 = **1h 24m** (500 samples, RF depth=26)
- select-model SHAP recompute (XGB, LLM selected different model): 14:59:30 → 15:01:20 = ~2 min
- global-xai SHAP (XGB, redundant recompute): 15:01:20 → 15:02:50 = ~1.5 min

## Model results

| Model | CV score | Test macro_f1 | Test accuracy | CV-Test gap |
|-------|----------|---------------|---------------|-------------|
| Logistic Regression | 0.6297 | 0.6365 | 0.6425 | +0.7pp |
| Random Forest | 0.7731 | 0.6774 | 0.6951 | **-9.6pp** |
| XGBoost | 0.8084 | 0.6757 | 0.6928 | **-13.3pp** |

**Selected model**: XGBoost (LLM chose over RF due to smaller CV-test gap and better generalization, despite RF having slightly higher test macro_f1).

### Per-class performance (XGBoost)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Good | 0.573 | 0.646 | 0.607 | 3,694 |
| Poor | 0.687 | 0.707 | 0.697 | 5,726 |
| Standard | 0.747 | 0.702 | 0.723 | 10,580 |

### Confusion flow (XGBoost)
- Good→Standard: 32.97% (dominant error for Good)
- Poor→Standard: 22.7% (dominant error for Poor)
- Standard→Poor: 16.6%, Standard→Good: 13.3% (bidirectional bleed)

## New nodes audit

### generate-eda-hypotheses (o4-mini, 23s)
**Quality: GOOD** — Produced well-structured three-tier hypotheses grounded in EDA statistics.

Tested predictions (3):
1. "XGBoost will outperform LR by ≥5pp macro_f1 (≥0.78 vs ≤0.73)" — **refuted** (gap was 3.92pp, both models lower than predicted)
2. "Total_EMI_per_month will rank #1 in global SHAP" — **inconclusive** (SHAP not available at validation time)
3. "Binary missing indicator for Type_of_Loan will increase macro_f1 by ≥2pp" — **inconclusive** (feature not added)

Supported conjectures (2):
1. "Standard class will have lowest recall (≤0.65)" — **partially confirmed** (Standard recall=0.702, not the lowest — Good was lowest at 0.646)
2. "LR will rank Delay_from_due_date as top coefficient" — not validated

Exploratory lead (1):
1. "Total_EMI_per_month/Monthly_Inhand_Salary ratio will have MI>0.60" — FE did create EMI_to_salary_ratio, but it didn't appear in SHAP top-10

**Issue**: Predictions were too optimistic (macro_f1 ≥0.78 predicted, 0.68 actual). The hypotheses assumed features were more separable than they are.

### training-diagnostics (o4-mini, 30s)
**Quality: GOOD** — Correctly diagnosed overfitting in both RF and XGB, identified capacity limitation in LR.

Key findings:
- RF: "deep trees (max_depth=26, min_samples_split=3) memorize training but generalize poorly" — correct diagnosis
- XGB: "growing train–val logloss gap (0.106 vs 0.459) across 1000 rounds" — correctly identified non-convergence
- LR: "C=0.0017 imposes heavy regularization" — correct underfitting diagnosis
- Validated 1 of 3 EDA hypotheses, correctly marked 2 as inconclusive

### global-xai (partial — killed during/after)
SHAP top-10: Outstanding_Debt, Interest_Rate, Credit_Mix_Good, Credit_Mix_Standard, delay_inquiry_product, Delay_from_due_date, Payment_of_Min_Amount_No, Num_Credit_Card, Num_Credit_Inquiries, Changed_Credit_Limit

**Notable**: Total_EMI_per_month and Monthly_Inhand_Salary (top-2 MI features) are NOT in SHAP top-10. This suggests the log1p transform may have destroyed their signal, or the model is relying on different feature interactions. EMI_to_salary_ratio (engineered from these) also absent.

## Root causes identified

### 1. Training time (3h 34m)
- **RF**: 15 trials with n_estimators=200-600 (LLM grid included n_estimators), max_depth up to 30 (no cap). Best params: n_estimators=400, max_depth=26.
- **XGB**: 15 trials, n_estimators=1000 ceiling with early_stopping_rounds=50 is correct, but max_depth=6-15 range + 80k rows made each fit slow. Best stopped at round 999/1000 — didn't actually early-stop.

### 2. SHAP time (1h 28m)
- TreeExplainer on RF with max_depth=26 = O(2^26) = 67M paths per tree × 400 trees × 500 samples. This is the dominant cost.
- SHAP computed 3 times: once for RF (metric-best), once for XGB (LLM selected different model), once in global-xai (redundant).

### 3. Model performance (macro_f1 ~0.68)
- Severe overfitting: RF CV=0.77→test=0.68 (-9.6pp), XGB CV=0.81→test=0.68 (-13.3pp)
- XGB never early-stopped (ran all 1000 rounds, best at round 999) — val loss still decreasing but train-val gap huge
- Top MI features (Total_EMI_per_month, Monthly_Inhand_Salary) not appearing in SHAP — possible signal loss from log1p transform or poor feature engineering

## Fixes applied after this run

All fixes are in the codebase but were NOT active during this run:

1. **Training speed**: Subsample to 15k rows for tuning CV, RF uses 100 trees during tuning (200 for final), max_depth capped at 20, n_estimators stripped from RF+XGB grids, trials reduced to 10, XGB early stopping patience=30 during tuning
2. **SHAP speed**: select_model_node uses compute_global_shap() instead of inline code, passes full result to global_xai_node via state (eliminates redundant recomputation), PFI subsamples to 5k rows
3. **SHAP correctness**: global_xai_node reuses SHAP from select_model instead of recomputing

## Comparison to Run 009

| Metric | Run 009 | Run 010 | Delta |
|--------|---------|---------|-------|
| Best model | XGBoost | XGBoost | same |
| Best macro_f1 | 0.802 | 0.677 | **-12.5pp** |
| RF macro_f1 | 0.681 | 0.677 | -0.4pp |
| XGB macro_f1 | 0.802 | 0.676 | **-12.6pp** |
| Total runtime | ~188 min | ~308 min (killed) | +120 min |
| RF tuning time | 146 min | 33 min | -113 min |
| XGB tuning time | ? | 171 min | — |
| SHAP time | ~41 min | ~88 min | +47 min |
| New nodes overhead | N/A | ~92s | — |

**The massive macro_f1 regression (0.802 → 0.677) needs investigation.** Run 009 achieved 0.802 on XGBoost — this run's 0.677 with similar data and more features (62 vs ?) suggests something went wrong in preprocessing or FE. Possible causes:
- Different preprocessing codegen output (LLM-generated code varies between runs)
- log1p transforms on key features destroying signal
- Feature engineering interactions not as useful as expected
- Different train/test split (grouped by Customer_ID)

## Next steps

1. Kill this run, apply all optimization fixes
2. Re-run with fixes — expect training <5 min, SHAP <2 min, total <30 min
3. Investigate macro_f1 regression: compare preprocessing outputs between run 009 and 010
4. Consider whether log1p transforms on top MI features are helping or hurting
