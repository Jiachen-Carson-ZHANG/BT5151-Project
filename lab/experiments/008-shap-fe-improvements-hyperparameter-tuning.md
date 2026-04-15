# Experiment 008: SHAP + FE improvements + Hyperparameter tuning

**Date**: 2026-04-10
**Log**: `logs/stage_full_20260410_032438.log`
**Seed**: 42
**Status**: PASSED (full e2e)

## Changes tested

1. **SHAP for XAI**: TreeExplainer for RF/XGBoost, LinearExplainer for LR Pipeline. Global importance (500 test samples) + per-prediction top-5 contributions grounding explain-risk LLM call.
2. **Feature engineering improvements**: Strengthened FE skill prompt with mandatory heuristic rules (drop constants, handle correlations, log-transform skewed, min 3 interaction features, mandatory inf cleanup, binning with safe edges, final NaN cleanup).
3. **Hyperparameter tuning**: LLM-reasoned grids via `reason-hyperparameter-grid` skill + `RandomizedSearchCV` (15 iterations, 5-fold stratified CV, scoring=f1_macro).
4. **Bug fixes**: subprocess TimeoutExpired bytes serialization, quality review tolerance (accept after 3 attempts), LR grid filtering (strip l1_ratio/penalty/solver — lbfgs only supports l2), preprocessing timeout increased to 120s.

## Pipeline stages

| Stage | Time | Notes |
|---|---|---|
| Preprocessing | ~6 min | 1 generate + 1 repair (audit found Type_of_Loan delimiter issue) |
| Quality review | pass on 2nd attempt | |
| Feature engineering | ~10s | 61 -> 64 cols (3 interaction features) |
| Training baseline | ~2.5 min | LR=0.6317, RF=0.6684, XGB=0.6810 |
| LR tuning | ~2 min | 3 iterations (only model__C), best_cv=0.6223 |
| RF tuning | ~111 min | 15 iterations, best_cv=0.7759, best_params={n_estimators=200, min_samples_split=5, max_depth=None} |
| XGB tuning | ~28 min | 15 iterations, best_cv=0.8148, best_params={subsample=0.9, n_estimators=300, max_depth=10, lr=0.1} |
| Evaluation | instant | |
| SHAP global | ~41 min | 500 samples on RF — very slow |
| Inference + SHAP | ~15s | Row 42 -> Good (confidence 0.8406) |
| Explain + Recommend | ~5s | |

**Total runtime**: ~188 min (3.1 hours)

## Results

| Model | Baseline macro_f1 | CV best | Test macro_f1 | Test accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.6317 | 0.6223 | 0.6317 | 0.6367 |
| Random Forest | 0.6684 | 0.7759 | 0.6814 | 0.6991 |
| XGBoost | 0.6810 | 0.8148 | 0.6781 | 0.6917 |

**Selected**: random_forest (macro_f1=0.6814)

### Per-class metrics (RF)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Good | 0.576 | 0.663 | 0.616 | 3694 |
| Poor | 0.700 | 0.689 | 0.695 | 5726 |
| Standard | 0.750 | 0.717 | 0.733 | 10580 |

### SHAP global top-10 features

1. Outstanding_Debt: 0.0372
2. Interest_Rate: 0.0344
3. Credit_Mix_Good: 0.0331
4. Debt_to_Income: 0.0292
5. Delay_from_due_date: 0.0281
6. Credit_Mix_Standard: 0.0269
7. Payment_of_Min_Amount_No: 0.0197
8. Payment_of_Min_Amount_Yes: 0.0195
9. Num_Credit_Card: 0.0169
10. Changed_Credit_Limit: 0.0151

### Inference (row 42)

- Predicted: Good (confidence 0.8406)
- SHAP top-5: Credit_Mix_Good (+0.084), Delay_from_due_date (+0.053), Credit_Mix_Standard (+0.045), Payment_of_Min_Amount_Yes (+0.041), Payment_of_Min_Amount_No (+0.040)
- Risk level: low, confidence band: high
- Action: standard_handling

## Observations

1. **CV vs test gap**: RF CV=0.7759 vs test=0.6814, XGB CV=0.8148 vs test=0.6781. Large gap suggests the group-based split creates harder test conditions than random CV folds. This is expected and desirable — grouped holdout prevents data leakage from same-customer rows.
2. **Tuning marginal benefit**: RF improved from 0.6684 to 0.6814 (+1.3pp) but took 111 min. XGB improved from 0.6810 to 0.6781 (-0.3pp) — tuning actually hurt slightly. Cost-benefit is poor.
3. **SHAP global computation is very slow**: 41 min for 500 samples on RF. Need to reduce sample count or use approximate methods.
4. **Feature engineering**: 3 interaction features added (61->64), no features dropped. Minimal impact but no harm.
5. **LR tuning useless**: Only 3 param combos for C, best CV was worse than baseline. LR is limited by model capacity, not hyperparameters.

## Action items

- [ ] Reduce SHAP global sample count (500 -> 100-200) or use approximate TreeExplainer
- [ ] Consider dropping LR tuning entirely — waste of compute
- [ ] Reduce RF/XGB tuning iterations (15 -> 8-10) or subsample training data for CV
- [ ] Profile RF tuning to understand why 15 iterations take 111 min (likely deep trees on 80k rows)
