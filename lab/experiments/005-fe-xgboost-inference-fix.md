# Experiment 005: FE + XGBoost — Inference Column Mismatch

**Date:** 2026-04-09
**Run:** Run 7 (22:36, failed at inference)

## What changed (from Run 6 / Experiment 004)

1. **Audit follow-up convergence tightened** — new issues on follow-up must match defined critical categories only; added "anticipate repair side effects" to first review

## Results

### Preprocessing: PASSED (attempt 4, convergence fix worked)

| Attempt | Outcome | Issues |
|---------|---------|--------|
| 1 | Execution crash | `str.extract` 2-group → single column (recurring) |
| 2 | Quality FAIL | 8 issues: Age/Annual_Income critical, garbage categories, negative counts |
| 3 | Quality FAIL | 3 major: Occupation_, Credit_Mix_, Payment_Behaviour_98 stubs |
| 4 | Quality PASS | 0 issues — convergence fix working |

Key: Round 4 passed with `Credit_History_Age` entirely NaN (100k NaNs) — reviewer correctly did not escalate this on follow-up. The FE node later dropped it as constant.

### Feature engineering: PASSED first try

- Generated on first attempt, no repairs needed
- All validation checks green: row counts, no NaNs, column alignment, feature cap
- 64 → 65 columns: dropped `Credit_History_Age` (constant), added `debt_to_income_ratio` and `emi_to_salary_ratio`

### Model training: ALL 3 MODELS TRAINED

| Model | accuracy | macro_f1 |
|-------|----------|----------|
| logistic_regression | 0.6301 | 0.6265 |
| random_forest | 0.6955 | 0.6696 |
| **xgboost** | **0.6979** | **0.6736** |

XGBoost wins — as expected for tabular data. Marginal improvement over RF (+0.4pp macro_f1).

### Crash: inference feature_names mismatch

XGBoost's `predict_proba` at `run_inference_node` failed because `full_feature_frame` still had pre-FE columns (64 cols including Credit_History_Age, missing the 2 new ratios). The FE node overwrote `train_frame`/`test_frame` but not `full_feature_frame`.

## Fix applied

`validate_feature_engineering_node` now rebuilds `full_feature_frame` by concatenating engineered train+test with correct index alignment.

## Insights

1. **Convergence fix works.** No escalation on follow-up — reviewer correctly passed after 3 repairs despite Credit_History_Age being entirely NaN. Previous run (006) would have escalated this.

2. **FE codegen is reliable.** Passed first try with sensible transforms (dropped constant, added 2 domain ratios). Lower risk than preprocessing codegen as predicted.

3. **XGBoost integrates cleanly.** sklearn API compatibility means zero changes to training/evaluation code. Only issue was inference column mismatch — an infrastructure bug, not a model bug.

4. **State consistency is tricky.** When the FE node overwrites train/test frames, ALL downstream state that references the feature set must be updated. `full_feature_frame` was the only one missed, but this pattern could recur if we add more nodes.

## Next steps

- Rerun to verify the full_feature_frame fix
- Compare metrics with/without FE to measure FE contribution
