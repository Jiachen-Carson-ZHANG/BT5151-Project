# Experiment 006: Full Pipeline End-to-End Pass

**Date:** 2026-04-10
**Run:** Run 8 (23:57, stage_full_20260409_235709.log)

## What changed (from Run 7 / Experiment 005)

1. **full_feature_frame rebuild** — `validate_feature_engineering_node` now reconstructs `full_feature_frame` by concatenating engineered train+test with original index alignment. Fixes the inference column mismatch crash.

## Results

### Preprocessing: PASSED (attempt 4)

| Attempt | Outcome | Issues |
|---------|---------|--------|
| 1 | Quality FAIL | 8 issues: Annual_Income NaN, garbage categories (Occupation________), negative counts, Interest_Rate 5797 |
| 2 | Quality FAIL | 10 issues: encoding artifacts persist (Type_of_Loan, Credit_Mix, Occupation), implausible max values in 6 numeric columns |
| 3 | Inspection FAIL | `inplace=True` violation |
| 4 | Quality PASS | 0 issues — reviewer correctly converged |

Note: Attempt 2 used an extra repair cycle because the reviewer flagged NEW distribution issues (implausible max values) that weren't in attempt 1. This is valid — they're legitimate `major/distribution_sanity` issues, not escalation. The convergence rules correctly allowed this.

### Feature engineering: PASSED first try

- 54 → 56 columns (added 2 features, dropped 0)
- All validation checks green
- No repairs needed

### Model training: ALL 3 MODELS TRAINED

| Model | accuracy | macro_f1 |
|-------|----------|----------|
| logistic_regression | 0.6435 | 0.6381 |
| random_forest | 0.6946 | 0.6663 |
| xgboost | 0.6937 | 0.6692 |

Compared to Run 7:
- LR: 0.6265 → 0.6381 (+1.2pp macro_f1)
- RF: 0.6696 → 0.6663 (-0.3pp macro_f1)
- XGBoost: 0.6736 → 0.6692 (-0.4pp macro_f1)

LR improved slightly, RF and XGBoost marginally lower — within noise for different preprocessing runs. The pattern is stable.

### Inference: PASSED

- Row 42 → predicted: Good, confidence: 0.5217
- No feature_names mismatch — full_feature_frame fix works

### Model selection: xgboost

Selected based on strongest macro_f1 and weighted_f1.

### Explanation + Recommendation

- Risk level: low, confidence band: medium
- Action: standard_handling

## Token usage

| Call | Model | Input | Output | Duration |
|------|-------|-------|--------|----------|
| dataset-policy-spec | gpt-4o | 4,153 | 145 | 3.4s |
| column-transform-spec | gpt-4o | 2,938 | 919 | 8.2s |
| generate-preprocessing | gpt-4o | 5,797 | 1,470 | 14.7s |
| audit-preprocessing (1) | gpt-4o | 11,433 | 651 | 11.7s |
| repair-preprocessing (1) | gpt-4o | 4,994 | 1,750 | 13.2s |
| audit-preprocessing (2) | gpt-4o | 10,954 | 834 | 7.3s |
| repair-preprocessing (2) | gpt-4o | 5,373 | 1,861 | 22.7s |
| repair-preprocessing (3) | gpt-4o | 5,549 | 1,794 | 21.7s |
| audit-preprocessing (3) | gpt-4o | 11,066 | 46 | 2.3s |
| generate-FE | gpt-4o | 10,841 | 929 | 10.8s |
| explain-risk | gpt-4o-mini | 720 | 74 | 3.4s |
| recommend-action | gpt-4o-mini | 815 | 41 | 2.2s |
| **Total** | | **74,633** | **10,514** | **121.5s** |

Total wall time: 348s (~5.8 min)

## Insights

1. **full_feature_frame fix works.** The inference column mismatch from Run 7 is resolved. Pipeline runs end-to-end.

2. **Preprocessing still takes 4 attempts.** The codegen produces reasonable first-pass code, but the audit consistently finds 8-10 issues. This is expected — the column-transform-spec sees limited data (5 rows), so garbage values and outliers slip through. Enriching the spec payload (top-10 values, min/max/std) would reduce repair rounds.

3. **FE codegen is reliable.** Second consecutive first-try pass. The heuristic rules in the skill prompt produce consistent, reasonable transforms.

4. **Metrics are stable across runs.** XGBoost macro_f1 hovers around 0.67, RF around 0.67, LR around 0.63. Variation between runs is ±0.5pp — attributable to different preprocessing code paths producing slightly different feature sets.

5. **inplace=True still slips through.** Despite the skill prompt warning, the LLM generated `inplace=True` in repair attempt 3. The AST inspector caught it. This is a known pattern — the warning reduces frequency but doesn't eliminate it.

## Status

**FIRST FULL END-TO-END PASS** — all nodes execute, all models train, inference succeeds, explanation generated.

## Next steps

Priority improvements (from lab/analysis/improving-model-performance.md):
1. XGBoost class imbalance handling (compute_sample_weight)
2. Confusion matrix + per-class logging in evaluation
3. SHAP for XAI (architectural change)
4. Enrich column-transform-spec payload (reduce repair rounds)
5. Feature engineering improvements
