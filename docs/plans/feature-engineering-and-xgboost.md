# Plan: Feature Engineering Node + XGBoost

**Date:** 2026-04-09
**Status:** Pending review

## Context

The pipeline currently goes straight from preprocessing → training with no feature engineering. LR macro_f1=0.617 and RF macro_f1=0.667 on 62 features. There's room for improvement by pruning noise features, handling correlations, and creating domain-meaningful interactions. We also want XGBoost as a third model (3 distinct families: linear, bagging, boosting).

## Architecture change

```
Current:  ... → review-preprocessing-quality →(pass)→ train-models → ...
Proposed: ... → review-preprocessing-quality →(pass)→ generate-feature-engineering-code
          → inspect-feature-engineering-code →(pass)→ execute-feature-engineering
                                             →(fail)→ repair-feature-engineering-code → (loop back to inspect)
          → validate-feature-engineering →(pass)→ train-models
                                         →(fail)→ repair-feature-engineering-code → (loop)
```

Max 3 attempts (1 generate + 2 repairs). Lighter than preprocessing's 5 because input is already clean.

## Key design decisions

1. **Codegen approach**: Free codegen with heuristic rules in skill prompt (not pure template). Rationale: heuristic rules guarantee basics (drop constants, handle correlations), LLM adds domain reasoning (which ratios to create, which columns to log-transform).

2. **Function signature**: `def engineer_features(train_df, test_df, workspace_path)`. Train+test together ensures the LLM fits statistics on train only and applies to test — prevents leakage. Writes `engineered_train.csv`, `engineered_test.csv`, `feature_engineering_report.json`.

3. **Execution**: Same subprocess pattern as preprocessing. Reuse `inspect_preprocessing_code` for AST checks (it already parameterizes entrypoint name).

4. **State**: Overwrite `train_frame`/`test_frame`/`feature_columns` in place. No new frame fields — downstream nodes don't change. Add only loop-tracking fields.

5. **No LLM quality review**: Unlike preprocessing, no audit step. Validation is structural only (row counts match, no NaNs, features non-empty). Feature engineering failures are less catastrophic than cleaning failures.

## Files to create

| File | Purpose |
|------|---------|
| `skills/generate-feature-engineering-code.md` | Skill prompt with heuristic rules + domain reasoning |
| `skills/repair-feature-engineering-code.md` | Repair skill for failed FE code |
| `src/bt5151_credit_risk/feature_engineering.py` | generate, execute, validate, repair functions |

## Files to modify

| File | Change |
|------|--------|
| `src/bt5151_credit_risk/state.py` | Add 5 fields: `feature_engineering_code`, `feature_engineering_code_review`, `feature_engineering_execution_log`, `feature_engineering_validation_report`, `feature_engineering_attempt_count` |
| `src/bt5151_credit_risk/graph.py` | Add 5 nodes, 2 routing functions, change quality-review success target from `train-models` to `generate-feature-engineering-code` |
| `src/bt5151_credit_risk/train.py` | Add XGBClassifier as third model |
| `requirements.txt` | Add `xgboost` |
| `tests/test_graph.py` | Add FE nodes to expected set, monkeypatch FE functions in e2e test |
| `tests/test_state.py` | Add new fields to expected fields test |
| `docs/architecture/current-state.md` | Update pipeline diagram and key modules table |

## Skill prompt: generate-feature-engineering-code.md

Heuristic rules the LLM must always apply:
1. Drop constant/near-constant features (nunique ≤ 1)
2. Drop one of highly correlated pairs (|r| > 0.95, keep higher variance)
3. Log-transform right-skewed continuous columns (|skewness| > 2)
4. Consider domain-meaningful interaction features (ratios, products)
5. Report what was added/removed/transformed and why

Critical rules (same as preprocessing):
- No `inplace=True`
- No forbidden imports
- Fit all statistics on train_df only, apply to test_df
- After creating features, check for NaN/inf

## Validation checks

| Check | Type |
|-------|------|
| `train_file_exists` | hard error |
| `test_file_exists` | hard error |
| `report_file_exists` | hard error |
| `train_row_count_match` | hard error |
| `test_row_count_match` | hard error |
| `no_nans_introduced` | hard error |
| `features_non_empty` | hard error |
| `max_feature_cap` | hard error (output cols ≤ 5x input cols) |

## XGBoost

Add to `train.py:build_candidate_models()`:
```python
"xgboost": XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_SEED,
)
```

No `class_weight` — deliberate choice for model diversity (LR and RF already use balanced weights). XGBoost implements sklearn API so no changes to training/evaluation/inference nodes.

## Implementation order

1. `requirements.txt` + install xgboost + `train.py` + update train test
2. `state.py` — add FE state fields
3. `feature_engineering.py` — generate, execute, validate, repair functions
4. `skills/generate-feature-engineering-code.md` + `skills/repair-feature-engineering-code.md`
5. `graph.py` — add nodes, routing, rewire edges
6. Update tests (`test_graph.py`, `test_state.py`)
7. Update `docs/architecture/current-state.md`
8. Full pipeline run + experiment record

## Verification

1. `pytest tests/` — all existing + new tests pass
2. Full pipeline run: `PYTHONPATH=src .venv/bin/python3 run_stage.py full 42`
3. Check log for: FE node executes, features change (count/names), model metrics, no crashes
4. Create experiment record in `lab/experiments/004-*.md`
