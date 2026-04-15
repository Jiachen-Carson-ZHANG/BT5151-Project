# Experiment 007: Class Imbalance Fix + Confusion Matrix + Enriched Payload

**Date:** 2026-04-10
**Run:** Run 11 (00:23, stage_full_20260410_002309.log)

## What changed (from Run 8 / Experiment 006)

1. **XGBoost class imbalance handling** — `compute_sample_weight('balanced', train_target)` passed to XGBoost's `.fit()`. LR and RF already use `class_weight='balanced'`.
2. **Confusion matrix** — added to `compute_multiclass_metrics()`, logged per-model in `evaluate_models_node`.
3. **Per-class metrics logging** — precision/recall/f1/support logged for each class per model.
4. **FE inf validation** — `validate_feature_engineering_output` now checks for `inf` values (not just NaN).
5. **Enriched column-transform-spec payload** — added `column_profiles` with top-10 values per categorical column and min/max/mean/p1/p99 per numeric column. Sample rows increased from 5 to 10.
6. **Enriched generate-preprocessing-code payload** — same column_profiles added.
7. **Skill prompt gotchas** — added `freq='M'`→`'ME'` and `mode()[0]` empty series warnings to both generate and repair preprocessing prompts.
8. **FE skill prompt** — added mandatory inf cleanup rule (rule 5).

## Results

### Preprocessing: PASSED (attempt 3)

| Attempt | Outcome | Issues |
|---------|---------|--------|
| 1 | Execution crash | `str.extract` 2-group → single column (recurring) |
| 2 | Quality FAIL | 3 issues: Type_of_Loan delimiter artifacts, Annual_Income outlier, negative Num_of_Delayed_Payment |
| 3 | Quality PASS | 0 issues |

**Improvement from enriched payload:** First audit found only 3 issues (down from 8-11 in previous runs). The column-transform-spec now sees garbage values in top_10_values and outliers in p1/p99, so the spec includes cleanup rules from the start. The generate-preprocessing-code also sees the profiles, producing better first-pass code.

### Feature engineering: PASSED first try

- 56 → 58 columns (added 2 features)
- All validation checks green, including new inf checks
- `no_infs_in_train: True, no_infs_in_test: True`

### Model training: ALL 3 MODELS TRAINED

| Model | accuracy | macro_f1 |
|-------|----------|----------|
| logistic_regression | 0.6301 | 0.6258 |
| random_forest | 0.6970 | 0.6680 |
| **xgboost** | **0.6877** | **0.6820** |

### Confusion matrices

**Logistic Regression:**
|  | Pred Good | Pred Poor | Pred Standard |
|--|-----------|-----------|---------------|
| True Good | 2862 | 166 | 666 |
| True Poor | 728 | 3867 | 1131 |
| True Standard | 2245 | 2461 | 5874 |

**Random Forest:**
|  | Pred Good | Pred Poor | Pred Standard |
|--|-----------|-----------|---------------|
| True Good | 2040 | 85 | 1569 |
| True Poor | 305 | 3812 | 1609 |
| True Standard | 1111 | 1381 | 8088 |

**XGBoost (with sample_weight):**
|  | Pred Good | Pred Poor | Pred Standard |
|--|-----------|-----------|---------------|
| True Good | 2939 | 124 | 631 |
| True Poor | 530 | 4329 | 867 |
| True Standard | 1971 | 2123 | 6486 |

### Per-class metrics

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| LR | Good | 0.490 | 0.775 | 0.601 |
| LR | Poor | 0.595 | 0.675 | 0.633 |
| LR | Standard | 0.766 | 0.555 | 0.644 |
| RF | Good | 0.590 | 0.552 | 0.571 |
| RF | Poor | 0.722 | 0.666 | 0.693 |
| RF | Standard | 0.718 | 0.764 | 0.740 |
| **XGB** | **Good** | **0.540** | **0.796** | **0.644** |
| **XGB** | **Poor** | **0.658** | **0.756** | **0.704** |
| **XGB** | **Standard** | **0.812** | **0.613** | **0.699** |

### XGBoost class imbalance impact

| Metric | Before (no balancing) | After (sample_weight) |
|--------|----------------------|----------------------|
| macro_f1 | 0.6692-0.6736 | **0.6820** |
| Good recall | ~0.55 | **0.796** |
| Poor recall | ~0.67 | **0.756** |

The sample_weight fix dramatically improved recall for minority classes. XGBoost now has the best macro_f1 across all models, confirming it was being held back by class imbalance.

### Inference: PASSED

- Row 42 → predicted: Good, confidence: 0.7086 (was 0.5217 before balancing)
- Higher confidence reflects better calibration after class balancing

## Token usage

Total: 79,513 tokens (10 LLM calls). Column-transform-spec input tokens increased from ~3k to ~8k due to enriched payload — a worthwhile trade for fewer repair rounds.

## Insights

1. **Enriched payload reduces repair rounds.** 3 attempts vs 4+ previously. The column-transform-spec now sees garbage values and outliers directly, producing better specs. The audit found only 3 issues on first review (vs 8-11 before).

2. **Class imbalance fix works.** XGBoost macro_f1 improved by ~1.3pp. More importantly, minority class recall jumped significantly (Good: +24pp, Poor: +9pp). This is the right trade-off for a credit risk classifier — missing Good customers is worse than slightly over-predicting them.

3. **Model narrative is clear.** LR is the weakest (expected — linear model on non-linear data). RF and XGBoost are competitive. XGBoost wins on macro_f1, RF wins on Standard class F1. This gives us a strong comparison narrative for the report.

4. **Inf validation caught nothing this run.** But it's protective — without it, the previous run would have passed FE validation and crashed at training (as it did in Run 9).

## Next steps

From lab/analysis/improving-model-performance.md:
1. ~~XGBoost class imbalance~~ DONE
2. ~~Confusion matrix + per-class logging~~ DONE
3. SHAP for XAI (architectural change — next priority)
4. ~~Enrich column-transform-spec payload~~ DONE
5. Feature engineering improvements (more transforms)
6. Hyperparameter tuning (optional)
