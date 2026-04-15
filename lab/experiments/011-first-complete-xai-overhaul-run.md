# Experiment 011: First Complete XAI Overhaul Run

**Date**: 2026-04-15  
**Source log**: [logs/stage_full_20260415_010459.log](../../logs/stage_full_20260415_010459.log)  
**Status**: COMPLETED end-to-end, but not yet production-solid  
**Baseline comparisons**:
- Run 010: [010-xai-overhaul-first-run.md](./010-xai-overhaul-first-run.md)
- Run 009: [009-eda-reasoning-chain-optuna-early-stopping.md](./009-eda-reasoning-chain-optuna-early-stopping.md)

## What changed since Run 010

This is the first rerun after the major stabilization work:
- training-speed optimizations landed
- SHAP reuse landed
- FE prompt contract was tightened to preserve raw-value interactions
- grouped-entity inner validation was wired into training
- FE dual-view support was live in the runtime
- semantic-role validation and preprocessing repair escalation were active

So this run is the first fair test of whether the new architecture can complete the full 4-layer XAI loop.

## Pipeline flow summary

The pipeline executed the intended full chain:

1. `dataset-policy-spec`
2. programmatic EDA
3. `generate-eda-hypotheses`
4. `column-transform-spec`
5. preprocessing codegen + repair loop
6. FE codegen
7. training with grouped-entity validation
8. evaluation
9. training diagnostics
10. model selection
11. global XAI
12. local XAI casebook
13. XAI interpretation
14. analysis bundle packaging
15. single-row inference, explanation, recommendation

This is the first run where the full story executed end-to-end rather than stopping in training or global-XAI.

## Timing

Total runtime: **1566.3s (~26.1 min)**  
Pipeline completed at [log line 357](../../logs/stage_full_20260415_010459.log#L357).

Notable checkpoints:
- preprocessing reached FE after 3 attempts, but only by quality-review escape hatch
- FE completed on first attempt
- baseline metrics finished quickly
- tuning and final training completed
- SHAP reuse worked
- local-XAI, interpretation, bundle, inference, and recommendation all completed

## Model results

| Model | Test macro_f1 | Accuracy |
|---|---:|---:|
| Logistic Regression | 0.6369 | 0.6435 |
| Random Forest | 0.6789 | 0.6826 |
| XGBoost | **0.6833** | **0.6875** |

Selected model: **XGBoost**

Training-side notes:
- grouped-entity validation policy was active
- XGBoost tuning best CV score was **0.6714**
- early stopping still saturated at **best_n_trees=1000**

## What improved

### 1. The system actually completed

Compared with the previous failed rerun log (`stage_full_20260414_213430.log`), this run no longer died inside training. It completed the full pipeline and saved the analysis bundle.

### 2. Runtime is now in the right order of magnitude

Run 010 took ~5 hours before being killed. This run completed in ~26 minutes.  
The biggest wins came from:
- tuning-speed reductions
- grouped but deterministic inner validation
- SHAP reuse instead of redundant recomputation

### 3. FE semantics improved

The earlier log-ordering bug (log-transforming parent columns before ratio construction) appears resolved. The FE hypothesis and resulting downstream artifacts were coherent enough to let training and XAI complete normally.

### 4. The 4-layer XAI architecture is now operational

This run produced:
- EDA hypotheses
- training diagnostics
- global XAI evidence
- local XAI casebook
- cross-method interpretation
- packaged analysis bundle

That means the architectural shape is now real, not just designed.

## What is still not solid

### 1. Preprocessing still did not converge cleanly

The repair loop hit repeated quality issues and then moved forward anyway:
- [Annual_Income implausibly high max](../../logs/stage_full_20260415_010459.log#L195)
- [Credit_History_Age collapsed to a constant](../../logs/stage_full_20260415_010459.log#L196)
- [accepting and moving on](../../logs/stage_full_20260415_010459.log#L198)

These were not just cosmetic:
- `Credit_History_Age` was later dropped by FE as constant
- `Annual_Income` still carried a 24M-scale tail

So this run proves the pipeline can complete, but not yet that preprocessing is trustworthy.

### 2. Global XAI was still partial

SHAP reuse succeeded, but:
- grouped PFI failed completely at [line 314](../../logs/stage_full_20260415_010459.log#L314)
- PDP failed for one integer feature at [line 315](../../logs/stage_full_20260415_010459.log#L315)
- final methods used were only `['shap', 'pdp']` at [line 317](../../logs/stage_full_20260415_010459.log#L317)

So the global-XAI layer ran, but it was not method-complete.

### 3. Model quality is still far below Run 009

Performance improved slightly over Run 010:
- Run 010 XGB macro_f1: **0.6757**
- Run 011 XGB macro_f1: **0.6833**

But it is still nowhere near Run 009:
- Run 009 XGB macro_f1: **0.8017**

So the architecture is more stable, but the old performance level has not been recovered.

## Hypothesis-chain highlights

### Confirmed
- EDA predicted XGBoost would beat Logistic Regression by ≥4pp macro_f1
- Actual result: **0.6833 vs 0.6369**, gap **4.42pp**

### Refined / challenged
- FE expected `EMI_to_sal_ratio` to become a top SHAP feature
- The selected-model justification and XAI evidence still centered more strongly on `Outstanding_Debt`, `Interest_Rate`, and related signals

This suggests the EMI-driven story is still weaker in practice than the EDA/FE chain expected.

## Comparison summary

| Comparison | Verdict |
|---|---|
| vs previous failed rerun | Much better operationally — now completes end-to-end |
| vs Run 010 | Faster, cleaner, slightly better metrics, full bundle/explanation path works |
| vs Run 009 | Still materially worse on model quality |

## Immediate next actions

1. Tighten preprocessing convergence so major audit failures cannot slip through
2. Fix global-XAI robustness (PFI index alignment, PDP dtype handling)
3. Reassess why current performance remains in the ~0.68 range despite the stronger architecture

## Post-run fixes applied after this experiment

After this run, the following hardening work was implemented:
- preprocessing quality escape hatch now accepts only **minor** residual issues
- grouped PFI subsamples by **position**, not index label
- PDP casts integer features to float before grid construction
- preprocessing prompts were strengthened for:
  - percentile clipping on heavy-tail numeric columns
  - connector-tolerant duration parsing
