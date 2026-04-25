# Architecture: Current State

Last updated: 2026-04-25

## Pipeline

LangGraph `StateGraph` with Pydantic `BaseModel` state (`CreditRiskState`).

26-node pipeline with EDA, hypothesis generation, two conditional repair loops (preprocessing and feature engineering), training diagnostics, split XAI interpretation (global + local casebook), analysis bundle packaging, and a reasoning chain from EDA through to final explanation:

```
dataset-policy-spec
  → exploratory-data-analysis (programmatic — correlations, class separability, skewness, missing patterns, cardinality, raw discriminative features, model-eligible discriminative features, leakage alerts)
  → generate-eda-hypotheses (LLM — three-tier directional hypotheses from EDA statistics)
  → column-transform-spec (receives EDA insights, produces reasoning per column)
  → generate-preprocessing-code
  → inspect-preprocessing-code
  →   [pass] execute-generated-preprocessing
  →   [fail] repair-preprocessing-code → (back to inspect)
  → validate-preprocessing-output (hardcoded structural checks)
  → review-preprocessing-quality (LLM quality review, two-mode)
  →   [both pass] generate-feature-engineering-code
  →   [either fail] repair-preprocessing-code → (back to inspect)
  → inspect-feature-engineering-code
  →   [pass] execute-feature-engineering
  →   [fail] repair-feature-engineering-code → (back to inspect)
  → validate-feature-engineering (structural + lineage/formula replay checks)
  →   [pass] train-models
  →   [fail] repair-feature-engineering-code → (back to inspect)
  → evaluate-models
  → training-diagnostics (LLM — per-class struggle, capacity analysis, confidence summary + programmatic confidence stats, hypothesis validation)
  → select-model
  → global-xai (programmatic — SHAP beeswarm/dependence, grouped PFI, complementary PDP + ALE on the same top continuous features when feasible)
  → local-xai (programmatic — casebook: representative + borderline + worst misclassification per class)
  → interpret-global-xai (LLM — cross-method consensus, feature-effect shapes, global hypotheses)
  → interpret-local-xai (LLM — per-class stories, failure patterns, boundary analysis; receives global XAI interpretation as context)
  → package-analysis-bundle (serialize to disk, pass full semantic bundle through to explain-risk)
  → run-inference
  → explain-risk (LLM — grounded in analysis bundle summary and emits both explanation + recommended action)
```

## Key modules

| Module | Responsibility |
|--------|---------------|
| `state.py` | Pydantic state schema for the full pipeline (~60 fields) |
| `config.py` | Constants: target column, group column, seed, test size, default model |
| `profile.py` | Builds dataset profile (row count, target distribution, missing counts) |
| `eda.py` | Programmatic EDA: correlations, ANOVA, mutual information, skewness, missing patterns, cardinality, raw/model-eligible MI split |
| `feature_eligibility.py` | Filters raw high-MI fields into model-eligible discriminative features plus leakage/review alerts |
| `hypotheses.py` | LLM-driven analytical interpretation: EDA hypotheses, training diagnostics, global XAI interpretation, local XAI interpretation |
| `preprocess.py` | LLM-driven codegen, static inspection, isolated execution, artifact validation (structural + semantic role contract), LLM quality review, repair with escalation |
| `semantic_cleaning.py` | Reusable deterministic cleaning primitives for dirty numerics, ages, durations, missing-string handling, multi-hot membership, and group-then-global imputation |
| `schema_contracts.py` | Extracted deterministic preprocessing contract layer for semantic-role validation and cross-field invariants |
| `feature_engineering.py` | Deterministic production feature engineering, optional LLM-driven FE codegen, subprocess execution, single/dual-view validation, formula-lineage replay, repair/fallback |
| `deterministic_fe.py` | Extracted deterministic FE library: conservative ratio features, deferred categorical encoding, blocked-column filtering, artifact writing |
| `train.py` | Builds candidate models, Optuna Bayesian tuning, policy-aware validation splits, XGBoost early stopping, learning curves |
| `evaluate.py` | Multiclass metrics, deterministic metric-based model selection, LLM justification with hypothesis validation |
| `xai.py` | Global SHAP, grouped PFI, PDP, ALE, per-case SHAP, casebook strategy |
| `llm.py` | OpenAI API wrapper with JSON retry and markdown fence stripping |
| `business.py` | LLM-driven customer-facing explanation + recommended action in one call, grounded in the analysis bundle |
| `skill_prompts.py` | Loads skill markdown files as runtime system prompts |
| `codegen_audit.py` | Persists preprocessing / FE codegen attempts into stable run-scoped snapshot folders with redacted prompt payloads |
| `trace_events.py` | Structured `trace_events_<run_id>.jsonl` writer helpers for run lifecycle + node update artifacts |
| `ui_trace.py` | Parses structured trace artifacts or raw stage logs into Developer Trace markdown cards |
| `graph.py` | Node functions, routing logic, graph construction |

## EDA report contract

Programmatic EDA now keeps two discriminative-feature views instead of one:

- `raw_top_discriminative_features` — highest mutual-information fields before leakage/identifier filtering; audit-only context
- `model_eligible_top_discriminative_features` — highest-MI fields the downstream model is actually allowed to train on
- `top_discriminative_features` — backward-compatible alias for `model_eligible_top_discriminative_features`
- `leakage_alerts` — blocked or suspicious high-MI fields with severity/reason metadata

Downstream modeling prompts must use the model-eligible ranking for hypotheses, feature engineering, and model-selection reasoning. Raw high-MI identifier/group/leakage fields remain visible for audit, but they are not valid modeling opportunities.

## Preprocessing codegen loop

The LLM generates Python code that runs in a subprocess. The contract:

1. **Generate**: LLM produces `{"code": "...", "entrypoint": "run_preprocessing"}`
   - Each generate / repair attempt is now snapshotted under `lab/logs/codegen/<run_id>/preprocessing/<attempt_label>/`
   - Snapshot contents may include `generated.py`, `response.json`, `prompt_payload.json`, `metadata.json`, `execution_log.json`, `validation_report.json`, and `audit_report.json`
   - The preprocessing payload now includes `allowed_cleaning_primitives`, and the column transform spec may add additive primitive hints such as `primitive`, `primitive_params`, and `semantic_subtype`
2. **Inspect**: Static AST analysis blocks forbidden imports/calls and `inplace=True`
3. **Execute**: Code runs in isolated subprocess with 180s timeout; must write 5 artifacts
4. **Validate (structural + deterministic normalization + semantic role)**: Before reading the generated artifacts for validation, `validate_preprocessing_output()` now runs a deterministic normalization pass over the saved CSVs for the most failure-prone families using shared helpers from `semantic_cleaning.py`: `Age` is reparsed from the raw frame and clipped/imputed to a human range, `Credit_History_Age` is reparsed from the raw frame and capped by plausible adulthood tenure using the normalized age, `multi_value_set` columns such as `Type_of_Loan` have `_missing` recomputed from the raw source, `Not Specified` duplicate dummies removed, and sibling indicators zeroed when missingness fires, and kept numeric columns now have declared bounds re-enforced from the transform spec (`primitive_params.bounds`, `lower_bound` / `upper_bound`, or explicit cleaning text such as `clip to [low, high]`). Then validation checks artifact existence, target exclusion, feature frame non-empty, group overlap zero, and explicit failure on pandas-mangled duplicate column names (e.g. `.1` suffixes from broken multi-hot encoding). Finally it runs the extracted deterministic contract layer from `schema_contracts.py` — semantic-role checks plus cross-field invariants — and emits structured findings `{column, declared_role, violation, observed, expected, likely_cause}` as `role_violations` / `cross_field_violations` in the validation report. Any violation fails `passed`.
5. **Review (LLM quality)**: Two-mode review — comprehensive first pass, focused follow-up on subsequent rounds. Checks distribution sanity, encoding quality (garbage categories, delimiter artifacts, one-sided clipping), spec compliance, completeness
6. **Repair**: On failure at inspect, validate, or review, LLM receives error context and produces patched code; max 5 total attempts. The graph only accepts residual **minor** quality issues after repeated repair attempts — critical/major audit issues keep the repair loop active.
7. **Capability-ceiling escalation**: If the same `(column, violation)` pair appears in consecutive attempts, the repair call is routed to an escalated caller (`repair-preprocessing-code-escalated`) which resolves to a stronger model via `OPENAI_MODEL_REPAIR_PREPROCESSING_CODE_ESCALATED` env var. The escalation is logged as a `CAPABILITY CEILING` warning. The trigger resets after each repair so it re-arms only if the same violation persists.

## Semantic role contract

Every column in `column_transform_spec.transforms` declares:

- `semantic_role` — one of 12 enumerated roles: `identifier`, `group_identifier`, `target`, `numeric_continuous`, `numeric_count`, `ordered_categorical`, `unordered_categorical`, `binary_flag`, `multi_value_set`, `temporal_feature`, `free_text`, `leakage_risk_feature`.
- `representation_intent` — optional encoding choice when the role admits multiple valid encodings (e.g. `binary_membership` vs `count_membership` for `multi_value_set`; `one_hot` vs `target_encoded` for `unordered_categorical`).
- `primitive` / `primitive_params` / optional `semantic_subtype` — additive hints for deterministic preprocessing codegen. These do not replace `semantic_role`; they point codegen and repair toward approved runtime cleaning primitives such as `parse_dirty_numeric`, `parse_age_series`, `parse_duration_months`, `missing_string_mask`, `multi_hot_membership`, `fill_numeric_by_group_then_global`, and `cap_credit_history_by_adulthood`.

Role assignment is reasoning work (done by the reasoning model in the `column-transform-spec` node). Enforcement is deterministic (`validate_semantic_roles` in `preprocess.py`). Each role has a per-column output invariant checked after preprocessing codegen:

| Role | Invariant |
|---|---|
| `identifier`, `group_identifier`, `target`, `leakage_risk_feature` | Absent from feature frame |
| `binary_flag` | Values in {0, 1} |
| `multi_value_set` (binary_membership) | Indicators in {0, 1}, no count artifacts |
| `ordered_categorical` | Integer codes 0..K-1 preserving order |
| `numeric_count` | Values >= 0 |
| `numeric_continuous` | No NaN after imputation |

Violations produce structured findings with `likely_cause` that the repair prompt renders concretely. Repeated same-violation across attempts triggers model escalation (see ADR 0001).

## Feature engineering loop

Production/default mode is deterministic feature engineering (`BT5151_FEATURE_ENGINEERING_MODE=deterministic`, the default). The implementation now lives in `deterministic_fe.py`, which preserves validated preprocessing columns, drops any explicitly blocked identifier/leakage columns before FE, adds a conservative ratio set (`EMI_to_Salary_Ratio`, `Debt_to_Income_Ratio`, `Balance_to_Salary_Ratio`) when the parents exist, encodes deferred categoricals, fills numeric gaps with train medians, writes the normal FE artifacts, and proceeds directly into validation. This is the assignment/demo-safe path: training receives a validated feature matrix without depending on generated ratio code.

LLM feature engineering remains available as an explicit experiment by launching with `BT5151_FEATURE_ENGINEERING_MODE=llm`. In that mode, the LLM generates Python code for feature transforms, running in a subprocess. Max 3 attempts (1 generate + 2 repairs). The contract:

1. **Generate**: LLM produces `{"code": "...", "entrypoint": "engineer_features"}` with heuristic rules (drop constants, handle correlations, log-transform skewed, domain interactions)
   - Each generate / repair attempt is now snapshotted under `lab/logs/codegen/<run_id>/feature_engineering/<attempt_label>/`
2. **Inspect**: Reuses same AST analysis as preprocessing (forbidden imports, `inplace=True` ban)
3. **Execute**: Code runs in subprocess; takes `(train_df, test_df, workspace_path)`, writes either:
   - legacy single-view artifacts: `engineered_train.csv`, `engineered_test.csv`, `feature_engineering_report.json`
   - or dual-view artifacts: `engineered_train_linear.csv`, `engineered_test_linear.csv`, `engineered_train_tree.csv`, `engineered_test_tree.csv`, `feature_engineering_report.json`, `view_metadata.json`
   - The executor now injects the repository `src/` path into both `sys.path` and subprocess `PYTHONPATH`, so generated FE code can import stable project helpers (for example `bt5151_credit_risk.semantic_cleaning`) without relying on the caller shell environment.
4. **Validate**: Structural checks — row counts match, no NaNs, no infs, features non-empty, column alignment, max feature cap (≤5x input), and no remaining string/category/object columns. Boolean dummy columns are accepted as valid numeric model inputs. In dual-view mode, these checks run separately for each view declared in `view_metadata.json`. The validator also requires `feature_lineage.json` and replays engineered formulas from `feature_engineering_report.json` against the pre-FE training frame when formulas are available. This catches wrong math (for example log-transformed parents in debt-service ratios) while allowing explicit formulas such as `(Monthly_Inhand_Salary - Total_EMI_per_month) / Monthly_Inhand_Salary`, denominator offsets, and percent scaling. FE validation now also enforces `blocked_columns_absent`: any column dropped/quarantined by the transform spec, or carrying identifier/group/target/leakage roles, must stay absent from engineered outputs.
5. **Repair / fallback**: On failure, LLM receives error context and produces patched code. The subprocess also receives `deferred_categorical_columns` as a module global so generated code can deterministically encode object columns such as `Occupation`. If all FE repair attempts are exhausted, the graph emits the same deterministic safe FE program used by production/default mode. This fallback is logged and prevents invalid generated FE artifacts from entering training.

No LLM quality review — validation is structural only. Feature engineering failures are lower risk than data cleaning failures.

### Feature views

The FE node can now emit two model-facing views without adding a new graph node:

- `linear_view` — intended for linear models such as logistic regression
- `tree_view` — intended for tree models such as random forest and XGBoost

If dual views are present:
- training uses `linear_view` for logistic regression and `tree_view` for RF/XGB
- evaluation and XAI also use the correct view per model
- inference uses the selected model's full feature frame for row lookup and SHAP

If dual views are absent, the pipeline falls back to the legacy single shared feature frame.

When both views exist, the default active `train_frame` / `test_frame` / `full_feature_frame` / `feature_columns` aliases point to `tree_view`. Model-aware paths should use the view accessors instead of assuming those aliases are neutral.

## Training

Three-step process:
1. **Baseline fit**: All models trained with defaults to get initial metrics
2. **LLM-reasoned grids**: `reason-hyperparameter-grid` skill takes dataset characteristics + baseline metrics, returns range-based search spaces per model
3. **Tuning**: Optuna Bayesian optimization (TPESampler, macro_f1). The numeric tuning controls are now env-backed via `config.py` (`OPTUNA_TRIALS`, `TUNING_SUBSAMPLE_MAX`, `TUNING_CV_FOLDS`, `RF_TUNING_ESTIMATORS`, `XGB_TUNING_ESTIMATORS`, `XGB_TUNING_EARLY_STOPPING_ROUNDS`, `XGB_FINAL_EARLY_STOPPING_ESTIMATORS`, `XGB_FINAL_EARLY_STOPPING_ROUNDS`, `TUNING_MAX_DEPTH_CAP`). The fold builder is policy-aware:
   - `iid_stratified` → stratified row-level folds
   - `grouped_entity` → grouped folds / grouped early-stopping holdout
   - `temporal` → time-ordered folds / last-window early-stopping holdout
   Deterministic grid sanitization runs before Optuna search: tree-model `n_estimators` is stripped from the LLM grid, `max_depth` is capped, reversed numeric ranges are swapped into valid order, and bounded fractions such as `subsample` / `colsample_bytree` are clipped to valid ranges. XGBoost uses a two-step early-stopping pattern: find best tree count on an inner validation split, then retrain on the full training set with that fixed count.

Three candidate models:
- **Logistic Regression**: Wrapped in `sklearn.Pipeline` with `StandardScaler`. Uses `class_weight='balanced'`. Tuning limited to `model__C` (lbfgs solver only supports l2).
- **Random Forest**: 300 estimators, `class_weight='balanced'`, `n_jobs=-1`. Tuned on n_estimators, max_depth, min_samples_split.
- **XGBoost**: `multi:softprob` objective. Receives `sample_weight` via `compute_sample_weight('balanced')`. Early stopping determines n_estimators. Tuned on max_depth, learning_rate, subsample, colsample_bytree.

Learning curves are extracted from the early-stopped inner-fit model and attached to the final retrained XGBoost estimator.

### Validation policy

`dataset-policy-spec` now carries a separate `validation_policy` for tuning / early stopping. This is distinct from the final train/test split.

- `iid_stratified` — default for independent classification rows
- `grouped_entity` — use when repeated rows belong to the same entity and inner validation must keep entities disjoint
- `temporal` — use when rows have meaningful time order and validation should occur on later observations

The reasoning model chooses the policy; Python code executes the split mechanics deterministically.

### Tuning artifacts

`train-models` now persists tuning artifacts under `lab/logs/tuning/<run_id>/`:

- `grids.json` — LLM-proposed grid payload used for tuning
- `trial_history.json` — per-model Optuna trial history
- `summary.json` — validation policy, baseline metrics, and best tuned results

The pipeline state exposes `tuning_artifact_path` so these artifacts can be traced back to the exact run alongside other run-scoped evidence.

Selection: deterministic in Python. `choose_best_model()` selects the winner by `max(macro_f1, weighted_f1)` on held-out evaluation metrics. The `reason-model-selection` skill is now justification-only: it can explain the metric winner and add hypothesis validation context, but it cannot override `selected_model_name`.

## Hypothesis chain (4-layer)

Every analytical node produces bold three-tier hypotheses: tested predictions (closed-loop), supported conjectures (partially testable), and exploratory leads (open threads).

1. **EDA hypotheses** (`generate-eda-hypotheses` node): LLM interprets model-eligible EDA statistics → model selection prediction, class struggle prediction, feature behavior predictions, while treating raw blocked MI leaders as audit-only leakage alerts
2. **Training diagnostics** (`training-diagnostics` node): LLM interprets training results → per-class struggle level, capacity analysis, confidence analysis, validates EDA hypotheses, generates new hypotheses
3. **Global XAI**:
   - `global-xai` node: programmatic evidence — SHAP beeswarm/dependence, grouped PFI, conditional PDP/ALE
   - `interpret-global-xai` node: LLM interpretation of cross-method consensus, feature-effect shapes, and global three-tier hypotheses
4. **Local XAI**:
   - `local-xai` node: programmatic casebook — representative + borderline + worst misclassification per class, SHAP waterfall per case
   - `interpret-local-xai` node: LLM interpretation of per-class stories, confusion patterns, and decision-boundary hypotheses
   - `interpret-local-xai` receives `global_xai_interpretation` plus a thin global ranking reference so local reasoning can cite global findings without re-deriving them

The analysis bundle packages all 4 layers into a persisted JSON artifact with stable schema. The LLM-authored semantic outputs pass through verbatim into `analysis_bundle_summary`; only raw numeric-heavy artifacts are kept separate so `explain-risk` receives the full meaning chain without beeswarm/PDP payload bloat.

## Explainability (XAI)

### Global methods
- **SHAP** (always): Global importance (mean |SHAP|), beeswarm data, dependence data. TreeExplainer for RF/XGBoost, LinearExplainer for LR Pipeline. Computed on 500 test samples.
- **Grouped PFI** (always): Permutation feature importance with one-hot columns grouped by shared prefix. Uses `sklearn.permutation_importance` with f1_macro scoring and position-based subsampling so view-frame indices do not need to match target indices.
- **PDP** (when feasible): Computed for the same small set of top continuous features used for ALE. Uses `sklearn.partial_dependence`; integer-valued features are cast to float before grid construction to avoid dtype/grid mismatches. Returns per-class probability curves.
- **ALE** (when feasible): Computed on that same top-feature set as a complementary view rather than a mutually-exclusive alternative. Custom implementation: quantile binning → finite differences → accumulate + centre. Returns per-class curves.

### Local methods (casebook)
- **Classification strategy**: Per class — representative (most confident correct), worst misclassification (most confident wrong), borderline (least confident correct). Up to 9 cases total.
- **Per-case SHAP**: Top-10 features by |SHAP value| for each selected case.

### Analysis bundle
- Persisted JSON artifact with stable schema: `eda_hypotheses`, `training_diagnostics`, `global_xai_interpretation`, `local_xai_interpretation`, `local_casebook`, `feature_engineering_hypothesis`, `selection_justification`
- Saved to `lab/logs/` using the stage `run_id` when available, so the stage log and analysis bundle share one stable artifact identity
- `analysis_bundle_summary` is now the full semantic bundle passed to explain-risk
- Raw numeric artifacts (`shap_importance`, `pfi`, `pdp`, `ale`, dependence feature names) are saved separately in `numeric_artifacts` on disk and kept out of the LLM payload

### Run artifacts and Developer Trace
- `run_stage.py` now emits three provenance-linked artifacts per run when applicable:
  - `stage_<stage>_<run_id>.log` — human-readable raw log
  - `analysis_bundle_<run_id>.json` — semantic analysis bundle
  - `trace_events_<run_id>.jsonl` — structured run lifecycle + node update events
- Code generation now has a separate stable artifact tree:
  - `lab/logs/codegen/<run_id>/preprocessing/<attempt_label>/...`
  - `lab/logs/codegen/<run_id>/feature_engineering/<attempt_label>/...`
  These folders persist the generated code, redacted prompt payload, model response, metadata, and downstream execution / validation / audit reports for each attempt.
- `active_run.json` stores `log_path`, `bundle_path`, and `trace_path` for the currently running job. `run_stage.py` remains the sole writer of this file; `app.py` is read-only.
- Saved cache provenance now preserves `cache_log_path`, `cache_bundle_path`, and `cache_trace_path`, so the Gradio app can bind a cached model back to the exact run that produced it.
- LangGraph state now surfaces `preprocessing_codegen_snapshot_path` and `feature_engineering_codegen_snapshot_path` as top-level `_path` fields so the existing trace artifact pipeline can expose those snapshot roots automatically.
- The Gradio Developer Trace tab prefers structured trace artifacts first:
  - live run: `active_run.trace_path`
  - cached run: `state.cache_trace_path`
  - raw stage log fallback only when the trace artifact is unavailable
- Structured trace JSONL now includes both node-complete events and run-level lifecycle events such as `run_start`, `run_complete`, `run_failed`, and `cache_saved`.

## Confidence diagnostics

`training_diagnostics.confidence_analysis` is now a hybrid object:

- `summary` — LLM-written narrative interpretation of confidence patterns
- `by_model` — programmatic machine-readable statistics used by inference-time caution logic:
  - `correct_mean_confidence`, `wrong_mean_confidence`
  - `correct_std_confidence`, `wrong_std_confidence`
  - `per_class_correct_mean_confidence`, `per_class_wrong_mean_confidence`

This keeps the interpretive value of the LLM output while preventing downstream inference-time caution logic from depending on free-form prose.

## Class label encoding

`class_names` are sorted alphabetically to ensure consistent `label_to_id` mapping that aligns with sklearn's `.classes_` attribute.

## LLM integration

All LLM calls go through `llm.py:call_json_response`, which strips markdown fences and retries up to 3 times on `json.JSONDecodeError`. System prompts are loaded at runtime from `skills/*.md` files. Model: o4-mini for reasoning/analytical nodes, gpt-4o for codegen, gpt-4o-mini for simple text generation.
