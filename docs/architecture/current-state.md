# Architecture: Current State

Last updated: 2026-04-15

## Pipeline

LangGraph `StateGraph` with Pydantic `BaseModel` state (`CreditRiskState`).

27-node pipeline with EDA, hypothesis generation, two conditional repair loops (preprocessing and feature engineering), training diagnostics, split XAI interpretation (global + local casebook), analysis bundle packaging, and a reasoning chain from EDA through to final explanation:

```
dataset-policy-spec
  → exploratory-data-analysis (programmatic — correlations, class separability, skewness, missing patterns, cardinality, discriminative features)
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
  → validate-feature-engineering (structural checks)
  →   [pass] train-models
  →   [fail] repair-feature-engineering-code → (back to inspect)
  → evaluate-models
  → training-diagnostics (LLM — per-class struggle, capacity analysis, confidence, hypothesis validation)
  → select-model
  → global-xai (programmatic — SHAP beeswarm/dependence, grouped PFI, conditional PDP/ALE)
  → local-xai (programmatic — casebook: representative + borderline + worst misclassification per class)
  → interpret-global-xai (LLM — cross-method consensus, feature-effect shapes, global hypotheses)
  → interpret-local-xai (LLM — per-class stories, failure patterns, boundary analysis; receives global XAI interpretation as context)
  → package-analysis-bundle (serialize to disk, pass full semantic bundle through to explain-risk)
  → run-inference
  → explain-risk (LLM — grounded in analysis bundle summary)
  → recommend-action
```

## Key modules

| Module | Responsibility |
|--------|---------------|
| `state.py` | Pydantic state schema for the full pipeline (~60 fields) |
| `config.py` | Constants: target column, group column, seed, test size, default model |
| `profile.py` | Builds dataset profile (row count, target distribution, missing counts) |
| `eda.py` | Programmatic EDA: correlations, ANOVA, mutual information, skewness, missing patterns, cardinality |
| `hypotheses.py` | LLM-driven analytical interpretation: EDA hypotheses, training diagnostics, global XAI interpretation, local XAI interpretation |
| `preprocess.py` | LLM-driven codegen, static inspection, isolated execution, artifact validation (structural + semantic role contract), LLM quality review, repair with escalation |
| `feature_engineering.py` | LLM-driven feature engineering codegen, subprocess execution, single/dual-view validation, repair |
| `train.py` | Builds candidate models, Optuna Bayesian tuning, policy-aware validation splits, XGBoost early stopping, learning curves |
| `evaluate.py` | Multiclass metrics, hardcoded + LLM-driven model selection with hypothesis validation |
| `xai.py` | Global SHAP, grouped PFI, PDP, ALE, per-case SHAP, casebook strategy |
| `llm.py` | OpenAI API wrapper with JSON retry and markdown fence stripping |
| `business.py` | LLM-driven risk explanation (grounded in analysis bundle) and action recommendation |
| `skill_prompts.py` | Loads skill markdown files as runtime system prompts |
| `graph.py` | Node functions, routing logic, graph construction |

## Preprocessing codegen loop

The LLM generates Python code that runs in a subprocess. The contract:

1. **Generate**: LLM produces `{"code": "...", "entrypoint": "run_preprocessing"}`
2. **Inspect**: Static AST analysis blocks forbidden imports/calls and `inplace=True`
3. **Execute**: Code runs in isolated subprocess with 180s timeout; must write 5 artifacts
4. **Validate (structural + semantic role)**: Checks artifact existence, target exclusion, feature frame non-empty, group overlap zero. Then runs the deterministic **semantic role validator** — checks each column's post-encoding output against the invariant implied by its declared `semantic_role` in the transform spec. Structured findings `{column, declared_role, violation, observed, expected, likely_cause}` are emitted as `role_violations` in the validation report. Any violation fails `passed`.
5. **Review (LLM quality)**: Two-mode review — comprehensive first pass, focused follow-up on subsequent rounds. Checks distribution sanity, encoding quality (garbage categories, delimiter artifacts, one-sided clipping), spec compliance, completeness
6. **Repair**: On failure at inspect, validate, or review, LLM receives error context and produces patched code; max 5 total attempts. The graph only accepts residual **minor** quality issues after repeated repair attempts — critical/major audit issues keep the repair loop active.
7. **Capability-ceiling escalation**: If the same `(column, violation)` pair appears in consecutive attempts, the repair call is routed to an escalated caller (`repair-preprocessing-code-escalated`) which resolves to a stronger model via `OPENAI_MODEL_REPAIR_PREPROCESSING_CODE_ESCALATED` env var. The escalation is logged as a `CAPABILITY CEILING` warning. The trigger resets after each repair so it re-arms only if the same violation persists.

## Semantic role contract

Every column in `column_transform_spec.transforms` declares:

- `semantic_role` — one of 12 enumerated roles: `identifier`, `group_identifier`, `target`, `numeric_continuous`, `numeric_count`, `ordered_categorical`, `unordered_categorical`, `binary_flag`, `multi_value_set`, `temporal_feature`, `free_text`, `leakage_risk_feature`.
- `representation_intent` — optional encoding choice when the role admits multiple valid encodings (e.g. `binary_membership` vs `count_membership` for `multi_value_set`; `one_hot` vs `target_encoded` for `unordered_categorical`).

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

## Feature engineering codegen loop

LLM generates Python code for feature transforms, running in a subprocess. Max 3 attempts (1 generate + 2 repairs). The contract:

1. **Generate**: LLM produces `{"code": "...", "entrypoint": "engineer_features"}` with heuristic rules (drop constants, handle correlations, log-transform skewed, domain interactions)
2. **Inspect**: Reuses same AST analysis as preprocessing (forbidden imports, `inplace=True` ban)
3. **Execute**: Code runs in subprocess; takes `(train_df, test_df, workspace_path)`, writes either:
   - legacy single-view artifacts: `engineered_train.csv`, `engineered_test.csv`, `feature_engineering_report.json`
   - or dual-view artifacts: `engineered_train_linear.csv`, `engineered_test_linear.csv`, `engineered_train_tree.csv`, `engineered_test_tree.csv`, `feature_engineering_report.json`, `view_metadata.json`
4. **Validate**: Structural checks — row counts match, no NaNs, no infs, features non-empty, column alignment, max feature cap (≤5x input), and no remaining string/category/object columns. Boolean dummy columns are accepted as valid numeric model inputs. In dual-view mode, these checks run separately for each view declared in `view_metadata.json`.
5. **Repair**: On failure, LLM receives error context and produces patched code

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
3. **Tuning**: Optuna Bayesian optimization (TPESampler, 10 trials, training-only validation folds, macro_f1). The fold builder is policy-aware:
   - `iid_stratified` → stratified row-level folds
   - `grouped_entity` → grouped folds / grouped early-stopping holdout
   - `temporal` → time-ordered folds / last-window early-stopping holdout
   XGBoost uses a two-step early-stopping pattern: find best tree count on an inner validation split, then retrain on the full training set with that fixed count.

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

Selection: LLM-driven (`reason-model-selection` skill) with hypothesis validation. Receives evaluation metrics, tuning results, SHAP importance, EDA top features, and FE hypothesis. Falls back to metric-based max(macro_f1) on failure.

## Hypothesis chain (4-layer)

Every analytical node produces bold three-tier hypotheses: tested predictions (closed-loop), supported conjectures (partially testable), and exploratory leads (open threads).

1. **EDA hypotheses** (`generate-eda-hypotheses` node): LLM interprets EDA statistics → model selection prediction, class struggle prediction, feature behavior predictions
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
- **ALE** (conditional): Computed when EDA shows |r| > 0.5 correlation among top features. Custom implementation: quantile binning → finite differences → accumulate + centre. Returns per-class curves.
- **PDP** (conditional): Computed for top continuous features without high correlation. Uses `sklearn.partial_dependence`; integer-valued features are cast to float before grid construction to avoid dtype/grid mismatches. Returns per-class probability curves.

### Local methods (casebook)
- **Classification strategy**: Per class — representative (most confident correct), worst misclassification (most confident wrong), borderline (least confident correct). Up to 9 cases total.
- **Per-case SHAP**: Top-10 features by |SHAP value| for each selected case.

### Analysis bundle
- Persisted JSON artifact with stable schema: `eda_hypotheses`, `training_diagnostics`, `global_xai_interpretation`, `local_xai_interpretation`, `local_casebook`, `feature_engineering_hypothesis`, `selection_justification`
- Saved to disk alongside experiment record
- `analysis_bundle_summary` is now the full semantic bundle passed to explain-risk
- Raw numeric artifacts (`shap_importance`, `pfi`, `pdp`, `ale`, dependence feature names) are saved separately in `numeric_artifacts` on disk and kept out of the LLM payload

## Class label encoding

`class_names` are sorted alphabetically to ensure consistent `label_to_id` mapping that aligns with sklearn's `.classes_` attribute.

## LLM integration

All LLM calls go through `llm.py:call_json_response`, which strips markdown fences and retries up to 3 times on `json.JSONDecodeError`. System prompts are loaded at runtime from `skills/*.md` files. Model: o4-mini for reasoning/analytical nodes, gpt-4o for codegen, gpt-4o-mini for simple text generation.
