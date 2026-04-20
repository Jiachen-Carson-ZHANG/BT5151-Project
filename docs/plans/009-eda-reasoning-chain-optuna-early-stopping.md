# Plan: EDA + Reasoning Chain + Optuna + Early Stopping

## Context

Run 008 passed end-to-end but exposed several weaknesses:
- **Metrics are mediocre** (RF macro_f1=0.68): No EDA means the pipeline is blind to correlation structure, class separability, and feature quality before making decisions.
- **No reasoning trail**: Column-transform-spec, FE, and model selection make critical decisions silently. Only the final explain/recommend nodes produce reasoning. There's no hypothesis chain from data understanding → feature decisions → model selection → SHAP validation.
- **Tuning is inefficient**: RandomizedSearchCV with `max_depth=None` took 111 min for RF. Bayesian optimization (Optuna) converges faster and prevents degenerate configs.
- **XGBoost wastes trees**: No early stopping means all 300 trees are built even when the model overfit 100 rounds ago. CV-test gap was 13.7pp.
- **No learning visibility**: No loss curves to diagnose underfitting vs overfitting.

## Architecture change

```
Current:
  dataset-policy-spec → column-transform-spec → preprocessing → FE → train → evaluate → select → inference → explain → recommend

Proposed:
  dataset-policy-spec → EDA (new) → column-transform-spec (+ reasoning) → preprocessing (+ enhanced logging)
  → FE (+ hypothesis) → train (Optuna + early stopping + learning curves) → evaluate
  → select-model (+ LLM reasoning) → inference → explain (+ hypothesis validation) → recommend
```

New node: `exploratory-data-analysis` (programmatic, no LLM).
Changed nodes: column-transform-spec, generate-feature-engineering-code, train-models, select-model, explain-risk.

## Workstreams

### A. EDA Node

**Why**: The pipeline currently decides column transforms and feature engineering from 10 sample rows and column profiles. It has no visibility into correlation structure, class-conditional distributions, or feature importance signals. This leads to blind decisions — e.g., not dropping correlated one-hot columns, not knowing which features separate classes.

**New file**: `src/bt5151_credit_risk/eda.py`

Programmatic computation (no LLM), producing a structured report:

```python
def build_eda_report(df: pd.DataFrame, target_column: str) -> dict:
```

Computes:
1. **Correlation matrix** (numeric cols): full pairwise Pearson. Extract pairs with |r| > 0.8.
2. **Class-conditional means**: For each numeric feature, mean per target class. Highlights features where class means differ significantly (ANOVA F-stat or simple ratio of between-class to within-class variance).
3. **Skewness per numeric column**: |skew| values to inform log-transform decisions.
4. **Missing value patterns**: % missing per column, and whether missingness correlates with target (missing-not-at-random detection).
5. **Cardinality per categorical**: nunique. Flags high-cardinality columns (>20 unique) that will explode with one-hot.
6. **Top discriminative features**: Mutual information or F-classif scores (from `sklearn.feature_selection`) — ranks features by predictive power before any modeling.

Output: `{"correlations": {...}, "class_separability": {...}, "skewness": {...}, "missing_patterns": {...}, "cardinality": {...}, "top_discriminative_features": [...]}` stored in state as `eda_report`.

**State change**: Add `eda_report: dict | None = None` to `CreditRiskState`.

**Graph change**: Insert `exploratory-data-analysis` node between `dataset-policy-spec` and `column-transform-spec`.

**Logging**: Log top-10 discriminative features, high-correlation pairs, high-cardinality columns.

### B. Enhanced Logging

Add logging at nodes that currently log only outcomes:

| Node | Currently logged | Add |
|------|-----------------|-----|
| column-transform-spec | `transform spec keys` | Per-column action + cleaning + encoding decisions (one line per column) |
| generate-preprocessing-code | `code length` | Nothing extra needed |
| validate-preprocessing-output | `feature_frame rows x cols` | Column stats summary: numeric cols min/max range, categorical cols cardinality |
| generate-feature-engineering-code | `code length` | Nothing extra needed |
| validate-feature-engineering | `features: X → Y cols` | Log the FE report JSON (what was added/removed/transformed) |
| train-models | `baseline metrics`, `tuning reasoning` (truncated) | Full tuning grids (not just param names), full reasoning, per-model trial history |
| select-model | `selected: model_name` | Full selection reasoning (when LLM-driven, see workstream C) |

Implementation: Changes only in `graph.py` node functions — add `logger.info()` calls reading from state or intermediate results. No new files.

### C. Reasoning Chain

Transform analytical nodes from silent decision-makers into reasoning-producing nodes. Each node's reasoning flows downstream, creating a traceable hypothesis chain.

#### C1. column-transform-spec reasoning

**Skill prompt change** (`skills/column-transform-spec.md`): Add `"reasoning"` field to output schema — a dict mapping column names to a short explanation of why that action was chosen.

```json
{
  "transforms": { ... },
  "reasoning": {
    "Type_of_Loan": "Multi-value delimited column — must split before encoding to avoid cardinality explosion",
    "Annual_Income": "Numeric with extreme outliers (max 24M vs p99 150k) — clip to [p1, p99]"
  }
}
```

**Payload enrichment**: Pass `eda_report.top_discriminative_features` and `eda_report.correlations.high_pairs` to the column-transform-spec LLM so it can make informed decisions.

**State change**: Store reasoning in `column_transform_spec` dict (already a dict, just add the key).

#### C2. FE hypothesis

**Skill prompt change** (`skills/generate-feature-engineering-code.md`): Add a required `"hypothesis"` field to the output:

```json
{
  "code": "...",
  "entrypoint": "engineer_features",
  "hypothesis": {
    "interactions_rationale": "Created Debt_to_Income because EDA shows Outstanding_Debt and Annual_Income have high discriminative power individually — their ratio should capture repayment capacity",
    "dropped_features_rationale": "Dropped 3 constant one-hot columns and 2 highly correlated pairs identified in EDA",
    "expected_impact": "Expect macro_f1 improvement from noise reduction and domain-meaningful ratios"
  }
}
```

**Payload enrichment**: Pass `eda_report` (correlation pairs, discriminative features, skewness) to the FE LLM.

**State change**: Add `feature_engineering_hypothesis: dict | None = None` to `CreditRiskState`.

#### C3. Model selection reasoning (LLM-driven)

**Current**: `choose_best_model()` in `evaluate.py` is a hardcoded `max(macro_f1)`. This produces a generic justification.

**Change**: Make model selection LLM-driven. New skill `skills/reason-model-selection.md`. LLM receives:
- Evaluation results (per-class metrics, confusion matrices)
- EDA report (class separability)
- FE hypothesis
- Tuning results (best params, CV scores)
- Global SHAP importance

Returns: `{"model_name": "...", "justification": "...", "hypothesis_validation": "..."}` — where `hypothesis_validation` explicitly compares what was expected (from FE/EDA) with what was observed (from SHAP/metrics).

**Change in evaluate.py**: Add `reason_model_selection()` function that calls the LLM. Keep `choose_best_model()` as fallback.

**Change in graph.py**: `select_model_node` calls LLM-driven selection instead of hardcoded max.

#### C4. Explain-risk grounded in full chain

**Change in business.py**: `explain_risk()` receives the full hypothesis chain:
- EDA top discriminative features
- FE hypothesis (what interactions were created and why)
- Model selection reasoning
- SHAP contributions (per-prediction)
- Global SHAP importance

The system prompt instructs the LLM to: (1) explain in business terms, (2) note which hypotheses were confirmed/refuted by SHAP, (3) flag any surprising SHAP features not anticipated by the hypothesis.

### D. Optuna + Early Stopping + Learning Curves

#### D1. Optuna replaces RandomizedSearchCV

**Add**: `optuna>=4.0.0` to `requirements.txt`.

**Change LLM grid format**: The skill prompt (`skills/reason-hyperparameter-grid.md`) currently returns list-based grids. Change to range-based specs:

```json
{
  "grids": {
    "random_forest": {
      "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
      "max_depth": {"type": "int", "low": 6, "high": 20},
      "min_samples_split": {"type": "int", "low": 2, "high": 10}
    },
    "xgboost": {
      "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 100},
      "max_depth": {"type": "int", "low": 4, "high": 12},
      "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
      "subsample": {"type": "float", "low": 0.6, "high": 1.0}
    }
  },
  "reasoning": "..."
}
```

For LR: only `model__C` with log-uniform range — Optuna's `suggest_float(log=True)` is ideal for regularization.

**Rewrite `tune_models()`** in `train.py`:

```python
def tune_models(models, grids, train_frame, train_target, sample_weights=None):
    for model_name, model in models.items():
        grid_spec = grids.get(model_name)

        def objective(trial):
            params = _suggest_params(trial, grid_spec, model_name)
            cloned = clone(model).set_params(**params)

            cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
            scores = []
            for train_idx, val_idx in cv.split(train_frame, train_target):
                X_train, X_val = train_frame.iloc[train_idx], train_frame.iloc[val_idx]
                y_train, y_val = train_target.iloc[train_idx], train_target.iloc[val_idx]

                fit_kwargs = {}
                if model_name == "xgboost" and sample_weights is not None:
                    fit_kwargs["sample_weight"] = sample_weights[train_idx]
                    # Early stopping
                    fit_kwargs["eval_set"] = [(X_val, y_val)]
                    cloned.set_params(early_stopping_rounds=50, n_estimators=1000)

                cloned.fit(X_train, y_train, **fit_kwargs)
                preds = cloned.predict(X_val)
                scores.append(f1_score(y_val, preds, average="macro", zero_division=0))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        study.optimize(objective, n_trials=15, n_jobs=1)

        # Retrain best model on full data
        best_params = study.best_trial.params
        final_model = clone(model).set_params(**best_params)
        # ... fit on full train_frame
```

Helper `_suggest_params(trial, grid_spec, model_name)` translates grid spec types to `trial.suggest_int()`, `trial.suggest_float()`, etc.

#### D2. XGBoost early stopping

Integrated into Optuna objective (see above). Key points:
- Set `n_estimators=1000` (high ceiling), let early stopping find the right number
- `early_stopping_rounds=50` — stop if no improvement in 50 rounds
- `eval_set=[(X_val, y_val)]` from the CV fold's validation split
- After finding best params, final retraining also uses early stopping with a held-out fraction

#### D3. Learning curves

After final retraining of XGBoost with best params:
- Use `eval_set` during final fit to capture per-round train/val loss
- Extract from `model.evals_result()` → dict of `{"train": {"mlogloss": [...]}, "val": {"mlogloss": [...]}}`
- Store in state as `learning_curves: dict | None`
- Log summary: "XGBoost converged at round N/1000, final val_loss=X.XXX"

For RF and LR: no per-round learning curves (not iterative learners). Could add validation curve (score vs hyperparam value) from Optuna trial history instead — simpler and still useful.

**State change**: Add `learning_curves: dict | None = None`, `tuning_trial_history: dict | None = None`.

### E. Reasoning Model Configuration

The `llm.py` already supports per-caller model overrides via env vars (`OPENAI_MODEL_{CALLER}`). No code changes needed — just configuration.

**Update `.env.example`** with recommended reasoning model assignments:

```env
# Analytical/reasoning nodes — benefit from deeper reasoning:
OPENAI_MODEL_COLUMN_TRANSFORM_SPEC=o4-mini
OPENAI_MODEL_GENERATE_FEATURE_ENGINEERING_CODE=o4-mini
OPENAI_MODEL_REASON_HYPERPARAMETER_GRID=o4-mini
OPENAI_MODEL_REASON_MODEL_SELECTION=o4-mini
OPENAI_MODEL_EXPLAIN_RISK=o4-mini

# Codegen nodes — instruction-following matters more:
OPENAI_MODEL_GENERATE_PREPROCESSING_CODE=gpt-4o
OPENAI_MODEL_REPAIR_PREPROCESSING_CODE=gpt-4o

# Business text nodes — simple generation:
OPENAI_MODEL_RECOMMEND_ACTION=gpt-4o-mini
```

Note: `o4-mini` is a reasoning model with good cost/quality balance. The user can swap to `o3-mini` or `o1` if preferred. The important thing is that reasoning nodes use a reasoning model, not gpt-4o.

**Caveat**: Reasoning models may not reliably produce raw JSON output — they tend to include thinking traces. May need to adjust `llm.py` to handle reasoning model response format (strip `<thinking>` tags or use structured output mode).

## Files to create

| File | Purpose |
|------|---------|
| `src/bt5151_credit_risk/eda.py` | Programmatic EDA: correlations, class separability, skewness, missing patterns, cardinality |
| `skills/reason-model-selection.md` | Skill prompt for LLM-driven model selection with hypothesis validation |

## Files to modify

| File | Change |
|------|--------|
| `src/bt5151_credit_risk/state.py` | Add `eda_report`, `feature_engineering_hypothesis`, `learning_curves`, `tuning_trial_history` |
| `src/bt5151_credit_risk/graph.py` | Add EDA node, enhanced logging at all nodes, pass EDA to downstream LLMs, LLM-driven model selection, pass hypothesis chain to explain-risk |
| `src/bt5151_credit_risk/train.py` | Rewrite `tune_models()` with Optuna, add early stopping for XGB, save learning curves + trial history |
| `src/bt5151_credit_risk/evaluate.py` | Add `reason_model_selection()` LLM-driven function alongside existing `choose_best_model()` |
| `src/bt5151_credit_risk/business.py` | `explain_risk()` receives full hypothesis chain (EDA, FE hypothesis, model selection reasoning) |
| `src/bt5151_credit_risk/preprocess.py` | Pass `eda_report` to `generate_column_transform_spec()` payload |
| `src/bt5151_credit_risk/feature_engineering.py` | `generate_feature_engineering_code()` receives `eda_report`, returns hypothesis |
| `skills/column-transform-spec.md` | Add `reasoning` field to output, add EDA inputs |
| `skills/generate-feature-engineering-code.md` | Add `hypothesis` field to output, add EDA inputs |
| `skills/reason-hyperparameter-grid.md` | Change grid format from list-based to range-based for Optuna |
| `skills/repair-feature-engineering-code.md` | Add `hypothesis` to expected output |
| `requirements.txt` | Add `optuna>=4.0.0` |
| `.env.example` | Add reasoning model recommendations per node |
| `tests/test_graph.py` | Add EDA node to expected set, monkeypatch new functions, update e2e test |
| `tests/test_state.py` | Add new fields |
| `tests/test_train.py` | Update for Optuna-based tune_models |
| `docs/architecture/current-state.md` | Update pipeline diagram, add EDA and reasoning chain sections |
| `docs/changes/implementation-log.md` | Log all changes |

## Implementation order

1. **EDA node** — `eda.py`, state field, graph node + edge, logging
2. **Enhanced logging** — add logging throughout graph.py node functions
3. **Optuna + early stopping + learning curves** — `train.py` rewrite, skill prompt, requirements.txt
4. **Reasoning chain: column-transform-spec** — skill prompt + payload enrichment with EDA
5. **Reasoning chain: FE hypothesis** — skill prompt + payload + state field
6. **Reasoning chain: model selection** — new skill + evaluate.py + graph.py
7. **Reasoning chain: explain-risk** — business.py receives full chain
8. **Reasoning model config** — .env.example update, test with reasoning model
9. **Tests** — update all test files
10. **Docs** — architecture, implementation log

## Verification

1. `pytest tests/` — all tests pass
2. Full pipeline run (user runs): `PYTHONPATH=src .venv/bin/python3 run_stage.py full 42`
3. Check log for:
   - EDA node runs and logs top discriminative features + high correlations
   - Column-transform-spec logs per-column decisions with reasoning
   - Preprocessing passes with enhanced column stats logging
   - FE logs hypothesis and report (what was added/removed)
   - Optuna logs trial history per model, early stopping rounds for XGB
   - Learning curves logged for XGB
   - Model selection logs LLM reasoning with hypothesis validation
   - Explain-risk references hypothesis chain
4. Experiment record in `lab/experiments/009-*.md`
5. Compare metrics with Run 008 baseline (RF macro_f1=0.6814)
