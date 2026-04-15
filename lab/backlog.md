# Experiment Backlog

Ideas discussed but not yet tested. Prioritized by expected impact.

## High priority

### Temperature tuning for audit calls
- **Context**: Quality reviewer uses temperature=1.0 (OpenAI default). Stochastic variation between calls causes different issues found each round.
- **Hypothesis**: temperature=0 for audit-preprocessing calls would reduce round-to-round drift and improve convergence.
- **What to measure**: Number of repair rounds, consistency of issues found across identical runs.
- **Discussed**: 2026-04-09

### Reasoning model for remaining analytical nodes
- **Context**: o4-mini validated for column-transform-spec (see `lab/analysis/reasoning-model-for-analytical-nodes.md`). Remaining analytical nodes (FE, model selection, explain-risk) still on gpt-4o.
- **Hypothesis**: o4-mini will produce more insightful FE hypotheses and better model selection justification, same as it did for column-transform-spec.
- **What to measure**: FE hypothesis accuracy vs SHAP, model selection justification quality, explain-risk hypothesis notes quality.
- **Trade-off**: o4-mini is 3× slower and 3× more output tokens per call. Worth it only for analytical reasoning, not codegen.
- **Discussed**: 2026-04-10
- **Prerequisite**: Full pipeline run with o4-mini on column-transform-spec to confirm downstream improvements.

## Medium priority

### Core principles section in remaining skill prompts
- **Context**: Principles-based prompt validated for column-transform-spec with o4-mini. Other skill prompts (FE, repair) still use step-by-step instruction style.
- **Hypothesis**: Same principles-based approach will work for FE and repair prompts when paired with reasoning model.
- **Decision**: Apply when enabling o4-mini for those nodes — prompt style should match model type.
- **Discussed**: 2026-04-09, validated 2026-04-10 for column-transform-spec

### Feature sample as percentage instead of fixed 5 rows
- **Context**: Audit reviewer sees only 5 sample rows from feature_frame. For 100k rows, this is 0.005% — may miss patterns.
- **Counter-argument**: The reviewer already gets full per-column stats (min, max, mean, nunique). The sample is for spot-checking encoding patterns, not statistical significance.
- **Decision**: Deferred — stats are more important than sample size for the reviewer's job.
- **Discussed**: 2026-04-09

### AUC-ROC and additional evaluation metrics
- **Context**: Currently compute accuracy, macro_f1, weighted_f1, per-class P/R/F1, confusion matrix. Missing AUC-ROC (one-vs-rest) which is standard for imbalanced classification.
- **Hypothesis**: AUC-ROC provides a threshold-independent view of model discrimination. Would strengthen the evaluation section of the report.
- **What to measure**: AUC-ROC per class and macro-average.
- **Discussed**: 2026-04-09

## Low priority / Exploratory

### Deterministic pre-filter before LLM audit
- **Context**: Could move some checks (NaN detection, cardinality explosion, constant features) to hardcoded Python and only use LLM for judgment calls (distribution sanity, encoding quality).
- **Trade-off**: Reduces LLM scope (faster, cheaper, deterministic) but less flexible.
- **Discussed**: 2026-04-09

### Self-consistency voting for audit
- **Context**: Run audit 3x at temperature=1.0, only keep issues found in 2+ runs.
- **Trade-off**: 3x cost per audit round. Only justified if single-pass audit has high false positive rate.
- **Discussed**: 2026-04-09

### Feature engineering quality review (LLM-driven)
- **Context**: FE validation is structural only (row counts, no NaN/inf, column alignment). No LLM review of whether the engineered features make sense.
- **Question**: Should feature engineering have an LLM quality review like preprocessing? Risk: never-ending review cycles.
- **Discussed**: 2026-04-09

---

## Completed (moved from backlog)

| Item | Completed in | Notes |
|------|-------------|-------|
| Feature scaling in training pipeline | Run 007 | LR Pipeline with StandardScaler |
| Add XGBoost as third candidate model | Run 007 | XGBoost with multi:softprob |
| Class weight balancing | Run 008 | XGBoost sample_weight via compute_sample_weight |
| Improve quality reviewer scope | Run 007 | Enriched column profiles with top-10 values, garbage detection |
| SHAP for XAI | Run 008 | TreeExplainer for RF/XGB, LinearExplainer for LR |
| Hyperparameter tuning | Run 008 → 009 | RandomizedSearchCV in 008, upgraded to Optuna in 009 |
| Confusion matrix + per-class logging | Run 007 | Added to compute_multiclass_metrics |
| Reasoning model for codegen | Run 009 | .env.example per-node config |
| Reasoning model A/B (column-transform-spec) | Run 009 | o4-mini validated — 3× richer reasoning, stable bounds/encoding |
| EDA before specs | Run 009 | Programmatic EDA node with MI, ANOVA, correlations |
| Reasoning chain (hypothesis validation) | Run 009 | column-transform-spec → FE → model selection → explain-risk |
| Optuna Bayesian optimization | Run 009 | Replaced RandomizedSearchCV with TPESampler |
| XGBoost early stopping + learning curves | Run 009 | n_estimators=1000, early_stopping_rounds=50 |
