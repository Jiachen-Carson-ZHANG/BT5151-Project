# Hyperparameter Tuning: LLM-Reasoned Search Grid

**Date:** 2026-04-09
**Status:** SUPERSEDED by `bayesian-optimization-over-random-search.md` — RandomizedSearchCV replaced with Optuna in Run 009.

## Current state (as of Run 007)

All 3 models use hardcoded hyperparameters:
- LR: max_iter=1000 (just "enough to converge")
- RF: n_estimators=300 (arbitrary)
- XGBoost: n_estimators=300, max_depth=6, learning_rate=0.1 (defaults)

No tuning is performed. This likely leaves significant performance on the table.

## Should the LLM reason the search grid?

**Yes, partially.** The LLM is good at:
- Reasoning which hyperparameters matter given dataset characteristics (100k rows, 65 features, 3-class imbalance)
- Suggesting sensible ranges (e.g., "with 100k rows, max_depth 6-12 is reasonable; below 6 underfits, above 12 risks overfitting")
- Knowing which parameters interact (e.g., learning_rate and n_estimators are inversely related)

The LLM should NOT:
- Run the actual search (that's GridSearchCV/Optuna's job)
- Pick final values without cross-validation evidence

## Proposed approach: LLM-reasoned grid + deterministic search

1. **New skill: `reason-hyperparameter-grid.md`** — LLM receives dataset profile, model type, class distribution, feature count. Returns a JSON search grid per model.
2. **Template code runs `RandomizedSearchCV`** — with the LLM-suggested grid, 5-fold stratified CV, scoring=macro_f1.
3. **Results logged** — best params + CV scores stored in state for the report.

### Why RandomizedSearchCV over GridSearchCV?

With 3 models and multiple hyperparameters, full grid search is too expensive. Randomized search with 20-50 iterations per model gives good coverage in reasonable time.

### Why not Optuna?

Optuna is better for large search spaces with Bayesian optimization. For our case (3 models, 3-5 params each, 20-50 iterations), RandomizedSearchCV is simpler, stays within sklearn, and is easier to explain in the report.

## Example LLM-reasoned grid

For XGBoost on our dataset (100k rows, 65 features, 3-class):
```json
{
  "n_estimators": [100, 300, 500],
  "max_depth": [4, 6, 8, 10],
  "learning_rate": [0.01, 0.05, 0.1],
  "subsample": [0.7, 0.8, 1.0],
  "colsample_bytree": [0.7, 0.8, 1.0]
}
```

## Implementation considerations

- Tuning is expensive (~5-10 min per model with CV). Could make it optional via a flag.
- Tuning should happen inside `train-models` node, not as a separate node.
- The rubric says "AI must NOT be used to select or configure models" — the LLM reasoning the grid is borderline. The safe interpretation: the LLM suggests ranges, the search selects values. Document this transparently.
