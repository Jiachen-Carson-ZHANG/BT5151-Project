# Bayesian Optimization (Optuna) over Random Search

**Date:** 2026-04-10

## The question

Run 008 used RandomizedSearchCV (15 iterations per model). The earlier analysis (`hyperparameter-tuning-approach.md`) argued *against* Optuna: "for our case (3 models, 3-5 params each, 20-50 iterations), RandomizedSearchCV is simpler, stays within sklearn, and is easier to explain in the report."

After seeing Run 008 results (RF tuning took 111 min, XGB tuning actually *hurt* test score), we reconsidered.

## Why the original argument was wrong

1. **Random search doesn't learn.** Trial 15 is as blind as trial 1. It can waste iterations on clearly bad regions (e.g., max_depth=None which caused the 111 min RF problem).
2. **Bayesian search converges faster.** Optuna's TPE sampler builds a probabilistic model of which parameter regions yield good scores. After 5-6 trials it focuses on promising regions.
3. **The complexity argument was overstated.** Optuna's API is as simple as sklearn's: define an objective function, call `study.optimize(n_trials=15)`. The `_suggest_params` helper is ~20 lines.
4. **Early stopping integration.** RandomizedSearchCV doesn't naturally support XGBoost early stopping (n_estimators as a dynamic ceiling). Optuna's trial-based approach makes this trivial — set n_estimators=1000 in the objective, let early stopping find the right count.

## What changed

- `train.py`: Replaced `RandomizedSearchCV` with `optuna.create_study(direction="maximize", sampler=TPESampler)`
- Grid format: Changed from list-based (`[100, 300, 500]`) to range-based (`{"type": "int", "low": 100, "high": 500, "step": 100}`) — Optuna works with continuous/integer ranges, not discrete lists
- XGBoost `n_estimators` removed from grid entirely — early stopping determines optimal rounds

## Expected benefits

- **Faster convergence**: 15 Bayesian trials should find better params than 15 random trials
- **No degenerate configs**: Optuna won't repeatedly sample max_depth=None or extreme values
- **Early stopping for XGB**: Prevents overfitting (Run 008 had 13.7pp CV-test gap)
- **Learning curves**: XGBoost's `evals_result()` reveals train/val loss per round

## Trade-offs

- Adds `optuna` dependency (already widely used, well-maintained)
- Optuna's TPE is sequential by default (`n_jobs=1`) — can't parallelize trials as easily as random search. But for our 15-trial budget this doesn't matter.
- Results are slightly less reproducible than grid/random search due to the adaptive sampler (mitigated by seeded TPESampler)

## Report narrative

The original analysis worried about "easier to explain in the report." Optuna is actually *more* explainable: the trial history shows how the sampler explored the space, and the best_trial shows exactly which parameters won. The report can include the Optuna trial history as evidence of systematic hyperparameter search.
