# Model combination choice

**Date:** 2026-04-09

## Rubric constraint

"2+ models compared fairly" — ML Development component (30pts, highest weight).

## Candidate combinations

| Combination | Families | Narrative strength | Risk |
|-------------|----------|-------------------|------|
| LR + RF (current) | Linear + bagging | Strong: explains why nonlinearity matters | No boosting baseline |
| LR + XGBoost | Linear + boosting | Strong: linear vs best-in-class tabular | Large gap may make comparison less interesting |
| RF + XGBoost | Bagging + boosting | Subtler: similar performance, nuanced comparison | Both nonlinear, weaker contrast |
| LR + RF + XGBoost | Linear + bagging + boosting | Richest: three distinct families, full spectrum | More work, more to explain |
| LR + RF + XGBoost + CatBoost | Linear + bagging + 2x boosting | Diminishing returns: XGBoost ≈ CatBoost on this data | Two models from same family adds noise, not insight |

## Why not XGBoost + CatBoost together?

Both are gradient boosting. On pre-encoded tabular data (which we have — everything is already one-hot or numeric), they'll produce very similar metrics. CatBoost's advantage is native categorical handling, but we've already encoded categoricals. The comparison would be "two similar models produce similar results" — not instructive.

## Recommendation

**LR + RF + XGBoost** (3 models, 3 distinct families):
- LR = linear baseline → shows what's achievable without nonlinearity
- RF = bagging ensemble → robust to noise, scale-invariant, interpretable via feature importance
- XGBoost = boosting → usually best on tabular data, shows marginal gain over RF

This gives the strongest report narrative: linear → ensemble → boosting, with clear reasons why each step improves.

## Implementation note

XGBoost is an optional addition from the backlog. It goes in `train.py:build_candidate_models()` as a third entry. No architectural change needed — the training and evaluation loops already iterate over all models in the dict.
