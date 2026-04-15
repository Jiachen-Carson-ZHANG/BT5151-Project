# When is preprocessing model-specific?

**Date:** 2026-04-09

## The three-layer spectrum

| Layer | Model-specific? | Examples |
|-------|----------------|---------|
| **Data cleaning** | No — always shared | NaN imputation, garbage removal, type coercion, outlier clipping |
| **Feature engineering** | Mostly shared, some model-specific | Shared: ratios, interactions, log transforms. Model-specific: polynomial features help LR but are noise for trees; target encoding helps GBM but can leak for LR |
| **Feature scaling/encoding** | Yes | LR needs scaling, trees don't. One-hot works for all but high cardinality hurts LR more than trees. Ordinal encoding: trees can split on it meaningfully, LR treats it as linear which may be wrong |

## Why shared preprocessing is right for this project

1. The rubric requires 2+ models compared fairly — model-specific preprocessing makes comparison harder to justify
2. The scaling difference is already handled inside the sklearn Pipeline wrapper (StandardScaler only applies to LR)
3. Standard ML workflow is: clean → engineer features → train all candidates on same features → compare → select

## When you'd go model-specific (production)

In production (e.g., at a bank), you'd have a shared feature store for common features, then each model pipeline adds model-specific transforms. This makes sense when:
- You're optimizing for maximum performance on one deployed model
- You have different teams owning different models
- Feature computation is expensive and you want to skip unnecessary transforms

## Implication for our pipeline

Our architecture (shared feature frame → all models train on same data) is correct for the academic/comparison use case. The only model-specific transform is StandardScaler, which lives in the training template, not preprocessing.
