# XAI: Current Gap and SHAP Proposal

**Date:** 2026-04-09
**Status:** IMPLEMENTED in Run 008. Extended with hypothesis chain in Run 009.

## Current explain-risk approach (before SHAP)

```
model predicts → LLM gets prediction + raw record → LLM writes text explanation
```

The LLM has **no idea which features the model actually used**. It could say "low debt levels indicate good credit" when the model actually relied on payment history. This is a hallucinated explanation — it sounds plausible but isn't grounded in model reasoning.

## Why this is problematic

- The explanation is not faithful to the model's decision process
- It could mislead business users into wrong conclusions about risk drivers
- It cannot be audited or verified — there's no trace from model weights to explanation
- For academic work, this would not pass scrutiny as "explainable AI"

## Rigorous XAI approaches for our models

| Method | What it does | Per-prediction? | Model support |
|--------|-------------|----------------|---------------|
| **SHAP** | Shapley values — each feature's additive contribution to the prediction | Yes | Native for XGBoost, KernelSHAP for LR/RF |
| **Feature importances** | Global ranking of feature influence | No (global only) | Built-in for RF/XGBoost |
| **LIME** | Local linear model approximation around one prediction | Yes | Model-agnostic |
| **Partial Dependence Plots** | How a feature affects predictions on average | No (global) | Any model |

## Recommendation: SHAP

SHAP is the right choice because:
- Per-prediction explanations (not just global)
- Mathematically grounded (Shapley values from cooperative game theory)
- XGBoost has native TreeSHAP support — very fast
- Works for all 3 of our models (TreeSHAP for RF/XGBoost, LinearSHAP for LR)
- Industry standard for model explainability in finance/credit risk

## Proposed architecture change

### Current flow
```
run-inference → explain-risk (LLM guesses from raw record)
```

### Proposed flow
```
run-inference (+ SHAP computation) → explain-risk (LLM translates SHAP values to business language)
```

### What changes in run-inference
1. After `model.predict_proba()`, compute SHAP values for the predicted row
2. Extract top-5 features by |SHAP value| with direction (positive = pushes toward predicted class, negative = pushes away)
3. Include SHAP values in `prediction_output`

### What changes in explain-risk
1. LLM receives SHAP-grounded feature contributions instead of guessing
2. Prompt says: "The model's prediction was driven by these feature contributions: [SHAP top-5]. Translate this into business language."
3. The explanation is now **faithful** — it reflects what the model actually computed

### What to add to evaluate-models
1. Confusion matrix (rubric requires it)
2. Optional: global SHAP summary (feature importance across all test predictions)

## Dependencies
- `shap` package (add to requirements.txt)
- Confusion matrix: `sklearn.metrics.confusion_matrix` (already available)
