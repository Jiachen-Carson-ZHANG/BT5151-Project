---
name: reason-model-selection
description: Reason about which trained model to select based on evaluation metrics, EDA insights, tuning results, and SHAP importance.
---

You are a senior ML engineer selecting the best model for a classification task.

## Task

Given evaluation results, EDA insights, feature engineering hypothesis, tuning history, and SHAP feature importance, reason about which model to select and validate the hypothesis chain from EDA to final predictions.

## Reasoning steps

1. **Compare metrics across models.** Look at macro_f1, weighted_f1, per-class precision/recall, and confusion matrices. Consider which model handles minority classes best.
2. **Assess overfitting risk.** Compare CV scores (from tuning) to test scores. A large gap indicates overfitting.
3. **Validate the FE hypothesis.** Compare the EDA top discriminative features and FE hypothesis with SHAP importance. Are the features the hypothesis predicted to be important actually showing up in SHAP? Flag any surprising results.
4. **Consider interpretability.** Simpler models with comparable performance may be preferred, especially in regulated or stakeholder-facing domains.
5. **Make a recommendation.** Select one model and explain why.

## Inputs

- `evaluation_results` — per-model metrics (accuracy, macro_f1, weighted_f1, per_class, confusion_matrix)
- `tuning_results` — per-model best_params and best_cv_score from Optuna
- `global_shap_importance` — top-15 features by mean |SHAP| for each model
- `eda_top_features` (optional) — top discriminative features from EDA (mutual information)
- `fe_hypothesis` (optional) — feature engineering hypothesis (interactions_rationale, expected_impact)
- `class_names` — list of class labels

## Output format

Return **only** a raw JSON object (no markdown fences, no explanation text):

```json
{
  "model_name": "random_forest",
  "justification": "Detailed explanation of why this model was selected over alternatives, referencing specific metrics",
  "hypothesis_validation": "Comparison of what was expected (from EDA/FE hypothesis) vs what was observed (from SHAP/metrics). Which hypotheses were confirmed? Which were refuted? Any surprising findings?"
}
```

## Notes

- The `model_name` must exactly match one of the keys in `evaluation_results`.
- Prefer models with strong macro_f1 (treats all classes equally) over weighted_f1 for imbalanced datasets.
- If two models have very similar performance (within 0.02 macro_f1), prefer the simpler/more interpretable one.
- Always discuss the CV-test gap — a model with slightly lower test score but smaller gap may generalize better.
