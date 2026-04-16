import logging

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt

logger = logging.getLogger(__name__)


# Compute the core multi-class metrics used for model comparison.
def compute_multiclass_metrics(y_true, y_pred, class_names):
    label_indices = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class": {name: report[name] for name in class_names},
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


# Pick the winning model from the evaluation summary.
def choose_best_model(results):
    model_name = max(
        results,
        key=lambda name: (results[name]["macro_f1"], results[name]["weighted_f1"]),
    )
    return {
        "model_name": model_name,
        "justification": (
            f"Selected {model_name} based on stronger macro_f1 and weighted_f1."
        ),
    }


# LLM-driven model selection with hypothesis validation.
def reason_model_selection(
    evaluation_results,
    tuning_results=None,
    global_shap_importance=None,
    eda_top_features=None,
    fe_hypothesis=None,
    class_names=None,
):
    system_prompt = load_skill_prompt("reason-model-selection")
    payload = {
        "evaluation_results": evaluation_results,
        "class_names": class_names or [],
    }
    if tuning_results:
        payload["tuning_results"] = tuning_results
    if global_shap_importance:
        payload["global_shap_importance"] = global_shap_importance
    if eda_top_features:
        payload["eda_top_features"] = eda_top_features
    if fe_hypothesis:
        payload["fe_hypothesis"] = fe_hypothesis

    metric_best = choose_best_model(evaluation_results)

    try:
        result = call_json_response(system_prompt, payload, caller="reason-model-selection") or {}
        advisory_model_name = result.get("model_name")
        if advisory_model_name and advisory_model_name != metric_best["model_name"]:
            logger.warning(
                "    LLM model selection suggested %s, but metric-best %s remains authoritative",
                advisory_model_name,
                metric_best["model_name"],
            )

        advisory = dict(metric_best)
        llm_justification = (result.get("justification") or "").strip()
        if llm_justification:
            advisory["justification"] = (
                f"{metric_best['justification']} LLM rationale: {llm_justification}"
            )
        else:
            advisory["justification"] = metric_best["justification"]
        if advisory_model_name:
            advisory["llm_model_name"] = advisory_model_name
        if "hypothesis_validation" in result:
            advisory["hypothesis_validation"] = result.get("hypothesis_validation")
        return advisory
    except Exception as exc:
        logger.warning("    LLM model selection failed (%s), falling back to metric-based", exc)

    # Fallback to hardcoded max.
    fallback = choose_best_model(evaluation_results)
    fallback["hypothesis_validation"] = None
    return fallback
