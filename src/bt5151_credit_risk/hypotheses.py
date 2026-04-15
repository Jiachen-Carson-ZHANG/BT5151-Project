import logging

from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt

logger = logging.getLogger(__name__)


def generate_eda_hypotheses(eda_report: dict, dataset_profile: dict) -> dict:
    """LLM interprets programmatic EDA statistics into three-tier directional hypotheses."""
    system_prompt = load_skill_prompt("generate-eda-hypotheses")
    payload = {
        "eda_report": eda_report,
        "dataset_profile": dataset_profile,
    }
    return call_json_response(system_prompt, payload, caller="generate-eda-hypotheses")


def interpret_xai_evidence(
    global_xai_results: dict,
    local_xai_cases: list[dict],
    class_names: list[str],
    training_diagnostics: dict | None = None,
    eda_hypotheses: dict | None = None,
    feature_engineering_hypothesis: dict | None = None,
) -> dict:
    """LLM interprets global + local XAI results into observations, insights, hypotheses."""
    system_prompt = load_skill_prompt("interpret-xai-evidence")
    payload = {
        "global_xai_results": _compact_xai_for_llm(global_xai_results),
        "local_xai_cases": local_xai_cases,
        "class_names": class_names,
    }
    if training_diagnostics:
        payload["training_diagnostics"] = {
            k: training_diagnostics[k]
            for k in ("per_class_analysis", "capacity_analysis", "new_hypotheses")
            if k in training_diagnostics
        }
    if eda_hypotheses:
        payload["eda_hypotheses"] = {
            k: eda_hypotheses[k]
            for k in ("tested_predictions", "supported_conjectures", "model_selection_prediction")
            if k in eda_hypotheses
        }
    if feature_engineering_hypothesis:
        payload["feature_engineering_hypothesis"] = feature_engineering_hypothesis
    return call_json_response(system_prompt, payload, caller="interpret-xai-evidence")


def _compact_xai_for_llm(xai_results: dict) -> dict:
    """Strip large arrays (beeswarm, raw PFI) from XAI results to fit LLM context."""
    compact = {"methods_used": xai_results.get("methods_used", [])}
    if "shap" in xai_results:
        compact["shap_importance"] = xai_results["shap"].get("importance", [])[:15]
        # Include dependence summary but not full beeswarm arrays
        dep = xai_results["shap"].get("dependence_data", {})
        if dep:
            compact["shap_dependence_features"] = list(dep.keys())
    if "pfi" in xai_results:
        compact["pfi_grouped"] = xai_results["pfi"].get("grouped", [])[:15]
    if "pdp" in xai_results:
        # Send grid + values for each feature (already small)
        compact["pdp"] = xai_results["pdp"]
    if "ale" in xai_results:
        compact["ale"] = xai_results["ale"]
    return compact


def generate_training_diagnostics(
    evaluation_results: dict,
    tuning_results: dict,
    learning_curves: dict | None,
    eda_hypotheses: dict | None,
    feature_engineering_hypothesis: dict | None,
    class_names: list[str],
    confidence_stats: dict | None = None,
) -> dict:
    """LLM interprets training results, validates EDA hypotheses, generates new ones."""
    system_prompt = load_skill_prompt("generate-training-diagnostics")
    payload = {
        "evaluation_results": evaluation_results,
        "tuning_results": tuning_results,
        "class_names": class_names,
    }
    if learning_curves:
        payload["learning_curves"] = learning_curves
    if eda_hypotheses:
        payload["eda_hypotheses"] = eda_hypotheses
    if feature_engineering_hypothesis:
        payload["feature_engineering_hypothesis"] = feature_engineering_hypothesis
    if confidence_stats:
        payload["confidence_stats"] = confidence_stats
    return call_json_response(system_prompt, payload, caller="generate-training-diagnostics")
