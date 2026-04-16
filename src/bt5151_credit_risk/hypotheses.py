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


def _strip_beeswarm(global_xai_results: dict) -> dict:
    """Strip only the raw beeswarm SHAP value arrays (~5k×37 numbers) — everything
    else is semantic/small and passes through verbatim so the LLM sees full evidence."""
    if not global_xai_results:
        return {}
    out = dict(global_xai_results)
    shap = out.get("shap")
    if isinstance(shap, dict) and "beeswarm_data" in shap:
        shap = dict(shap)
        shap["beeswarm_feature_names"] = list(shap.get("beeswarm_data", {}).keys())
        shap.pop("beeswarm_data", None)
        out["shap"] = shap
    return out


def interpret_global_xai(
    global_xai_results: dict,
    class_names: list[str],
    training_diagnostics: dict | None = None,
    eda_hypotheses: dict | None = None,
    feature_engineering_hypothesis: dict | None = None,
    shortcut_audit: dict | None = None,
) -> dict:
    """LLM interprets global XAI (SHAP/PFI/PDP/ALE) into consensus + insights + hypotheses."""
    system_prompt = load_skill_prompt("interpret-global-xai")
    payload = {
        "global_xai_results": _strip_beeswarm(global_xai_results),
        "class_names": class_names,
    }
    if training_diagnostics:
        payload["training_diagnostics"] = training_diagnostics
    if eda_hypotheses:
        payload["eda_hypotheses"] = eda_hypotheses
    if feature_engineering_hypothesis:
        payload["feature_engineering_hypothesis"] = feature_engineering_hypothesis
    if shortcut_audit:
        payload["shortcut_audit"] = shortcut_audit
    return call_json_response(system_prompt, payload, caller="interpret-global-xai")


def interpret_local_xai(
    local_xai_cases: list[dict],
    class_names: list[str],
    global_xai_interpretation: dict | None = None,
    global_xai_results: dict | None = None,
    training_diagnostics: dict | None = None,
) -> dict:
    """LLM interprets the casebook (representative / borderline / misclassification per class)
    into per-class stories, boundary analysis, and fresh hypotheses that cite specific cases."""
    system_prompt = load_skill_prompt("interpret-local-xai")
    payload = {
        "local_xai_cases": local_xai_cases,
        "class_names": class_names,
    }
    if global_xai_interpretation:
        payload["global_xai_interpretation"] = global_xai_interpretation
    if global_xai_results:
        # Pass a thin reference — top SHAP/PFI rankings — so local reasoning can cite globals.
        thin = {}
        shap_imp = (global_xai_results.get("shap") or {}).get("importance")
        if shap_imp:
            thin["shap_top"] = shap_imp[:10]
        pfi_grp = (global_xai_results.get("pfi") or {}).get("grouped")
        if pfi_grp:
            thin["pfi_grouped_top"] = pfi_grp[:10]
        if thin:
            payload["global_xai_reference"] = thin
    if training_diagnostics:
        payload["training_diagnostics"] = training_diagnostics
    return call_json_response(system_prompt, payload, caller="interpret-local-xai")


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
