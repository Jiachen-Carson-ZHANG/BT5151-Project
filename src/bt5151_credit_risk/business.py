from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt


def _call_json_agent(system_prompt, payload, caller="business"):
    return call_json_response(system_prompt, payload, caller=caller)


# Turn the model prediction into a short business-friendly explanation.
def explain_risk(
    predicted_label,
    probabilities,
    selected_model_name=None,
    evaluation_metrics=None,
    source_record=None,
    shap_waterfall=None,
    pdp_position=None,
    confidence_diagnosis=None,
    nearest_casebook_case=None,
    global_shap_importance=None,
    analysis_bundle_summary=None,
    selection_justification=None,
):
    system_prompt = load_skill_prompt("explain-risk")
    payload = {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "selected_model_name": selected_model_name,
        "evaluation_metrics": evaluation_metrics,
        "source_record": source_record,
        "global_shap_importance": global_shap_importance,
        "selection_justification": selection_justification,
    }
    if shap_waterfall:
        payload["shap_waterfall"] = shap_waterfall
    if pdp_position:
        payload["pdp_position"] = pdp_position
    if confidence_diagnosis:
        payload["confidence_diagnosis"] = confidence_diagnosis
    if nearest_casebook_case:
        payload["nearest_casebook_case"] = nearest_casebook_case
    if analysis_bundle_summary:
        payload["analysis_bundle_summary"] = analysis_bundle_summary
    return _call_json_agent(system_prompt, payload, caller="explain-risk")
