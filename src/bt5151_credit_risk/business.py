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
    shap_contributions=None,
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
        "shap_contributions": shap_contributions,
        "global_shap_importance": global_shap_importance,
        "selection_justification": selection_justification,
    }
    if analysis_bundle_summary:
        payload["analysis_bundle_summary"] = analysis_bundle_summary
    return _call_json_agent(system_prompt, payload, caller="explain-risk")


# Turn the explanation into a suggested next action.
def recommend_action(risk_explanation, prediction_output=None):
    system_prompt = (
        "You are a business operations specialist interpreting a classification model's output. "
        "Return only valid JSON with keys: action, reason. "
        "Choose a concise action code that reflects the appropriate business response to the predicted class "
        "(e.g., escalate, review, monitor, standard_handling). "
        "Write the reason for a non-technical business user."
    )
    payload = {
        "risk_explanation": risk_explanation,
        "prediction_output": prediction_output,
    }
    return _call_json_agent(system_prompt, payload, caller="recommend-action")
