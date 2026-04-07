from bt5151_credit_risk.llm import call_json_response


def _call_json_agent(system_prompt, payload, caller="business"):
    return call_json_response(system_prompt, payload, caller=caller)


# Turn the model prediction into a short business-friendly explanation.
def explain_risk(
    predicted_label,
    probabilities,
    selected_model_name=None,
    evaluation_metrics=None,
    source_record=None,
):
    system_prompt = (
        "You are a credit risk analyst. "
        "Return only valid JSON with keys: "
        "predicted_label, risk_level, confidence_band, summary. "
        "Use risk_level values low, moderate, or high. "
        "Use confidence_band values low, medium, or high. "
        "Write the summary for a non-technical business user."
    )
    payload = {
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "selected_model_name": selected_model_name,
        "evaluation_metrics": evaluation_metrics,
        "source_record": source_record,
    }
    return _call_json_agent(system_prompt, payload, caller="explain-risk")


# Turn the explanation into a suggested next action.
def recommend_action(risk_explanation, prediction_output=None):
    system_prompt = (
        "You are a credit operations specialist. "
        "Return only valid JSON with keys: action, reason. "
        "Choose a concise action code such as manual_review, monitor_account, or standard_handling. "
        "Write the reason for a non-technical business user."
    )
    payload = {
        "risk_explanation": risk_explanation,
        "prediction_output": prediction_output,
    }
    return _call_json_agent(system_prompt, payload, caller="recommend-action")
