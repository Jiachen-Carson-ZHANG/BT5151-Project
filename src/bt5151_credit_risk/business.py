def explain_risk(predicted_label, probabilities):
    confidence = max(probabilities.values())
    if confidence >= 0.75:
        confidence_band = "high"
    elif confidence >= 0.55:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    risk_level = {"Good": "low", "Standard": "moderate", "Poor": "high"}[predicted_label]

    return {
        "predicted_label": predicted_label,
        "risk_level": risk_level,
        "confidence_band": confidence_band,
        "summary": f"Customer is assessed as {risk_level} risk with {confidence_band} confidence.",
    }


def recommend_action(explanation):
    if explanation["risk_level"] == "high":
        return {
            "action": "manual_review",
            "reason": "Escalate due to likely poor credit standing.",
        }
    if explanation["risk_level"] == "moderate":
        return {
            "action": "monitor_account",
            "reason": "Review recent behavior and monitor next cycle.",
        }
    return {
        "action": "standard_handling",
        "reason": "Continue standard treatment.",
    }
