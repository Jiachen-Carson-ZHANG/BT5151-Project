from bt5151_credit_risk.business import explain_risk, recommend_action


def test_explain_risk_returns_business_language():
    explanation = explain_risk("Poor", {"Poor": 0.82, "Standard": 0.15, "Good": 0.03})
    assert "risk_level" in explanation
    assert "summary" in explanation


def test_recommend_action_escalates_high_risk():
    action = recommend_action({"risk_level": "high", "confidence_band": "high"})
    assert action["action"] == "manual_review"
