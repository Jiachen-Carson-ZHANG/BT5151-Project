import bt5151_credit_risk.business as business


def test_explain_risk_returns_business_language(monkeypatch):
    def fake_call_json_agent(system_prompt, payload, **kwargs):
        assert payload["predicted_label"] == "Poor"
        return {
            "predicted_label": "Poor",
            "risk_level": "high",
            "confidence_band": "high",
            "summary": "Customer appears high risk because repayment indicators are poor.",
        }

    monkeypatch.setattr(business, "_call_json_agent", fake_call_json_agent)
    explanation = business.explain_risk(
        "Poor",
        {"Poor": 0.82, "Standard": 0.15, "Good": 0.03},
    )

    assert explanation["risk_level"] == "high"
    assert "summary" in explanation


def test_recommend_action_escalates_high_risk(monkeypatch):
    def fake_call_json_agent(system_prompt, payload, **kwargs):
        assert payload["risk_explanation"]["risk_level"] == "high"
        return {
            "action": "manual_review",
            "reason": "Escalate due to likely poor credit standing.",
        }

    monkeypatch.setattr(business, "_call_json_agent", fake_call_json_agent)
    action = business.recommend_action({"risk_level": "high", "confidence_band": "high"})
    assert action["action"] == "manual_review"
