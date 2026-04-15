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


def test_explain_risk_can_embed_recommended_action(monkeypatch):
    def fake_call_json_agent(system_prompt, payload, **kwargs):
        assert payload["predicted_label"] == "Poor"
        return {
            "predicted_label": "Poor",
            "risk_level": "high",
            "confidence_band": "high",
            "summary": "Customer appears high risk because repayment indicators are poor.",
            "recommended_action": {
                "action": "manual_review",
                "urgency": "within_24h",
                "rationale": "Escalate due to likely poor credit standing.",
            },
        }

    monkeypatch.setattr(business, "_call_json_agent", fake_call_json_agent)
    explanation = business.explain_risk(
        "Poor",
        {"Poor": 0.82, "Standard": 0.15, "Good": 0.03},
    )
    assert explanation["recommended_action"]["action"] == "manual_review"
