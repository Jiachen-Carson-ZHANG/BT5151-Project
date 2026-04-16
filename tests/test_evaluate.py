import bt5151_credit_risk.evaluate as evaluate_module
from bt5151_credit_risk.evaluate import compute_multiclass_metrics, choose_best_model


def test_metrics_include_bt5151_requirements():
    metrics = compute_multiclass_metrics([0, 1, 2], [0, 1, 2], ["Poor", "Standard", "Good"])
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics


def test_model_selection_returns_name_and_reason():
    selected = choose_best_model(
        {
            "logistic_regression": {"macro_f1": 0.70, "weighted_f1": 0.76},
            "random_forest": {"macro_f1": 0.78, "weighted_f1": 0.80},
        }
    )
    assert selected["model_name"] == "random_forest"
    assert "justification" in selected


def test_reason_model_selection_keeps_metric_best_when_llm_disagrees(monkeypatch):
    def fake_call_json_response(*args, **kwargs):
        return {"model_name": "logistic_regression", "justification": "LLM prefers LR"}

    monkeypatch.setattr(evaluate_module, "call_json_response", fake_call_json_response)

    result = evaluate_module.reason_model_selection(
        {
            "random_forest": {"macro_f1": 0.82, "weighted_f1": 0.80},
            "logistic_regression": {"macro_f1": 0.71, "weighted_f1": 0.77},
        },
        class_names=["Good", "Poor", "Standard"],
    )

    assert result["model_name"] == "random_forest"
    assert result["llm_model_name"] == "logistic_regression"
    assert "LLM rationale: LLM prefers LR" in result["justification"]
