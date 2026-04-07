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
