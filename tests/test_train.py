from bt5151_credit_risk.train import build_candidate_models


def test_candidate_models_include_two_baselines():
    models = build_candidate_models()
    assert set(models) == {"logistic_regression", "random_forest"}
