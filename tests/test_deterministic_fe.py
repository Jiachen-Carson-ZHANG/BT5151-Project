import pandas as pd

from bt5151_credit_risk.deterministic_fe import engineer_deterministic_feature_views


def test_engineer_deterministic_feature_views_adds_safe_ratios_and_lineage():
    train = pd.DataFrame(
        {
            "Total_EMI_per_month": [1000.0, 500.0],
            "Monthly_Inhand_Salary": [4000.0, 2500.0],
            "Outstanding_Debt": [12000.0, 10000.0],
            "Annual_Income": [48000.0, 36000.0],
            "Occupation": ["Engineer", "Doctor"],
        }
    )
    test = pd.DataFrame(
        {
            "Total_EMI_per_month": [750.0],
            "Monthly_Inhand_Salary": [3000.0],
            "Outstanding_Debt": [9000.0],
            "Annual_Income": [42000.0],
            "Occupation": ["Engineer"],
        }
    )

    result = engineer_deterministic_feature_views(
        train,
        test,
        deferred_categorical_columns={"Occupation": 2},
    )

    engineered_train = result["train_frame"]
    lineage = result["lineage"]
    report = result["report"]

    assert "EMI_to_Salary_Ratio" in engineered_train.columns
    assert "Debt_to_Income_Ratio" in engineered_train.columns
    assert engineered_train["EMI_to_Salary_Ratio"].round(4).tolist() == [0.25, 0.2]
    assert any(
        entry["feature"] == "EMI_to_Salary_Ratio" and entry["operation"] == "ratio"
        for entry in lineage["derived_features"]
    )
    assert any(
        entry["column"] == "EMI_to_Salary_Ratio" and "formula" in entry
        for entry in report["added"]
    )
