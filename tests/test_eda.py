from bt5151_credit_risk.eda import build_eda_report


def test_build_eda_report_keeps_backward_compatible_top_features_as_model_eligible(monkeypatch):
    monkeypatch.setattr(
        "bt5151_credit_risk.eda._compute_discriminative_features",
        lambda *args, **kwargs: [
            {"column": "Customer_ID", "mutual_information": 0.91, "column_type": "categorical"},
            {"column": "Age", "mutual_information": 0.42, "column_type": "numeric"},
            {"column": "Outstanding_Debt", "mutual_information": 0.33, "column_type": "numeric"},
        ],
    )

    import pandas as pd

    df = pd.DataFrame(
        {
            "Customer_ID": ["C1", "C1", "C2", "C2"],
            "Age": [30, 31, 50, 51],
            "Outstanding_Debt": [1000.0, 1200.0, 5000.0, 5200.0],
            "Credit_Score": ["Good", "Good", "Poor", "Poor"],
        }
    )

    report = build_eda_report(
        df,
        "Credit_Score",
        dataset_policy_spec={
            "target_column": "Credit_Score",
            "group_column": "Customer_ID",
            "identifier_columns": [],
        },
    )

    assert [entry["column"] for entry in report["raw_top_discriminative_features"]] == [
        "Customer_ID",
        "Age",
        "Outstanding_Debt",
    ]
    assert [entry["column"] for entry in report["model_eligible_top_discriminative_features"]] == [
        "Age",
        "Outstanding_Debt",
    ]
    assert [entry["column"] for entry in report["top_discriminative_features"]] == [
        "Age",
        "Outstanding_Debt",
    ]
    assert report["leakage_alerts"][0]["column"] == "Customer_ID"
