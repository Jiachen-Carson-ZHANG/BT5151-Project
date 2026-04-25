from bt5151_credit_risk.feature_eligibility import apply_feature_eligibility


def test_apply_feature_eligibility_separates_raw_and_model_eligible_features():
    raw_top_features = [
        {"column": "Customer_ID", "mutual_information": 0.91, "column_type": "categorical"},
        {"column": "SSN", "mutual_information": 0.88, "column_type": "categorical"},
        {"column": "Age", "mutual_information": 0.42, "column_type": "numeric"},
        {"column": "Outstanding_Debt", "mutual_information": 0.33, "column_type": "numeric"},
        {"column": "Name", "mutual_information": 0.29, "column_type": "categorical"},
    ]

    feature_profiles = {
        "Customer_ID": {"nunique": 12500, "non_null_count": 100000},
        "SSN": {"nunique": 12501, "non_null_count": 100000},
        "Age": {"nunique": 80, "non_null_count": 100000},
        "Outstanding_Debt": {"nunique": 9000, "non_null_count": 100000},
        "Name": {"nunique": 10139, "non_null_count": 90015},
    }

    dataset_policy_spec = {
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
        "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
    }

    result = apply_feature_eligibility(
        raw_top_features=raw_top_features,
        feature_profiles=feature_profiles,
        dataset_policy_spec=dataset_policy_spec,
    )

    assert [entry["column"] for entry in result["model_eligible_top_discriminative_features"]] == [
        "Age",
        "Outstanding_Debt",
    ]
    assert [entry["column"] for entry in result["raw_top_discriminative_features"]] == [
        "Customer_ID",
        "SSN",
        "Age",
        "Outstanding_Debt",
        "Name",
    ]
    assert {entry["column"] for entry in result["leakage_alerts"]} == {"Customer_ID", "SSN", "Name"}
    assert any(entry["reason"] == "group_column" for entry in result["leakage_alerts"])
    assert any(entry["reason"] == "identifier_column" for entry in result["leakage_alerts"])


def test_apply_feature_eligibility_uses_review_first_name_heuristics():
    raw_top_features = [
        {"column": "CustomerSegmentName", "mutual_information": 0.51, "column_type": "categorical"},
        {"column": "Credit_Mix", "mutual_information": 0.44, "column_type": "categorical"},
    ]
    feature_profiles = {
        "CustomerSegmentName": {"nunique": 4, "non_null_count": 1000},
        "Credit_Mix": {"nunique": 3, "non_null_count": 1000},
    }

    result = apply_feature_eligibility(
        raw_top_features=raw_top_features,
        feature_profiles=feature_profiles,
        dataset_policy_spec={"target_column": "Credit_Score"},
    )

    assert [entry["column"] for entry in result["model_eligible_top_discriminative_features"]] == [
        "CustomerSegmentName",
        "Credit_Mix",
    ]
    assert result["leakage_alerts"] == [
        {
            "column": "CustomerSegmentName",
            "mutual_information": 0.51,
            "severity": "review",
            "reason": "identifier_like_name",
            "message": "Column name looks identifier-like but was kept because policy/spec did not block it and uniqueness was not near-identifier level.",
        }
    ]


def test_apply_feature_eligibility_respects_real_leakage_policy_contract():
    raw_top_features = [
        {"column": "Name", "mutual_information": 0.61, "column_type": "categorical"},
        {"column": "Outstanding_Debt", "mutual_information": 0.44, "column_type": "numeric"},
    ]
    feature_profiles = {
        "Name": {"nunique": 10139, "non_null_count": 90015},
        "Outstanding_Debt": {"nunique": 13178, "non_null_count": 100000},
    }

    result = apply_feature_eligibility(
        raw_top_features=raw_top_features,
        feature_profiles=feature_profiles,
        dataset_policy_spec={
            "target_column": "Credit_Score",
            "leakage_policy": {"columns_to_drop": ["Name"]},
        },
    )

    assert [entry["column"] for entry in result["model_eligible_top_discriminative_features"]] == [
        "Outstanding_Debt",
    ]
    assert result["leakage_alerts"][0]["column"] == "Name"
    assert result["leakage_alerts"][0]["reason"] == "leakage_drop_rule"


def test_apply_feature_eligibility_does_not_block_near_unique_numeric_measurements():
    raw_top_features = [
        {"column": "Credit_Utilization_Ratio", "mutual_information": 0.19, "column_type": "numeric"},
        {"column": "Monthly_Balance", "mutual_information": 0.11, "column_type": "numeric"},
    ]
    feature_profiles = {
        "Credit_Utilization_Ratio": {"nunique": 99980, "non_null_count": 100000},
        "Monthly_Balance": {"nunique": 98792, "non_null_count": 100000},
    }

    result = apply_feature_eligibility(
        raw_top_features=raw_top_features,
        feature_profiles=feature_profiles,
        dataset_policy_spec={"target_column": "Credit_Score"},
    )

    assert [entry["column"] for entry in result["model_eligible_top_discriminative_features"]] == [
        "Credit_Utilization_Ratio",
        "Monthly_Balance",
    ]
    assert result["leakage_alerts"] == []
