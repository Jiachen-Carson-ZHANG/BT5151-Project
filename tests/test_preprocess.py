import pandas as pd
import pytest

from bt5151_credit_risk.preprocess import audit_preprocessing_output
from bt5151_credit_risk.preprocess import execute_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        [
            {
                "ID": "0x1",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Good",
                "Age": 25,
                "Outstanding_Debt": 1000.0,
                "Occupation": "Engineer",
            },
            {
                "ID": "0x2",
                "Customer_ID": "CUS_A",
                "Name": "Alice",
                "SSN": "111-11-1111",
                "Credit_Score": "Standard",
                "Age": 26,
                "Outstanding_Debt": 1500.0,
                "Occupation": "Engineer",
            },
            {
                "ID": "0x3",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 40,
                "Outstanding_Debt": 3000.0,
                "Occupation": "Analyst",
            },
            {
                "ID": "0x4",
                "Customer_ID": "CUS_B",
                "Name": "Bob",
                "SSN": "222-22-2222",
                "Credit_Score": "Poor",
                "Age": 41,
                "Outstanding_Debt": 3200.0,
                "Occupation": "Analyst",
            },
            {
                "ID": "0x5",
                "Customer_ID": "CUS_C",
                "Name": "Cara",
                "SSN": "333-33-3333",
                "Credit_Score": "Good",
                "Age": 31,
                "Outstanding_Debt": 800.0,
                "Occupation": "Manager",
            },
            {
                "ID": "0x6",
                "Customer_ID": "CUS_D",
                "Name": "Dan",
                "SSN": "444-44-4444",
                "Credit_Score": "Standard",
                "Age": 29,
                "Outstanding_Debt": 2200.0,
                "Occupation": "Teacher",
            },
        ]
    )


def test_generate_dataset_policy_spec_returns_structured_policy(sample_frame, monkeypatch):
    def fake_agent(system_prompt, payload):
        return {
            "task_type": "multiclass_classification",
            "target_column": "Credit_Score",
            "group_column": "Customer_ID",
            "identifier_columns": ["ID", "Name", "SSN"],
            "split_strategy": {"type": "grouped_holdout", "test_size": 0.2},
            "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
            "imbalance_strategy": {"method": "none"},
            "feature_policy": {"categorical_encoding": "one_hot"},
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_agent", fake_agent)
    profile = {
        "row_count": len(sample_frame),
        "target_distribution": sample_frame["Credit_Score"].value_counts().to_dict(),
        "missing_counts": sample_frame.isna().sum().to_dict(),
    }

    policy = generate_dataset_policy_spec(sample_frame, profile)

    assert policy["target_column"] == "Credit_Score"
    assert policy["group_column"] == "Customer_ID"
    assert policy["split_strategy"]["type"] == "grouped_holdout"


def test_generate_column_transform_spec_returns_column_rules(sample_frame, monkeypatch):
    def fake_agent(system_prompt, payload):
        return {
            "columns": {
                "Age": {"action": "keep", "imputation": "median"},
                "Outstanding_Debt": {"action": "keep", "imputation": "median"},
                "Occupation": {"action": "keep", "encoding": "one_hot", "imputation": "mode"},
                "ID": {"action": "drop"},
                "Name": {"action": "drop"},
                "SSN": {"action": "drop"},
            }
        }

    monkeypatch.setattr("bt5151_credit_risk.preprocess._call_preprocess_agent", fake_agent)
    policy = {
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
    }

    spec = generate_column_transform_spec(sample_frame, policy)

    assert spec["columns"]["Age"]["imputation"] == "median"
    assert spec["columns"]["Occupation"]["encoding"] == "one_hot"


def _sample_policy_spec():
    return {
        "task_type": "multiclass_classification",
        "target_column": "Credit_Score",
        "group_column": "Customer_ID",
        "identifier_columns": ["ID", "Name", "SSN"],
        "split_strategy": {"type": "grouped_holdout", "test_size": 0.34},
        "leakage_rules": {"drop_columns": ["ID", "Name", "SSN"]},
        "imbalance_strategy": {"method": "none"},
        "feature_policy": {"categorical_encoding": "one_hot"},
    }


def _sample_column_transform_spec():
    return {
        "columns": {
            "ID": {"action": "drop"},
            "Customer_ID": {"action": "drop"},
            "Name": {"action": "drop"},
            "SSN": {"action": "drop"},
            "Credit_Score": {"action": "drop"},
            "Age": {"action": "keep", "imputation": "median"},
            "Outstanding_Debt": {"action": "keep", "imputation": "median"},
            "Occupation": {"action": "keep", "encoding": "one_hot", "imputation": "mode"},
        }
    }


def test_execute_preprocessing_drops_identifier_columns_and_splits_groups(sample_frame):
    result = execute_preprocessing(
        sample_frame,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert "ID" not in result.feature_frame.columns
    assert "Customer_ID" not in result.feature_frame.columns
    assert "Name" not in result.feature_frame.columns
    assert "SSN" not in result.feature_frame.columns
    assert "Credit_Score" not in result.feature_frame.columns
    assert set(result.train_groups).isdisjoint(set(result.test_groups))


def test_audit_preprocessing_output_reports_key_checks_passed(sample_frame):
    result = execute_preprocessing(
        sample_frame,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )
    audit_report = audit_preprocessing_output(
        sample_frame,
        result,
        _sample_policy_spec(),
        _sample_column_transform_spec(),
    )

    assert audit_report["checks"]["target_excluded"] is True
    assert audit_report["checks"]["group_leakage_free"] is True
    assert audit_report["checks"]["feature_frame_non_empty"] is True
