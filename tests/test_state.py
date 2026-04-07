import pandas as pd
import pytest

from bt5151_credit_risk.profile import build_dataset_profile
from bt5151_credit_risk.state import CreditRiskState


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        [
            {"Credit_Score": "Good", "Age": 25, "Outstanding_Debt": 1000.0},
            {"Credit_Score": "Poor", "Age": None, "Outstanding_Debt": 2500.0},
        ]
    )


def test_dataset_profile_contains_core_fields(sample_frame):
    profile = build_dataset_profile(sample_frame)
    assert "row_count" in profile
    assert "target_distribution" in profile
    assert "missing_counts" in profile


def test_state_has_expected_keys():
    state = CreditRiskState(raw_dataset_path="train.csv")
    assert state.raw_dataset_path == "train.csv"
    assert state.dataset_policy_spec is None
    assert state.column_transform_spec is None
    assert state.preprocessing_code is None
    assert state.preprocessing_codegen_metadata is None
    assert state.preprocessing_code_review is None
    assert state.preprocessing_workspace is None
    assert state.preprocessing_artifacts is None
    assert state.preprocessing_execution_log is None
    assert state.preprocessing_execution_report is None
    assert state.preprocessing_audit_report is None
    assert state.preprocessing_validation_report is None
    assert state.preprocessing_attempt_count == 0


def test_state_accepts_new_preprocessing_fields():
    state = CreditRiskState(
        raw_dataset_path="train.csv",
        preprocessing_code={"code": "print('hello')"},
        preprocessing_codegen_metadata={"model": "gpt-4o-mini"},
        preprocessing_code_review={"passed": True},
        preprocessing_workspace="/tmp/run-1",
        preprocessing_artifacts={"feature_frame": "feature_frame.csv"},
        preprocessing_execution_log={"stdout": "ok"},
        preprocessing_validation_report={"passed": True},
        preprocessing_attempt_count=2,
    )

    assert state.preprocessing_code == {"code": "print('hello')"}
    assert state.preprocessing_attempt_count == 2
