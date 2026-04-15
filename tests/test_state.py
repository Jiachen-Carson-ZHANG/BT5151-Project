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
    profile = build_dataset_profile(sample_frame, target_column="Credit_Score")
    assert "row_count" in profile
    assert "target_distribution" in profile
    assert "missing_counts" in profile


def test_dataset_profile_without_target_column(sample_frame):
    profile = build_dataset_profile(sample_frame)
    assert "row_count" in profile
    assert "missing_counts" in profile
    assert "target_distribution" not in profile


def test_state_has_expected_keys():
    state = CreditRiskState(raw_dataset_path="train.csv")
    assert state.raw_dataset_path == "train.csv"
    assert state.run_id is None
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
    assert state.feature_engineering_code is None
    assert state.feature_engineering_code_review is None
    assert state.feature_engineering_execution_log is None
    assert state.feature_engineering_validation_report is None
    assert state.feature_engineering_hypothesis is None
    assert state.feature_engineering_attempt_count == 0
    assert state.global_shap_importance is None
    assert state.eda_report is None
    assert state.learning_curves is None
    assert state.tuning_trial_history is None
    assert state.eda_hypotheses is None
    assert state.training_diagnostics is None
    assert state.train_group_values is None
    assert state.test_group_values is None
    assert state.train_time_values is None
    assert state.test_time_values is None
    assert state.train_views is None
    assert state.test_views is None
    assert state.full_feature_frames_by_view is None
    assert state.feature_columns_by_view is None
    assert state.model_view_map is None
    assert state.global_xai_results is None
    assert state.local_xai_cases is None
    assert state.global_xai_interpretation is None
    assert state.local_xai_interpretation is None
    assert state.analysis_bundle is None
    assert state.analysis_bundle_summary is None
    assert state.cache_log_path is None
    assert state.cache_bundle_path is None
    assert state.cache_saved_at is None


def test_state_accepts_new_preprocessing_fields():
    state = CreditRiskState(
        raw_dataset_path="train.csv",
        run_id="20260415_131136",
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
    assert state.run_id == "20260415_131136"
