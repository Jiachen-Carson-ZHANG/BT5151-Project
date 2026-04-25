from typing import Any

from pydantic import BaseModel, ConfigDict


class CreditRiskState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_dataset_path: str
    run_id: str | None = None
    dataset_profile: dict | None = None
    eda_report: dict | None = None
    eda_hypotheses: dict | None = None
    dataset_policy_spec: dict | None = None
    column_transform_spec: dict | None = None
    preprocessing_code: dict | None = None
    preprocessing_codegen_metadata: dict | None = None
    preprocessing_codegen_snapshot_path: str | None = None
    preprocessing_code_review: dict | None = None
    preprocessing_rules: dict | None = None
    preprocessing_workspace: str | None = None
    preprocessing_raw_frame_path: str | None = None
    preprocessing_artifacts: dict | None = None
    preprocessing_execution_log: dict | None = None
    preprocessing_execution_report: dict | None = None
    preprocessing_audit_report: dict | None = None
    preprocessing_validation_report: dict | None = None
    preprocessing_attempt_count: int = 0
    preprocessing_last_role_violations: list | None = None
    preprocessing_escalation_triggered: bool = False
    feature_engineering_code: dict | None = None
    feature_engineering_codegen_metadata: dict | None = None
    feature_engineering_codegen_snapshot_path: str | None = None
    feature_engineering_code_review: dict | None = None
    feature_engineering_execution_log: dict | None = None
    feature_engineering_validation_report: dict | None = None
    feature_engineering_hypothesis: dict | None = None
    feature_engineering_attempt_count: int = 0
    feature_columns: list[str] | None = None
    class_names: list[str] | None = None
    label_to_id: dict | None = None
    id_to_label: dict | None = None
    raw_frame: Any | None = None
    full_feature_frame: Any | None = None
    full_feature_frames_by_view: dict | None = None
    train_frame: Any | None = None
    train_views: dict | None = None
    test_frame: Any | None = None
    test_views: dict | None = None
    train_target: Any | None = None
    test_target: Any | None = None
    feature_columns_by_view: dict | None = None
    model_view_map: dict | None = None
    deferred_categorical_columns: dict | None = None  # {col: nunique} for object-dtype preprocess outputs
    train_group_values: list | None = None
    test_group_values: list | None = None
    train_time_values: list | None = None
    test_time_values: list | None = None
    candidate_model_specs: dict | None = None
    trained_models: dict | None = None
    learning_curves: dict | None = None
    tuning_trial_history: dict | None = None
    tuning_artifact_path: str | None = None
    evaluation_results: dict | None = None
    training_diagnostics: dict | None = None
    selected_model_name: str | None = None
    selection_justification: str | None = None
    global_shap_importance: list | None = None
    global_xai_results: dict | None = None
    local_xai_cases: list[dict] | None = None
    global_xai_interpretation: dict | None = None
    local_xai_interpretation: dict | None = None
    shortcut_audit: dict | None = None
    analysis_bundle: dict | None = None
    analysis_bundle_summary: dict | None = None
    cache_log_path: str | None = None
    cache_bundle_path: str | None = None
    cache_trace_path: str | None = None
    cache_saved_at: str | None = None
    inference_input: dict | None = None
    prediction_output: dict | None = None
    risk_explanation: dict | None = None
    recommended_action: dict | None = None
