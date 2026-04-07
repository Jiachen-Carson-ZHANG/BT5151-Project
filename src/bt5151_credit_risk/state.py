from typing import Any

from pydantic import BaseModel, ConfigDict


class CreditRiskState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_dataset_path: str
    dataset_profile: dict | None = None
    dataset_policy_spec: dict | None = None
    column_transform_spec: dict | None = None
    preprocessing_code: dict | None = None
    preprocessing_codegen_metadata: dict | None = None
    preprocessing_code_review: dict | None = None
    preprocessing_rules: dict | None = None
    preprocessing_workspace: str | None = None
    preprocessing_artifacts: dict | None = None
    preprocessing_execution_log: dict | None = None
    preprocessing_execution_report: dict | None = None
    preprocessing_audit_report: dict | None = None
    preprocessing_validation_report: dict | None = None
    preprocessing_attempt_count: int = 0
    feature_columns: list[str] | None = None
    class_names: list[str] | None = None
    label_to_id: dict | None = None
    id_to_label: dict | None = None
    raw_frame: Any | None = None
    full_feature_frame: Any | None = None
    train_frame: Any | None = None
    test_frame: Any | None = None
    train_target: Any | None = None
    test_target: Any | None = None
    candidate_model_specs: dict | None = None
    trained_models: dict | None = None
    evaluation_results: dict | None = None
    selected_model_name: str | None = None
    selection_justification: str | None = None
    inference_input: dict | None = None
    prediction_output: dict | None = None
    risk_explanation: dict | None = None
    recommended_action: dict | None = None
