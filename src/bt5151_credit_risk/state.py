from pydantic import BaseModel


class CreditRiskState(BaseModel):
    raw_dataset_path: str
    dataset_profile: dict | None = None
    preprocessing_rules: dict | None = None
    evaluation_results: dict | None = None
    prediction_output: dict | None = None
    risk_explanation: dict | None = None
    recommended_action: dict | None = None
