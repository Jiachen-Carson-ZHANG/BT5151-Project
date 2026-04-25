import pandas as pd

from bt5151_credit_risk.schema_contracts import validate_semantic_invariants
from bt5151_credit_risk.schema_contracts import validate_semantic_roles


def test_validate_semantic_roles_exposed_via_schema_contracts():
    feature_frame = pd.DataFrame({"Has_Defaulted": [1, 2, 0]})
    spec = {
        "transforms": {
            "Has_Defaulted": {
                "action": "keep",
                "semantic_role": "binary_flag",
            }
        }
    }

    violations = validate_semantic_roles(feature_frame, spec)

    assert any(v["violation"] == "not_binary" for v in violations)


def test_validate_semantic_invariants_exposed_via_schema_contracts():
    feature_frame = pd.DataFrame(
        {
            "Age": [25, 35],
            "Credit_History_Age": [500, 800],
        }
    )

    violations = validate_semantic_invariants(feature_frame, raw_frame=None, column_transform_spec={})

    assert any(v["violation"] == "credit_history_exceeds_adulthood_cap" for v in violations)
