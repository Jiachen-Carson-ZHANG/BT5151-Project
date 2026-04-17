import json
from pathlib import Path

import pandas as pd

from bt5151_credit_risk.feature_engineering import (
    deterministic_feature_engineering_fallback_code,
    execute_feature_engineering,
    validate_feature_engineering_output,
)


def test_validate_feature_engineering_output_accepts_dual_view_artifacts(tmp_path):
    train_linear = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    test_linear = pd.DataFrame({"a": [3.0], "b": [1.0]})
    train_tree = pd.DataFrame({"x": [10.0, 20.0]})
    test_tree = pd.DataFrame({"x": [30.0]})

    (tmp_path / "engineered_train_linear.csv").write_text(train_linear.to_csv(index=False), encoding="utf-8")
    (tmp_path / "engineered_test_linear.csv").write_text(test_linear.to_csv(index=False), encoding="utf-8")
    (tmp_path / "engineered_train_tree.csv").write_text(train_tree.to_csv(index=False), encoding="utf-8")
    (tmp_path / "engineered_test_tree.csv").write_text(test_tree.to_csv(index=False), encoding="utf-8")
    (tmp_path / "feature_engineering_report.json").write_text(json.dumps({"added": [], "dropped": [], "transformed": []}), encoding="utf-8")
    (tmp_path / "feature_lineage.json").write_text(
        json.dumps(
            {
                "derived_features": [],
                "dropped_features": [],
                "passthrough_features": ["a", "b", "x"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "view_metadata.json").write_text(
        json.dumps(
            {
                "views": {
                    "linear_view": {"train_artifact": "engineered_train_linear.csv", "test_artifact": "engineered_test_linear.csv"},
                    "tree_view": {"train_artifact": "engineered_train_tree.csv", "test_artifact": "engineered_test_tree.csv"},
                }
            }
        ),
        encoding="utf-8",
    )

    execution_result = {
        "artifacts": {
            "engineered_train_linear.csv": str(tmp_path / "engineered_train_linear.csv"),
            "engineered_test_linear.csv": str(tmp_path / "engineered_test_linear.csv"),
            "engineered_train_tree.csv": str(tmp_path / "engineered_train_tree.csv"),
            "engineered_test_tree.csv": str(tmp_path / "engineered_test_tree.csv"),
            "feature_engineering_report.json": str(tmp_path / "feature_engineering_report.json"),
            "view_metadata.json": str(tmp_path / "view_metadata.json"),
            "feature_lineage.json": str(tmp_path / "feature_lineage.json"),
        }
    }

    report = validate_feature_engineering_output(
        execution_result,
        original_train_rows=2,
        original_test_rows=1,
        original_feature_count=2,
    )

    assert report["passed"] is True
    assert report["view_mode"] == "dual"
    assert set(report["available_views"]) == {"linear_view", "tree_view"}


def test_execute_feature_engineering_exposes_deferred_categoricals_to_generated_code(tmp_path):
    """Generated FE code must be able to access deferred_categorical_columns.

    The repair prompt receives this dict, but the subprocess used to hide it from
    the generated module. That made generated one-hot blocks silently skip
    Occupation and crash on non-numeric columns.
    """
    train = pd.DataFrame({"Age": [30, 40, 50], "Occupation": ["Engineer", "Doctor", "Engineer"]})
    test = pd.DataFrame({"Age": [35], "Occupation": ["Doctor"]})
    generated = {
        "entrypoint": "engineer_features",
        "code": """
import json
import pandas as pd

def engineer_features(train_df, test_df, workspace_path):
    deferred = deferred_categorical_columns
    assert deferred == {'Occupation': 2}
    dummies = pd.get_dummies(train_df['Occupation'], prefix='Occupation', drop_first=True)
    test_dummies = pd.get_dummies(test_df['Occupation'], prefix='Occupation', drop_first=True).reindex(columns=dummies.columns, fill_value=0)
    train_df = pd.concat([train_df.drop(columns=['Occupation']), dummies], axis=1)
    test_df = pd.concat([test_df.drop(columns=['Occupation']), test_dummies], axis=1)
    train_df.to_csv(workspace_path / 'engineered_train.csv', index=False)
    test_df.to_csv(workspace_path / 'engineered_test.csv', index=False)
    (workspace_path / 'feature_engineering_report.json').write_text(json.dumps({'added': [], 'dropped': [], 'transformed': []}))
    (workspace_path / 'feature_lineage.json').write_text(json.dumps({
        'derived_features': [{'feature': 'Occupation', 'operation': 'one_hot', 'inputs': ['Occupation'], 'input_stage': 'pre_fe_raw_categorical'}],
        'dropped_features': [],
        'passthrough_features': ['Age']
    }))
""",
    }

    result = execute_feature_engineering(
        train,
        test,
        generated,
        tmp_path,
        deferred_categorical_columns={"Occupation": 2},
    )

    assert result["success"] is True, result["execution_log"]
    engineered = pd.read_csv(result["artifacts"]["engineered_train.csv"])
    assert "Occupation_Engineer" in engineered.columns


def test_deterministic_feature_engineering_fallback_writes_valid_artifacts(tmp_path):
    train = pd.DataFrame({
        "Age": [30, 40, 50],
        "Occupation": ["Engineer", "Doctor", "Engineer"],
        "Outstanding_Debt": [100.0, None, 300.0],
    })
    test = pd.DataFrame({
        "Age": [35],
        "Occupation": ["Doctor"],
        "Outstanding_Debt": [float("inf")],
    })

    execution = execute_feature_engineering(
        train,
        test,
        deterministic_feature_engineering_fallback_code(),
        tmp_path,
        deferred_categorical_columns={"Occupation": 2},
    )
    report = validate_feature_engineering_output(
        execution,
        original_train_rows=3,
        original_test_rows=1,
        original_feature_count=3,
        deferred_categoricals={"Occupation": 2},
        train_frame_pre_fe=train,
    )

    assert execution["success"] is True, execution["execution_log"]
    assert report["passed"] is True, report.get("errors", []) + report.get("lineage_violations", [])
