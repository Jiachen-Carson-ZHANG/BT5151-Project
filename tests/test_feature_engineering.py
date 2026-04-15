import json
from pathlib import Path

import pandas as pd

from bt5151_credit_risk.feature_engineering import validate_feature_engineering_output


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
