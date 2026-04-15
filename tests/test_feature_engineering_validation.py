import json

import pandas as pd

from bt5151_credit_risk.feature_engineering import validate_feature_engineering_output


def _write_artifacts(tmp_path, train_df, test_df, view_metadata=None):
    (tmp_path / "feature_engineering_report.json").write_text(json.dumps({}))
    if view_metadata:
        (tmp_path / "view_metadata.json").write_text(json.dumps(view_metadata))
        for view in view_metadata["views"].values():
            train_df.to_csv(tmp_path / view["train_artifact"], index=False)
            test_df.to_csv(tmp_path / view["test_artifact"], index=False)
    else:
        train_df.to_csv(tmp_path / "engineered_train.csv", index=False)
        test_df.to_csv(tmp_path / "engineered_test.csv", index=False)

    return {
        "feature_engineering_report.json": str(tmp_path / "feature_engineering_report.json"),
        "view_metadata.json": str(tmp_path / "view_metadata.json"),
        "engineered_train.csv": str(tmp_path / "engineered_train.csv"),
        "engineered_test.csv": str(tmp_path / "engineered_test.csv"),
        "engineered_train_linear.csv": str(tmp_path / "engineered_train_linear.csv"),
        "engineered_test_linear.csv": str(tmp_path / "engineered_test_linear.csv"),
        "engineered_train_tree.csv": str(tmp_path / "engineered_train_tree.csv"),
        "engineered_test_tree.csv": str(tmp_path / "engineered_test_tree.csv"),
    }


def test_dual_view_fails_when_view_has_non_numeric_column(tmp_path):
    view_metadata = {
        "views": {
            "linear_view": {
                "train_artifact": "engineered_train_linear.csv",
                "test_artifact": "engineered_test_linear.csv",
            },
            "tree_view": {
                "train_artifact": "engineered_train_tree.csv",
                "test_artifact": "engineered_test_tree.csv",
            },
        }
    }
    # tree_view carries a string column — should fail the numeric check
    train_df = pd.DataFrame({"age": [25, 30, 40], "occupation": ["eng", "doc", "eng"]})
    test_df = pd.DataFrame({"age": [35, 45], "occupation": ["doc", "eng"]})
    artifacts = _write_artifacts(tmp_path, train_df, test_df, view_metadata)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": view_metadata},
        original_train_rows=3,
        original_test_rows=2,
        original_feature_count=2,
    )

    assert result["passed"] is False
    failing = {e["rule"] for e in result["errors"]}
    assert "linear_view_train_fully_numeric" in failing
    assert "tree_view_train_fully_numeric" in failing


def test_dual_view_passes_when_all_views_numeric(tmp_path):
    view_metadata = {
        "views": {
            "linear_view": {
                "train_artifact": "engineered_train_linear.csv",
                "test_artifact": "engineered_test_linear.csv",
            },
            "tree_view": {
                "train_artifact": "engineered_train_tree.csv",
                "test_artifact": "engineered_test_tree.csv",
            },
        }
    }
    train_df = pd.DataFrame({"age": [25, 30, 40], "occ_code": [1, 2, 1]})
    test_df = pd.DataFrame({"age": [35, 45], "occ_code": [2, 1]})
    artifacts = _write_artifacts(tmp_path, train_df, test_df, view_metadata)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": view_metadata},
        original_train_rows=3,
        original_test_rows=2,
        original_feature_count=2,
    )

    assert result["passed"] is True, result["errors"]
    assert result["checks"]["linear_view_train_fully_numeric"] is True
    assert result["checks"]["tree_view_train_fully_numeric"] is True


def test_single_view_rejects_non_numeric_column(tmp_path):
    train_df = pd.DataFrame({"age": [25, 30], "occupation": ["eng", "doc"]})
    test_df = pd.DataFrame({"age": [35, 45], "occupation": ["doc", "eng"]})
    artifacts = _write_artifacts(tmp_path, train_df, test_df, view_metadata=None)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=2,
    )

    assert result["passed"] is False
    failing = {e["rule"] for e in result["errors"]}
    assert "train_fully_numeric" in failing
    assert "test_fully_numeric" in failing
