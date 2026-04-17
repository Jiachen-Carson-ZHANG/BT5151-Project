import json

import pandas as pd

from bt5151_credit_risk.feature_engineering import validate_feature_engineering_output


def _write_artifacts(tmp_path, train_df, test_df, view_metadata=None, lineage=None, fe_report=None):
    (tmp_path / "feature_engineering_report.json").write_text(json.dumps(fe_report or {}))
    if view_metadata:
        (tmp_path / "view_metadata.json").write_text(json.dumps(view_metadata))
        for view in view_metadata["views"].values():
            train_df.to_csv(tmp_path / view["train_artifact"], index=False)
            test_df.to_csv(tmp_path / view["test_artifact"], index=False)
    else:
        train_df.to_csv(tmp_path / "engineered_train.csv", index=False)
        test_df.to_csv(tmp_path / "engineered_test.csv", index=False)

    # Default lineage: declare every output column as passthrough (satisfies coverage).
    if lineage is None:
        lineage = {
            "derived_features": [],
            "dropped_features": [],
            "passthrough_features": list(train_df.columns),
        }
    (tmp_path / "feature_lineage.json").write_text(json.dumps(lineage))

    return {
        "feature_engineering_report.json": str(tmp_path / "feature_engineering_report.json"),
        "view_metadata.json": str(tmp_path / "view_metadata.json"),
        "engineered_train.csv": str(tmp_path / "engineered_train.csv"),
        "engineered_test.csv": str(tmp_path / "engineered_test.csv"),
        "engineered_train_linear.csv": str(tmp_path / "engineered_train_linear.csv"),
        "engineered_test_linear.csv": str(tmp_path / "engineered_test_linear.csv"),
        "engineered_train_tree.csv": str(tmp_path / "engineered_train_tree.csv"),
        "engineered_test_tree.csv": str(tmp_path / "engineered_test_tree.csv"),
        "feature_lineage.json": str(tmp_path / "feature_lineage.json"),
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


def test_dual_view_accepts_boolean_dummy_columns(tmp_path):
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
    train_df = pd.DataFrame({
        "age": [25, 30, 40],
        "occupation_Doctor": [True, False, False],
        "occupation_Engineer": [False, True, True],
    })
    test_df = pd.DataFrame({
        "age": [35, 45],
        "occupation_Doctor": [False, True],
        "occupation_Engineer": [True, False],
    })
    artifacts = _write_artifacts(tmp_path, train_df, test_df, view_metadata)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": view_metadata},
        original_train_rows=3,
        original_test_rows=2,
        original_feature_count=3,
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


def test_single_view_accepts_boolean_dummy_columns(tmp_path):
    train_df = pd.DataFrame({
        "age": [25, 30],
        "occupation_Doctor": [True, False],
        "occupation_Engineer": [False, True],
    })
    test_df = pd.DataFrame({
        "age": [35, 45],
        "occupation_Doctor": [False, True],
        "occupation_Engineer": [True, False],
    })
    artifacts = _write_artifacts(tmp_path, train_df, test_df, view_metadata=None)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=3,
    )

    assert result["passed"] is True, result["errors"]
    assert result["checks"]["train_fully_numeric"] is True
    assert result["checks"]["test_fully_numeric"] is True


# --- Phase 2: feature lineage manifest tests ---


def test_lineage_missing_artifact_fails(tmp_path):
    """FE output without feature_lineage.json must fail the contract."""
    train_df = pd.DataFrame({"age": [25, 30], "income": [100.0, 200.0]})
    test_df = pd.DataFrame({"age": [35, 45], "income": [150.0, 250.0]})
    # Write artifacts WITHOUT lineage
    (tmp_path / "feature_engineering_report.json").write_text(json.dumps({}))
    train_df.to_csv(tmp_path / "engineered_train.csv", index=False)
    test_df.to_csv(tmp_path / "engineered_test.csv", index=False)
    artifacts = {
        "feature_engineering_report.json": str(tmp_path / "feature_engineering_report.json"),
        "engineered_train.csv": str(tmp_path / "engineered_train.csv"),
        "engineered_test.csv": str(tmp_path / "engineered_test.csv"),
    }

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=2,
    )

    assert result["passed"] is False
    rules = {v["rule"] for v in result["lineage_violations"]}
    assert "lineage_artifact_present" in rules


def test_lineage_replay_catches_log_transformed_parents(tmp_path):
    """Lineage claims ratio uses raw parents, but code actually divided log1p(a)/b.

    Replay should recompute raw A/B, compare against the actual column, and fail."""
    pre_fe_train = pd.DataFrame({
        "A": [10.0, 20.0, 40.0, 80.0, 100.0] * 4,
        "B": [2.0, 4.0, 5.0, 10.0, 10.0] * 4,
    })
    # Code actually computed log1p(A)/B, not A/B as lineage claims.
    import numpy as np
    engineered_train = pre_fe_train.copy()
    engineered_train["AB_ratio"] = np.log1p(pre_fe_train["A"]) / pre_fe_train["B"]
    engineered_test = engineered_train.iloc[:4].reset_index(drop=True)

    lineage = {
        "derived_features": [
            {
                "feature": "AB_ratio",
                "operation": "ratio",
                "input_stage": "pre_fe_raw_numeric",
                "inputs": ["A", "B"],
            }
        ],
        "passthrough_features": ["A", "B"],
        "dropped_features": [],
    }
    artifacts = _write_artifacts(tmp_path, engineered_train, engineered_test, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=len(pre_fe_train),
        original_test_rows=len(engineered_test),
        original_feature_count=2,
        train_frame_pre_fe=pre_fe_train,
    )

    assert result["passed"] is False
    rules = {v["rule"] for v in result["lineage_violations"]}
    assert "lineage_replay_matches" in rules


def test_lineage_ratio_with_encoded_inputs_fails(tmp_path):
    """Ratios/products must declare input_stage=pre_fe_raw_numeric. Using
    pre_fe_encoded is a semantic error (ratio of log(A) to B is not A/B)."""
    train_df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "ratio_AB": [0.33, 0.5]})
    test_df = train_df.copy()
    lineage = {
        "derived_features": [
            {
                "feature": "ratio_AB",
                "operation": "ratio",
                "input_stage": "pre_fe_encoded",  # ← violates raw-parent rule
                "inputs": ["A", "B"],
            }
        ],
        "passthrough_features": ["A", "B"],
        "dropped_features": [],
    }
    artifacts = _write_artifacts(tmp_path, train_df, test_df, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=3,
    )

    assert result["passed"] is False
    rules = {v["rule"] for v in result["lineage_violations"]}
    assert "ratios_use_raw_parents" in rules


def test_top_mi_drop_without_justification_fails(tmp_path):
    """Top-MI feature dropped for correlation_with_higher_mi_feature fails the
    top_mi_drop_requires_justification rule — only leakage or deterministic_duplicate allowed."""
    train_df = pd.DataFrame({"B": [3.0, 4.0], "ratio_AB": [0.33, 0.5]})
    test_df = train_df.copy()
    lineage = {
        "derived_features": [],
        "dropped_features": [
            {"feature": "A", "drop_reason": "correlation_with_higher_mi_feature"}
        ],
        "passthrough_features": ["B", "ratio_AB"],
    }
    artifacts = _write_artifacts(tmp_path, train_df, test_df, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=3,
        top_mi_features=["A", "income"],
    )

    assert result["passed"] is False
    rules = {v["rule"] for v in result["lineage_violations"]}
    assert "top_mi_drop_requires_justification" in rules


def test_lineage_coverage_complete_flags_unaccounted_columns(tmp_path):
    """An engineered column with no lineage entry and no passthrough declaration
    must trigger lineage_coverage_complete."""
    train_df = pd.DataFrame({"A": [1.0, 2.0], "mystery_feature": [9.0, 10.0]})
    test_df = train_df.copy()
    lineage = {
        "derived_features": [],
        "dropped_features": [],
        "passthrough_features": ["A"],  # mystery_feature unaccounted
    }
    artifacts = _write_artifacts(tmp_path, train_df, test_df, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=2,
        original_feature_count=2,
    )

    assert result["passed"] is False
    rules = {v["rule"] for v in result["lineage_violations"]}
    assert "lineage_coverage_complete" in rules


def test_lineage_replay_accepts_correct_ratio(tmp_path):
    """Positive case: lineage declares A/B, code implements A/B — replay passes."""
    pre_fe_train = pd.DataFrame({
        "A": [10.0, 20.0, 40.0, 80.0, 100.0] * 4,
        "B": [2.0, 4.0, 5.0, 10.0, 10.0] * 4,
    })
    engineered_train = pre_fe_train.copy()
    engineered_train["AB_ratio"] = pre_fe_train["A"] / pre_fe_train["B"]
    engineered_test = engineered_train.iloc[:4].reset_index(drop=True)

    lineage = {
        "derived_features": [
            {
                "feature": "AB_ratio",
                "operation": "ratio",
                "input_stage": "pre_fe_raw_numeric",
                "inputs": ["A", "B"],
            }
        ],
        "passthrough_features": ["A", "B"],
        "dropped_features": [],
    }
    artifacts = _write_artifacts(tmp_path, engineered_train, engineered_test, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=len(pre_fe_train),
        original_test_rows=len(engineered_test),
        original_feature_count=2,
        train_frame_pre_fe=pre_fe_train,
    )

    assert result["passed"] is True, result.get("lineage_violations", []) + result.get("errors", [])


def test_lineage_replay_uses_report_formula_for_composed_features(tmp_path):
    """Lineage operation names are coarse; the report formula is the exact contract.

    Rich ratios like (salary - emi) / salary and debt * rate / 100 should replay
    against their declared formula instead of the old a/b or a*b fallback.
    """
    pre_fe_train = pd.DataFrame({
        "Monthly_Inhand_Salary": [1000.0, 2500.0, 4000.0, 6000.0, 8000.0] * 4,
        "Total_EMI_per_month": [100.0, 500.0, 1000.0, 1200.0, 1600.0] * 4,
        "Num_Credit_Inquiries": [1.0, 3.0, 5.0, 2.0, 8.0] * 4,
        "Num_Credit_Card": [0.0, 2.0, 4.0, 1.0, 3.0] * 4,
        "Outstanding_Debt": [500.0, 1000.0, 2500.0, 3000.0, 4500.0] * 4,
        "Interest_Rate": [5.0, 10.0, 12.5, 18.0, 25.0] * 4,
    })
    engineered_train = pre_fe_train.copy()
    engineered_train["Savings_Rate"] = (
        pre_fe_train["Monthly_Inhand_Salary"] - pre_fe_train["Total_EMI_per_month"]
    ) / pre_fe_train["Monthly_Inhand_Salary"]
    engineered_train["Inquiries_per_Credit_Card"] = (
        pre_fe_train["Num_Credit_Inquiries"] / (pre_fe_train["Num_Credit_Card"] + 1)
    )
    engineered_train["Interest_Burden"] = (
        pre_fe_train["Outstanding_Debt"] * pre_fe_train["Interest_Rate"] / 100
    )
    engineered_test = engineered_train.iloc[:4].reset_index(drop=True)

    lineage = {
        "derived_features": [
            {
                "feature": "Savings_Rate",
                "operation": "ratio",
                "input_stage": "pre_fe_raw_numeric",
                "inputs": ["Monthly_Inhand_Salary", "Total_EMI_per_month"],
            },
            {
                "feature": "Inquiries_per_Credit_Card",
                "operation": "ratio",
                "input_stage": "pre_fe_raw_numeric",
                "inputs": ["Num_Credit_Inquiries", "Num_Credit_Card"],
            },
            {
                "feature": "Interest_Burden",
                "operation": "product",
                "input_stage": "pre_fe_raw_numeric",
                "inputs": ["Outstanding_Debt", "Interest_Rate"],
            },
        ],
        "passthrough_features": list(pre_fe_train.columns),
        "dropped_features": [],
    }
    fe_report = {
        "added": [
            {
                "column": "Savings_Rate",
                "formula": "(Monthly_Inhand_Salary - Total_EMI_per_month) / Monthly_Inhand_Salary",
            },
            {
                "column": "Inquiries_per_Credit_Card",
                "formula": "Num_Credit_Inquiries / (Num_Credit_Card + 1)",
            },
            {
                "column": "Interest_Burden",
                "formula": "Outstanding_Debt * Interest_Rate / 100",
            },
        ]
    }
    artifacts = _write_artifacts(
        tmp_path,
        engineered_train,
        engineered_test,
        lineage=lineage,
        fe_report=fe_report,
    )

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=len(pre_fe_train),
        original_test_rows=len(engineered_test),
        original_feature_count=len(pre_fe_train.columns),
        train_frame_pre_fe=pre_fe_train,
    )

    assert result["passed"] is True, result.get("lineage_violations", []) + result.get("errors", [])


def test_lineage_accepts_generated_categorical_input_stage_alias(tmp_path):
    """FE repairs sometimes declare one-hot parents as pre_fe_raw_categorical.

    That spelling is semantically valid for deferred categoricals and should not
    block an otherwise numeric one-hot output.
    """
    train_df = pd.DataFrame({"Age": [30.0, 40.0], "Occupation_Doctor": [1, 0]})
    test_df = pd.DataFrame({"Age": [35.0], "Occupation_Doctor": [0]})
    lineage = {
        "derived_features": [
            {
                "feature": "Occupation",
                "operation": "one_hot",
                "input_stage": "pre_fe_raw_categorical",
                "inputs": ["Occupation"],
            }
        ],
        "passthrough_features": ["Age"],
        "dropped_features": [],
    }
    artifacts = _write_artifacts(tmp_path, train_df, test_df, lineage=lineage)

    result = validate_feature_engineering_output(
        {"artifacts": artifacts, "view_metadata": None},
        original_train_rows=2,
        original_test_rows=1,
        original_feature_count=2,
    )

    assert result["passed"] is True, result.get("lineage_violations", []) + result.get("errors", [])
