import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt


def _build_fe_artifact_paths(workspace_path: Path) -> dict[str, str]:
    artifact_names = [
        "engineered_train.csv",
        "engineered_test.csv",
        "engineered_train_linear.csv",
        "engineered_test_linear.csv",
        "engineered_train_tree.csv",
        "engineered_test_tree.csv",
        "feature_engineering_report.json",
        "view_metadata.json",
    ]
    return {name: str(workspace_path / name) for name in artifact_names}


def _load_view_metadata(artifacts: dict) -> dict | None:
    metadata_path = Path(artifacts.get("view_metadata.json", ""))
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata.get("views"), dict) and metadata["views"]:
            return metadata
    except Exception:
        return None
    return None


def _call_fe_codegen_agent(system_prompt, payload, caller="feature-engineering-codegen"):
    return call_json_response(system_prompt, payload, caller=caller)


def _build_feature_stats(df: pd.DataFrame) -> list[dict]:
    """Per-column statistics for the LLM to reason about transforms."""
    stats = []
    for col in df.columns:
        entry = {
            "column": col,
            "dtype": str(df[col].dtype),
            "nunique": int(df[col].nunique()),
            "null_count": int(df[col].isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["min"] = float(df[col].min()) if not df[col].isna().all() else None
            entry["max"] = float(df[col].max()) if not df[col].isna().all() else None
            entry["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
            entry["std"] = float(df[col].std()) if not df[col].isna().all() else None
            entry["skewness"] = float(df[col].skew()) if not df[col].isna().all() else None
        stats.append(entry)
    return stats


# Ask the LLM to generate feature engineering code.
def generate_feature_engineering_code(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    dataset_profile: dict,
    eda_report: dict | None = None,
    eda_hypotheses: dict | None = None,
) -> dict:
    system_prompt = load_skill_prompt("generate-feature-engineering-code")
    payload = {
        "feature_columns": feature_columns,
        "train_sample": train_frame.head(5).to_dict(orient="records"),
        "train_stats": _build_feature_stats(train_frame),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "dataset_profile": dataset_profile,
    }
    if eda_report:
        payload["eda_insights"] = {
            "top_discriminative_features": eda_report.get("top_discriminative_features", [])[:10],
            "high_correlation_pairs": eda_report.get("correlations", {}).get("high_pairs", [])[:10],
            "highly_skewed_columns": eda_report.get("skewness", {}).get("highly_skewed", {}),
        }
    if eda_hypotheses:
        payload["eda_hypotheses"] = {
            "tested_predictions": eda_hypotheses.get("tested_predictions", []),
            "exploratory_leads": eda_hypotheses.get("exploratory_leads", []),
        }
    return _call_fe_codegen_agent(system_prompt, payload, caller="generate-feature-engineering-code")


# Ask the LLM to repair failed feature engineering code.
def repair_feature_engineering_code(
    *,
    previous_generated_code: dict,
    code_review: dict | None,
    execution_log: dict | None,
    validation_report: dict | None,
    feature_columns: list[str],
    dataset_profile: dict,
) -> dict:
    system_prompt = load_skill_prompt("repair-feature-engineering-code")
    payload = {
        "previous_generated_code": previous_generated_code,
        "code_review": code_review,
        "execution_log": execution_log,
        "validation_report": validation_report,
        "feature_columns": feature_columns,
        "dataset_profile": dataset_profile,
    }
    return _call_fe_codegen_agent(system_prompt, payload, caller="repair-feature-engineering-code")


# Run feature engineering code in a subprocess, writing output CSVs.
def execute_feature_engineering(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    generated_code: dict,
    run_root,
) -> dict:
    entrypoint_name = generated_code.get("entrypoint", "engineer_features")
    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)

    workspace_path = run_root_path / f"feature_engineering_{uuid4().hex}"
    workspace_path.mkdir(parents=True, exist_ok=False)

    train_path = workspace_path / "input_train.csv"
    test_path = workspace_path / "input_test.csv"
    code_path = workspace_path / "generated_feature_engineering.py"
    runner_path = workspace_path / "_execute_feature_engineering.py"

    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    code_path.write_text(generated_code.get("code", ""), encoding="utf-8")

    runner_path.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import sys",
                "from pathlib import Path",
                "",
                "import pandas as pd",
                "",
                "",
                "def main() -> None:",
                "    code_path = Path(sys.argv[1])",
                "    train_path = Path(sys.argv[2])",
                "    test_path = Path(sys.argv[3])",
                "    workspace_path = Path(sys.argv[4])",
                f"    entrypoint_name = {entrypoint_name!r}",
                '    spec = importlib.util.spec_from_file_location("generated_feature_engineering", code_path)',
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    entrypoint = getattr(module, entrypoint_name, None)",
                "    if not callable(entrypoint):",
                "        raise AttributeError(f\"Entrypoint '{entrypoint_name}' was not found or is not callable.\")",
                "    train_df = pd.read_csv(train_path)",
                "    test_df = pd.read_csv(test_path)",
                "    entrypoint(train_df, test_df, workspace_path)",
                "",
                "",
                'if __name__ == "__main__":',
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    timeout_seconds = 120
    try:
        completed = subprocess.run(
            [sys.executable, str(runner_path), str(code_path), str(train_path), str(test_path), str(workspace_path)],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        execution_log = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
        }
    except subprocess.TimeoutExpired as exc:
        raw_stdout = exc.stdout or ""
        raw_stderr = exc.stderr or ""
        execution_log = {
            "returncode": None,
            "stdout": raw_stdout.decode("utf-8", errors="replace") if isinstance(raw_stdout, bytes) else raw_stdout,
            "stderr": raw_stderr.decode("utf-8", errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }

    artifacts = _build_fe_artifact_paths(workspace_path)
    view_metadata = _load_view_metadata(artifacts)

    if view_metadata:
        required_artifacts = ["feature_engineering_report.json", "view_metadata.json"]
        for view_spec in view_metadata.get("views", {}).values():
            train_artifact = view_spec.get("train_artifact")
            test_artifact = view_spec.get("test_artifact")
            if train_artifact:
                required_artifacts.append(train_artifact)
            if test_artifact:
                required_artifacts.append(test_artifact)
    else:
        required_artifacts = [
            "engineered_train.csv",
            "engineered_test.csv",
            "feature_engineering_report.json",
        ]

    missing_artifacts = [
        artifact_name for artifact_name in required_artifacts
        if not Path(artifacts.get(artifact_name, workspace_path / artifact_name)).is_file()
    ]

    return {
        "workspace_path": str(workspace_path),
        "run_root": str(run_root_path),
        "code_path": str(code_path),
        "artifacts": artifacts,
        "view_metadata": view_metadata,
        "missing_artifacts": missing_artifacts,
        "execution_log": execution_log,
        "success": execution_log["timed_out"] is False
        and execution_log["returncode"] == 0
        and not missing_artifacts,
    }


# Validate feature engineering output: row counts, NaNs, feature count.
def validate_feature_engineering_output(
    execution_result: dict,
    original_train_rows: int,
    original_test_rows: int,
    original_feature_count: int,
) -> dict:
    artifacts = execution_result.get("artifacts", {})
    view_metadata = execution_result.get("view_metadata") or _load_view_metadata(artifacts)
    checks: dict[str, bool] = {}
    errors: list[dict] = []
    views: dict[str, dict] = {}

    def add_check(rule: str, passed: bool, message: str) -> None:
        checks[rule] = passed
        if not passed:
            errors.append({"rule": rule, "message": message})

    report_path = Path(artifacts.get("feature_engineering_report.json", ""))

    add_check("report_file_exists", report_path.is_file(), "feature_engineering_report.json not found.")

    if view_metadata and isinstance(view_metadata.get("views"), dict):
        add_check("view_metadata_exists", True, "")
        for view_name, view_spec in view_metadata["views"].items():
            train_artifact = view_spec.get("train_artifact")
            test_artifact = view_spec.get("test_artifact")
            train_path = Path(artifacts.get(train_artifact, ""))
            test_path = Path(artifacts.get(test_artifact, ""))

            add_check(f"{view_name}_train_file_exists", train_path.is_file(), f"{train_artifact} not found.")
            add_check(f"{view_name}_test_file_exists", test_path.is_file(), f"{test_artifact} not found.")

            train_df = None
            test_df = None

            if train_path.is_file():
                try:
                    train_df = pd.read_csv(train_path)
                except Exception as exc:
                    add_check(f"{view_name}_train_readable", False, f"Could not read {train_artifact}: {exc}")
                else:
                    checks[f"{view_name}_train_readable"] = True
                    add_check(
                        f"{view_name}_train_row_count_match",
                        len(train_df) == original_train_rows,
                        f"{train_artifact} row count changed: expected {original_train_rows}, got {len(train_df)}.",
                    )

            if test_path.is_file():
                try:
                    test_df = pd.read_csv(test_path)
                except Exception as exc:
                    add_check(f"{view_name}_test_readable", False, f"Could not read {test_artifact}: {exc}")
                else:
                    checks[f"{view_name}_test_readable"] = True
                    add_check(
                        f"{view_name}_test_row_count_match",
                        len(test_df) == original_test_rows,
                        f"{test_artifact} row count changed: expected {original_test_rows}, got {len(test_df)}.",
                    )

            if train_df is not None:
                train_nans = int(train_df.isna().sum().sum())
                add_check(f"{view_name}_no_nans_in_train", train_nans == 0, f"{train_artifact} has {train_nans} NaN values.")
                train_infs = int(np.isinf(train_df.select_dtypes(include="number")).sum().sum())
                add_check(f"{view_name}_no_infs_in_train", train_infs == 0, f"{train_artifact} has {train_infs} inf values.")
                train_non_numeric = list(train_df.select_dtypes(exclude="number").columns)
                add_check(
                    f"{view_name}_train_fully_numeric",
                    not train_non_numeric,
                    f"{train_artifact} has non-numeric columns {train_non_numeric[:10]} — FE must encode deferred categoricals per view.",
                )
                add_check(
                    f"{view_name}_features_non_empty",
                    len(train_df.columns) > 0,
                    f"{train_artifact} has zero columns.",
                )
                max_cols = original_feature_count * 5
                add_check(
                    f"{view_name}_max_feature_cap",
                    len(train_df.columns) <= max_cols,
                    f"{train_artifact} exploded to {len(train_df.columns)} cols (max allowed: {max_cols}).",
                )

            if test_df is not None:
                test_nans = int(test_df.isna().sum().sum())
                add_check(f"{view_name}_no_nans_in_test", test_nans == 0, f"{test_artifact} has {test_nans} NaN values.")
                test_infs = int(np.isinf(test_df.select_dtypes(include="number")).sum().sum())
                add_check(f"{view_name}_no_infs_in_test", test_infs == 0, f"{test_artifact} has {test_infs} inf values.")
                test_non_numeric = list(test_df.select_dtypes(exclude="number").columns)
                add_check(
                    f"{view_name}_test_fully_numeric",
                    not test_non_numeric,
                    f"{test_artifact} has non-numeric columns {test_non_numeric[:10]} — FE must encode deferred categoricals per view.",
                )

            if train_df is not None and test_df is not None:
                add_check(
                    f"{view_name}_column_alignment",
                    list(train_df.columns) == list(test_df.columns),
                    f"{view_name} train and test columns do not match after feature engineering.",
                )
                views[view_name] = {
                    "train_artifact": train_artifact,
                    "test_artifact": test_artifact,
                    "feature_count": len(train_df.columns),
                }
    else:
        train_path = Path(artifacts.get("engineered_train.csv", ""))
        test_path = Path(artifacts.get("engineered_test.csv", ""))

        add_check("train_file_exists", train_path.is_file(), "engineered_train.csv not found.")
        add_check("test_file_exists", test_path.is_file(), "engineered_test.csv not found.")

        train_df = None
        test_df = None

        if train_path.is_file():
            try:
                train_df = pd.read_csv(train_path)
            except Exception as exc:
                add_check("train_readable", False, f"Could not read engineered_train.csv: {exc}")
            else:
                checks["train_readable"] = True
                add_check(
                    "train_row_count_match",
                    len(train_df) == original_train_rows,
                    f"Train row count changed: expected {original_train_rows}, got {len(train_df)}.",
                )

        if test_path.is_file():
            try:
                test_df = pd.read_csv(test_path)
            except Exception as exc:
                add_check("test_readable", False, f"Could not read engineered_test.csv: {exc}")
            else:
                checks["test_readable"] = True
                add_check(
                    "test_row_count_match",
                    len(test_df) == original_test_rows,
                    f"Test row count changed: expected {original_test_rows}, got {len(test_df)}.",
                )

        if train_df is not None:
            train_nans = int(train_df.isna().sum().sum())
            add_check("no_nans_in_train", train_nans == 0, f"engineered_train.csv has {train_nans} NaN values.")
            train_infs = int(np.isinf(train_df.select_dtypes(include="number")).sum().sum())
            add_check("no_infs_in_train", train_infs == 0, f"engineered_train.csv has {train_infs} inf values.")
            train_non_numeric = list(train_df.select_dtypes(exclude="number").columns)
            add_check(
                "train_fully_numeric",
                not train_non_numeric,
                f"engineered_train.csv has non-numeric columns {train_non_numeric[:10]} — FE must encode deferred categoricals.",
            )
            add_check(
                "features_non_empty",
                len(train_df.columns) > 0,
                "Engineered train frame has zero columns.",
            )
            max_cols = original_feature_count * 5
            add_check(
                "max_feature_cap",
                len(train_df.columns) <= max_cols,
                f"Feature count exploded: {len(train_df.columns)} cols (max allowed: {max_cols}).",
            )

        if test_df is not None:
            test_nans = int(test_df.isna().sum().sum())
            add_check("no_nans_in_test", test_nans == 0, f"engineered_test.csv has {test_nans} NaN values.")
            test_infs = int(np.isinf(test_df.select_dtypes(include="number")).sum().sum())
            add_check("no_infs_in_test", test_infs == 0, f"engineered_test.csv has {test_infs} inf values.")
            test_non_numeric = list(test_df.select_dtypes(exclude="number").columns)
            add_check(
                "test_fully_numeric",
                not test_non_numeric,
                f"engineered_test.csv has non-numeric columns {test_non_numeric[:10]} — FE must encode deferred categoricals.",
            )

        if train_df is not None and test_df is not None:
            add_check(
                "column_alignment",
                list(train_df.columns) == list(test_df.columns),
                "Train and test columns do not match after feature engineering.",
            )
            views["default"] = {
                "train_artifact": "engineered_train.csv",
                "test_artifact": "engineered_test.csv",
                "feature_count": len(train_df.columns),
            }

    return {
        "passed": all(checks.values()) and not errors,
        "checks": checks,
        "errors": errors,
        "view_mode": "dual" if view_metadata else "single",
        "available_views": list(views.keys()),
        "views": views,
    }
