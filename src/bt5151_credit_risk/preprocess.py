import ast
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd

from bt5151_credit_risk.config import GROUP_COLUMN, TARGET_COLUMN, TEST_SIZE
from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.skill_prompts import load_skill_prompt


def _call_preprocess_agent(system_prompt, payload):
    return call_json_response(system_prompt, payload)


def _call_preprocess_codegen_agent(system_prompt, payload):
    return call_json_response(system_prompt, payload)


def generate_dataset_policy_spec(df: pd.DataFrame, dataset_profile: dict) -> dict:
    system_prompt = load_skill_prompt("dataset-policy-spec")
    payload = {
        "columns": df.columns.tolist(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "dataset_profile": dataset_profile,
    }
    return _call_preprocess_agent(system_prompt, payload)


def generate_column_transform_spec(df: pd.DataFrame, dataset_policy_spec: dict) -> dict:
    system_prompt = load_skill_prompt("column-transform-spec")
    payload = {
        "columns": df.columns.tolist(),
        "sample_rows": df.head(5).to_dict(orient="records"),
        "dataset_policy_spec": dataset_policy_spec,
    }
    return _call_preprocess_agent(system_prompt, payload)


def generate_preprocessing_code(
    raw_df: pd.DataFrame,
    dataset_profile: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    system_prompt = load_skill_prompt("generate-preprocessing-code")
    payload = {
        "columns": raw_df.columns.tolist(),
        "sample_rows": raw_df.head(5).to_dict(orient="records"),
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
    }
    return _call_preprocess_codegen_agent(system_prompt, payload)


def repair_preprocessing_code(
    *,
    previous_generated_code: dict,
    code_review: dict,
    execution_log: dict,
    validation_report: dict,
    dataset_profile: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    system_prompt = load_skill_prompt("repair-preprocessing-code")
    payload = {
        "previous_generated_code": previous_generated_code,
        "code_review": code_review,
        "execution_log": execution_log,
        "validation_report": validation_report,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
    }
    return _call_preprocess_codegen_agent(system_prompt, payload)


def inspect_preprocessing_code(generated_code: dict) -> dict:
    code = generated_code.get("code", "")
    entrypoint = generated_code.get("entrypoint")
    issues: list[dict] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "passed": False,
            "entrypoint": entrypoint,
            "issues": [
                {
                    "rule": "syntax_error",
                    "message": str(exc),
                }
            ],
        }

    defined_entrypoints = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if entrypoint not in defined_entrypoints:
        issues.append(
            {
                "rule": "missing_entrypoint",
                "message": f"Declared entrypoint '{entrypoint}' was not found as a function definition.",
            }
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess" or alias.name.startswith("subprocess."):
                    issues.append(
                        {
                            "rule": "forbidden_import",
                            "message": "Importing subprocess is not allowed in generated preprocessing code.",
                        }
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess" or (node.module or "").startswith("subprocess."):
                issues.append(
                    {
                        "rule": "forbidden_import",
                        "message": "Importing subprocess is not allowed in generated preprocessing code.",
                    }
                )
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "system"
            ):
                issues.append(
                    {
                        "rule": "forbidden_call",
                        "message": "Calling os.system is not allowed in generated preprocessing code.",
                    }
                )

    return {
        "passed": not issues,
        "entrypoint": entrypoint,
        "issues": issues,
    }


def execute_generated_preprocessing(
    raw_df: pd.DataFrame,
    generated_code: dict,
    run_root,
) -> dict:
    entrypoint_name = generated_code.get("entrypoint", "run_preprocessing")
    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)

    workspace_path = run_root_path / f"generated_preprocessing_{uuid4().hex}"
    workspace_path.mkdir(parents=True, exist_ok=False)

    raw_frame_path = workspace_path / "raw_frame.csv"
    code_path = workspace_path / "generated_preprocessing.py"
    runner_path = workspace_path / "_execute_generated_preprocessing.py"

    raw_df.to_csv(raw_frame_path, index=False)
    code_path.write_text(generated_code.get("code", ""), encoding="utf-8")
    runner_path.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import json",
                "import sys",
                "from pathlib import Path",
                "",
                "import pandas as pd",
                "",
                "",
                "def main() -> None:",
                "    code_path = Path(sys.argv[1])",
                "    raw_frame_path = Path(sys.argv[2])",
                "    workspace_path = Path(sys.argv[3])",
                f"    entrypoint_name = {entrypoint_name!r}",
                "    spec = importlib.util.spec_from_file_location(\"generated_preprocessing\", code_path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    entrypoint = getattr(module, entrypoint_name, None)",
                "    if not callable(entrypoint):",
                "        raise AttributeError(f\"Generated preprocessing entrypoint '{entrypoint_name}' was not found or is not callable.\")",
                "    raw_df = pd.read_csv(raw_frame_path)",
                "    result = entrypoint(raw_df, workspace_path)",
                "    if result is not None:",
                "        print(json.dumps(result, default=str))",
                "",
                "",
                "if __name__ == \"__main__\":",
                "    main()",
                "",
            ]
        ),
        encoding="utf-8",
    )

    timeout_seconds = 60
    try:
        completed = subprocess.run(
            [sys.executable, str(runner_path), str(code_path), str(raw_frame_path), str(workspace_path)],
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
        execution_log = {
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }

    required_artifacts = [
        "cleaned_frame.csv",
        "feature_frame.csv",
        "target.csv",
        "split_manifest.json",
        "preprocessing_report.json",
    ]
    artifacts = {
        artifact_name: str(workspace_path / artifact_name) for artifact_name in required_artifacts
    }
    missing_artifacts = [
        artifact_name for artifact_name, artifact_path in artifacts.items() if not Path(artifact_path).is_file()
    ]

    return {
        "workspace_path": str(workspace_path),
        "run_root": str(run_root_path),
        "raw_frame_path": str(raw_frame_path),
        "code_path": str(code_path),
        "runner_path": str(runner_path),
        "entrypoint": entrypoint_name,
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "execution_log": execution_log,
        "success": execution_log["timed_out"] is False
        and execution_log["returncode"] == 0
        and not missing_artifacts,
    }


def validate_preprocessing_output(
    execution_result: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    artifacts = execution_result.get("artifacts", {})
    workspace_path = Path(execution_result.get("workspace_path", ""))
    raw_frame_path = Path(execution_result.get("raw_frame_path", workspace_path / "raw_frame.csv"))
    feature_frame_path = Path(artifacts.get("feature_frame.csv", ""))
    target_path = Path(artifacts.get("target.csv", ""))
    split_manifest_path = Path(artifacts.get("split_manifest.json", ""))
    target_column = dataset_policy_spec.get("target_column", TARGET_COLUMN)
    group_column = dataset_policy_spec.get("group_column", GROUP_COLUMN)
    split_strategy = dataset_policy_spec.get("split_strategy", {})

    checks: dict[str, bool] = {}
    errors: list[dict] = []

    def add_check(rule: str, passed: bool, message: str) -> None:
        checks[rule] = passed
        if not passed:
            errors.append({"rule": rule, "message": message})

    add_check(
        "split_manifest_exists",
        split_manifest_path.is_file(),
        f"Split manifest was not found at {split_manifest_path}.",
    )
    add_check(
        "target_file_exists",
        target_path.is_file(),
        f"Target file was not found at {target_path}.",
    )

    feature_frame = None
    if feature_frame_path.is_file():
        try:
            feature_frame = pd.read_csv(feature_frame_path)
        except Exception as exc:  # pragma: no cover - defensive parsing guard
            errors.append(
                {
                    "rule": "feature_frame_readable",
                    "message": f"Feature frame could not be read from {feature_frame_path}: {exc}",
                }
            )
            checks["feature_frame_readable"] = False
        else:
            checks["feature_frame_readable"] = True
            add_check(
                "target_excluded",
                target_column not in feature_frame.columns,
                f"Target column '{target_column}' is still present in the feature frame.",
            )
            add_check(
                "feature_frame_non_empty",
                not feature_frame.empty,
                f"Feature frame at {feature_frame_path} is empty.",
            )
    else:
        checks["feature_frame_readable"] = False
        checks["target_excluded"] = False
        checks["feature_frame_non_empty"] = False
        errors.append(
            {
                "rule": "feature_frame_exists",
                "message": f"Feature frame was not found at {feature_frame_path}.",
            }
        )
        errors.append(
            {
                "rule": "target_excluded",
                "message": f"Feature frame is unavailable, so target exclusion could not be verified.",
            }
        )
        errors.append(
            {
                "rule": "feature_frame_non_empty",
                "message": f"Feature frame is unavailable, so non-empty validation could not be performed.",
            }
        )

    if split_strategy.get("type") == "grouped_holdout" and group_column:
        group_overlap_zero = False
        if split_manifest_path.is_file() and raw_frame_path.is_file():
            try:
                manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
                train_indices = manifest["train_indices"]
                test_indices = manifest["test_indices"]
                raw_frame = pd.read_csv(raw_frame_path)
                train_groups = set(raw_frame.iloc[train_indices][group_column].dropna().tolist())
                test_groups = set(raw_frame.iloc[test_indices][group_column].dropna().tolist())
                group_overlap_zero = train_groups.isdisjoint(test_groups)
            except Exception as exc:  # pragma: no cover - defensive parsing guard
                errors.append(
                    {
                        "rule": "group_overlap_zero",
                        "message": (
                            "Grouped split overlap could not be validated "
                            f"using {split_manifest_path} and {raw_frame_path}: {exc}"
                        ),
                    }
                )
        else:
            errors.append(
                {
                    "rule": "group_overlap_zero",
                    "message": (
                        "Grouped split overlap could not be validated because the raw frame "
                        "or split manifest is missing."
                    ),
                }
            )
        checks["group_overlap_zero"] = group_overlap_zero
        if not group_overlap_zero and not any(error["rule"] == "group_overlap_zero" for error in errors):
            errors.append(
                {
                    "rule": "group_overlap_zero",
                    "message": "Grouped split contains overlapping group values between train and test.",
                }
            )
    else:
        checks["group_overlap_zero"] = True

    return {
        "passed": all(checks.values()) and not errors,
        "checks": checks,
        "errors": errors,
    }
