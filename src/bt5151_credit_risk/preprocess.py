import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from bt5151_credit_risk.llm import call_json_response
from bt5151_credit_risk.schema_contracts import validate_semantic_invariants as _validate_semantic_invariants_impl
from bt5151_credit_risk.schema_contracts import validate_semantic_roles as _validate_semantic_roles_impl
from bt5151_credit_risk.semantic_cleaning import allowed_cleaning_primitives
from bt5151_credit_risk.semantic_cleaning import cap_credit_history_by_adulthood as _cap_credit_history_by_adulthood
from bt5151_credit_risk.semantic_cleaning import coerce_numeric as _coerce_numeric
from bt5151_credit_risk.semantic_cleaning import fill_numeric_by_group_then_global as _fill_by_group_then_global
from bt5151_credit_risk.semantic_cleaning import missing_string_mask as _normalized_missing_string_mask
from bt5151_credit_risk.semantic_cleaning import parse_age_series as _parse_age_series
from bt5151_credit_risk.semantic_cleaning import parse_duration_months as _parse_credit_history_age_series
from bt5151_credit_risk.skill_prompts import load_skill_prompt


def _with_codegen_audit(result: dict, *, caller: str, prompt_payload: dict) -> dict:
    enriched = dict(result)
    enriched["_codegen_audit"] = {
        "caller": caller,
        "prompt_payload": prompt_payload,
    }
    return enriched


def _public_codegen_payload(value: dict | None) -> dict:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if not str(k).startswith("_")}


def _call_preprocess_agent(system_prompt, payload, caller="preprocess"):
    return call_json_response(system_prompt, payload, caller=caller)


def _call_preprocess_codegen_agent(system_prompt, payload, caller="preprocess-codegen"):
    return call_json_response(system_prompt, payload, caller=caller)


# Ask the LLM for dataset-level preprocessing decisions.
def generate_dataset_policy_spec(df: pd.DataFrame, dataset_profile: dict) -> dict:
    system_prompt = load_skill_prompt("dataset-policy-spec")
    column_summaries = {}
    for col in df.columns:
        nunique = df[col].nunique()
        column_summaries[col] = {
            "dtype": str(df[col].dtype),
            "nunique": nunique,
        }
        if nunique <= 20:
            column_summaries[col]["unique_values"] = df[col].dropna().unique().tolist()[:20]
    payload = {
        "columns": df.columns.tolist(),
        "column_summaries": column_summaries,
        "sample_rows": df.head(5).to_dict(orient="records"),
        "dataset_profile": dataset_profile,
    }
    return _call_preprocess_agent(system_prompt, payload, caller="dataset-policy-spec")


# Build per-column data profiles so the column-transform-spec LLM sees real distributions.
def _build_column_profiles(df: pd.DataFrame) -> dict:
    profiles = {}
    for col in df.columns:
        entry = {"dtype": str(df[col].dtype), "nunique": int(df[col].nunique())}
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe(percentiles=[0.01, 0.99])
            entry["min"] = float(desc["min"]) if "min" in desc else None
            entry["max"] = float(desc["max"]) if "max" in desc else None
            entry["mean"] = float(desc["mean"]) if "mean" in desc else None
            entry["p1"] = float(desc["1%"]) if "1%" in desc else None
            entry["p99"] = float(desc["99%"]) if "99%" in desc else None
        else:
            top_values = df[col].value_counts(dropna=False).head(10)
            entry["top_10_values"] = {str(k): int(v) for k, v in top_values.items()}
        profiles[col] = entry
    return profiles


# Ask the LLM for column-by-column transformation rules.
def generate_column_transform_spec(df: pd.DataFrame, dataset_policy_spec: dict, eda_report: dict | None = None) -> dict:
    system_prompt = load_skill_prompt("column-transform-spec")
    payload = {
        "columns": df.columns.tolist(),
        "sample_rows": df.head(10).to_dict(orient="records"),
        "column_profiles": _build_column_profiles(df),
        "dataset_policy_spec": dataset_policy_spec,
        "allowed_cleaning_primitives": allowed_cleaning_primitives(),
    }
    if eda_report:
        payload["eda_insights"] = {
            "top_discriminative_features": eda_report.get("top_discriminative_features", [])[:10],
            "raw_top_discriminative_features": eda_report.get("raw_top_discriminative_features", [])[:10],
            "model_eligible_top_discriminative_features": eda_report.get("model_eligible_top_discriminative_features", [])[:10],
            "leakage_alerts": eda_report.get("leakage_alerts", [])[:10],
            "high_correlation_pairs": eda_report.get("correlations", {}).get("high_pairs", [])[:10],
            "highly_skewed_columns": eda_report.get("skewness", {}).get("highly_skewed", {}),
            "high_cardinality_columns": eda_report.get("cardinality", {}).get("high_cardinality", []),
            "mnar_suspects": eda_report.get("missing_patterns", {}).get("mnar_suspects", []),
            "class_separability": eda_report.get("class_separability", {}).get("anova_top_features", [])[:10],
        }
    return _call_preprocess_agent(system_prompt, payload, caller="column-transform-spec")


# Ask the LLM to write the preprocessing code itself.
def generate_preprocessing_code(
    raw_df: pd.DataFrame,
    dataset_profile: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
) -> dict:
    system_prompt = load_skill_prompt("generate-preprocessing-code")
    payload = {
        "columns": raw_df.columns.tolist(),
        "sample_rows": raw_df.head(10).to_dict(orient="records"),
        "column_profiles": _build_column_profiles(raw_df),
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
        "allowed_cleaning_primitives": allowed_cleaning_primitives(),
    }
    result = _call_preprocess_codegen_agent(system_prompt, payload, caller="generate-preprocessing-code")
    return _with_codegen_audit(result, caller="generate-preprocessing-code", prompt_payload=payload)


# Ask the LLM to fix the last preprocessing attempt using failure feedback.
# When `escalate=True`, the call is routed through an escalated-caller identifier so
# operators can map it to a stronger model via OPENAI_MODEL_REPAIR_PREPROCESSING_CODE_ESCALATED.
def repair_preprocessing_code(
    *,
    previous_generated_code: dict,
    code_review: dict,
    execution_log: dict,
    validation_report: dict,
    dataset_profile: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
    escalate: bool = False,
) -> dict:
    system_prompt = load_skill_prompt("repair-preprocessing-code")
    payload = {
        "previous_generated_code": _public_codegen_payload(previous_generated_code),
        "code_review": code_review,
        "execution_log": execution_log,
        "validation_report": validation_report,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
        "allowed_cleaning_primitives": allowed_cleaning_primitives(),
    }
    if escalate:
        payload["escalation_notice"] = (
            "The same semantic-role violation appeared in the previous repair attempt. "
            "Treat this as a capability-ceiling signal: re-read the column_transform_spec's "
            "semantic_role and representation_intent carefully before emitting code. "
            "If a multi_value_set indicator is producing values outside {0,1}, the fix is almost "
            "always to replace a count-based encoding (str.count or summed dummies) with a "
            "presence-based one (str.get_dummies(sep=...) with whitespace-stripped column names)."
        )
    caller = "repair-preprocessing-code-escalated" if escalate else "repair-preprocessing-code"
    result = _call_preprocess_codegen_agent(system_prompt, payload, caller=caller)
    return _with_codegen_audit(result, caller=caller, prompt_payload=payload)


# Reject obviously unsafe or incomplete generated code before execution.
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

    # This is a lightweight denylist check, not a full sandbox.
    FORBIDDEN_MODULES = {"subprocess", "socket", "http", "urllib", "ftplib", "smtplib", "ctypes", "multiprocessing"}
    FORBIDDEN_OS_ATTRS = {"system", "popen", "exec", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe", "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"}
    FORBIDDEN_BUILTINS = {"eval", "exec", "__import__", "compile", "breakpoint"}

    # Reject inplace=True — incompatible with pandas 2.x Copy-on-Write.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    lineno = getattr(node, "lineno", "?")
                    issues.append({
                        "rule": "inplace_not_allowed",
                        "message": f"Line {lineno}: inplace=True is not allowed — pandas 2.x Copy-on-Write raises ChainedAssignmentError. Use assignment instead: df['col'] = df['col'].method(...).",
                    })

    # Walk the AST so we can catch unsafe patterns before running the generated file.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in FORBIDDEN_MODULES:
                    issues.append(
                        {
                            "rule": "forbidden_import",
                            "message": f"Importing {alias.name} is not allowed in generated preprocessing code.",
                        }
                    )
        elif isinstance(node, ast.ImportFrom):
            module_root = (node.module or "").split(".")[0]
            if module_root in FORBIDDEN_MODULES:
                issues.append(
                    {
                        "rule": "forbidden_import",
                        "message": f"Importing from {node.module} is not allowed in generated preprocessing code.",
                    }
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                issues.append(
                    {
                        "rule": "forbidden_call",
                        "message": f"Calling {node.func.id}() is not allowed in generated preprocessing code.",
                    }
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in FORBIDDEN_OS_ATTRS
            ):
                issues.append(
                    {
                        "rule": "forbidden_call",
                        "message": f"Calling os.{node.func.attr}() is not allowed in generated preprocessing code.",
                    }
                )
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                issues.append(
                    {
                        "rule": "forbidden_call",
                        "message": "Calling importlib.import_module() is not allowed in generated preprocessing code.",
                    }
                )

    return {
        "passed": not issues,
        "entrypoint": entrypoint,
        "issues": issues,
    }


# Delete older run folders so generated workspaces do not pile up forever.
def cleanup_old_workspaces(run_root_path: Path, keep_latest: int = 1) -> None:
    if not run_root_path.is_dir():
        return
    workspaces = sorted(
        [d for d in run_root_path.iterdir() if d.is_dir() and d.name.startswith("generated_preprocessing_")],
        key=lambda d: d.stat().st_mtime,
    )
    for workspace in workspaces[:-keep_latest] if keep_latest else workspaces:
        shutil.rmtree(workspace, ignore_errors=True)


# Run the generated preprocessing code in its own workspace and collect artifacts.
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
    # Run the generated file in a fresh Python process so failures stay isolated from the graph process.
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
                "    raw_df = pd.read_csv(raw_frame_path, low_memory=False)",
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

    timeout_seconds = 300
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
        raw_stdout = exc.stdout or ""
        raw_stderr = exc.stderr or ""
        execution_log = {
            "returncode": None,
            "stdout": raw_stdout.decode("utf-8", errors="replace") if isinstance(raw_stdout, bytes) else raw_stdout,
            "stderr": raw_stderr.decode("utf-8", errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
        }

    # A run is only considered usable if both the process succeeded and every required file exists.
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


# Roles that must NEVER appear in the feature frame (identifier-like or target).
_ABSENT_ROLES = {"identifier", "group_identifier", "target", "leakage_risk_feature"}

# Roles whose encoded output may explode into multiple prefixed indicator columns.
_PREFIX_ROLES = {"multi_value_set", "unordered_categorical", "free_text"}


def _unique_non_null(series: pd.Series) -> set:
    vals = series.dropna().unique()
    return set(vals.tolist()) if len(vals) else set()


def _is_binary_set(values: set) -> bool:
    return all(float(v) in (0.0, 1.0) for v in values if not isinstance(v, bool))


# Deterministic contract validator: for each column's declared semantic_role,
# check the post-preprocessing output against the role's invariant. Produces
# structured findings the repair prompt can act on directly.
def validate_semantic_roles(
    feature_frame: pd.DataFrame,
    column_transform_spec: dict,
) -> list[dict]:
    return _validate_semantic_roles_impl(feature_frame, column_transform_spec)


# Cross-field semantic invariants that are not captured by per-column role contracts.
# Each check emits a structured finding using the same shape as validate_semantic_roles.
# Thresholds are data-aware (relative to raw frame) where a magic-number absolute
# would be brittle across datasets.
def validate_semantic_invariants(
    feature_frame: pd.DataFrame,
    raw_frame: pd.DataFrame | None,
    column_transform_spec: dict,
) -> list[dict]:
    return _validate_semantic_invariants_impl(feature_frame, raw_frame, column_transform_spec)


def _spec_transforms(column_transform_spec: dict | None) -> dict:
    raw_spec = column_transform_spec or {}
    return raw_spec.get("transforms") or raw_spec.get("columns") or {}


def _infer_group_column(column_transform_spec: dict | None, raw_frame: pd.DataFrame) -> str | None:
    transforms = _spec_transforms(column_transform_spec)
    for col, spec in transforms.items():
        if isinstance(spec, dict) and spec.get("semantic_role") == "group_identifier" and col in raw_frame.columns:
            return col
    if "Customer_ID" in raw_frame.columns:
        return "Customer_ID"
    return None


def _coerce_bound_value(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _declared_numeric_bounds(spec: dict) -> tuple[float | None, float | None]:
    primitive_params = spec.get("primitive_params") if isinstance(spec, dict) else None
    if isinstance(primitive_params, dict):
        bounds = primitive_params.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            low = _coerce_bound_value(bounds[0])
            high = _coerce_bound_value(bounds[1])
            if low is not None or high is not None:
                return low, high
        low = _coerce_bound_value(primitive_params.get("lower_bound"))
        high = _coerce_bound_value(primitive_params.get("upper_bound"))
        if low is not None or high is not None:
            return low, high

    cleaning = str(spec.get("cleaning") or "")
    patterns = [
        re.compile(r"clip to \[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", re.IGNORECASE),
        re.compile(r"values?\s*<\s*(-?\d+(?:\.\d+)?)\s*or\s*>\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(cleaning)
        if match:
            low = _coerce_bound_value(match.group(1))
            high = _coerce_bound_value(match.group(2))
            if low is not None or high is not None:
                return low, high

    return None, None


def normalize_preprocessing_artifacts(
    execution_result: dict,
    column_transform_spec: dict,
) -> dict:
    artifacts = execution_result.get("artifacts", {})
    workspace_path = Path(execution_result.get("workspace_path", ""))
    raw_frame_path = Path(execution_result.get("raw_frame_path", workspace_path / "raw_frame.csv"))
    feature_frame_path = Path(artifacts.get("feature_frame.csv", ""))
    cleaned_frame_path = Path(artifacts.get("cleaned_frame.csv", ""))

    report = {
        "applied": False,
        "normalized_columns": [],
        "actions": [],
        "skipped": [],
    }

    if not raw_frame_path.is_file():
        report["skipped"].append("raw_frame_missing")
        return report
    if not feature_frame_path.is_file():
        report["skipped"].append("feature_frame_missing")
        return report

    try:
        raw_frame = pd.read_csv(raw_frame_path, low_memory=False)
        feature_frame = pd.read_csv(feature_frame_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive artifact guard
        report["skipped"].append(f"artifact_read_error:{type(exc).__name__}")
        return report

    cleaned_frame = None
    if cleaned_frame_path.is_file():
        try:
            cleaned_frame = pd.read_csv(cleaned_frame_path, low_memory=False)
        except Exception as exc:  # pragma: no cover - defensive artifact guard
            report["skipped"].append(f"cleaned_frame_read_error:{type(exc).__name__}")

    if len(raw_frame) != len(feature_frame):
        report["skipped"].append(
            f"row_count_mismatch:raw={len(raw_frame)} feature={len(feature_frame)}"
        )
        return report
    if cleaned_frame is not None and len(cleaned_frame) != len(raw_frame):
        report["skipped"].append(
            f"cleaned_row_count_mismatch:raw={len(raw_frame)} cleaned={len(cleaned_frame)}"
        )
        cleaned_frame = None

    transforms = _spec_transforms(column_transform_spec)
    group_column = _infer_group_column(column_transform_spec, raw_frame)
    changed_feature = False
    changed_cleaned = False
    normalized_age = None

    if "Age" in transforms and "Age" in raw_frame.columns and "Age" in feature_frame.columns:
        age_spec = transforms.get("Age") or {}
        parsed_age = _parse_age_series(raw_frame["Age"])
        parsed_age = parsed_age.where(parsed_age.between(18, 100))
        group_values = raw_frame[group_column] if group_column and group_column in raw_frame.columns else None
        normalized_age = _fill_by_group_then_global(parsed_age, group_values, default=30.0).clip(18, 100)

        feature_frame["Age"] = normalized_age.astype(float)
        changed_feature = True
        if cleaned_frame is not None and "Age" in cleaned_frame.columns:
            cleaned_frame["Age"] = normalized_age.astype(float)
            changed_cleaned = True
        report["normalized_columns"].append("Age")
        report["actions"].append(
            {
                "column": "Age",
                "action": "reparsed_from_raw_and_clipped_to_human_range",
                "group_column": group_column,
                "valid_range": [18, 100],
                "source_cleaning": age_spec.get("cleaning"),
            }
        )

    if "Credit_History_Age" in transforms and "Credit_History_Age" in raw_frame.columns and "Credit_History_Age" in feature_frame.columns:
        cha_spec = transforms.get("Credit_History_Age") or {}
        age_source = normalized_age
        if age_source is None and "Age" in feature_frame.columns:
            age_source = feature_frame["Age"]
        elif cleaned_frame is not None and "Age" in cleaned_frame.columns:
            age_source = cleaned_frame["Age"]
        elif "Age" in raw_frame.columns:
            age_source = raw_frame["Age"]

        parsed_months = _parse_credit_history_age_series(raw_frame["Credit_History_Age"])
        parsed_months = parsed_months.where(parsed_months >= 0)
        parsed_months = parsed_months.where(parsed_months <= 1000)
        parsed_months = _cap_credit_history_by_adulthood(parsed_months, age_source)

        group_values = raw_frame[group_column] if group_column and group_column in raw_frame.columns else None
        default_months = np.nan
        if age_source is not None:
            age_cap = _cap_credit_history_by_adulthood(pd.Series([0.0] * len(raw_frame)), age_source)
            if age_cap.notna().any():
                default_months = float(age_cap.median())
        if np.isnan(default_months):
            default_months = 0.0
        parsed_months = _fill_by_group_then_global(parsed_months, group_values, default=default_months)
        parsed_months = _cap_credit_history_by_adulthood(parsed_months, age_source).round(0)

        feature_frame["Credit_History_Age"] = parsed_months.astype(float)
        changed_feature = True
        if cleaned_frame is not None and "Credit_History_Age" in cleaned_frame.columns:
            cleaned_frame["Credit_History_Age"] = parsed_months.astype(float)
            changed_cleaned = True
        report["normalized_columns"].append("Credit_History_Age")
        report["actions"].append(
            {
                "column": "Credit_History_Age",
                "action": "reparsed_from_raw_and_capped_by_adulthood",
                "group_column": group_column,
                "upper_bound_months": 1000,
                "source_cleaning": cha_spec.get("cleaning"),
            }
        )

    for col, spec in transforms.items():
        if not isinstance(spec, dict) or spec.get("semantic_role") != "multi_value_set":
            continue
        if col not in raw_frame.columns:
            continue

        prefix = f"{col}_"
        missing_col = f"{col}_missing"
        sibling_cols = [c for c in feature_frame.columns if c.startswith(prefix) and c != missing_col]
        cleaning_text = str(spec.get("cleaning") or "").lower()
        should_manage_missing = bool(sibling_cols) or missing_col in feature_frame.columns or "missing" in cleaning_text
        if not should_manage_missing:
            continue

        missing_mask = _normalized_missing_string_mask(raw_frame[col]).astype(int)
        feature_frame[missing_col] = missing_mask
        changed_feature = True

        dropped_cols: list[str] = []
        for sibling in list(sibling_cols):
            suffix = sibling[len(prefix):].replace("_", " ").strip().lower()
            if suffix in {"not specified", "nan", "none", "null"}:
                feature_frame = feature_frame.drop(columns=[sibling])
                dropped_cols.append(sibling)
                changed_feature = True

        remaining_siblings = [
            c for c in feature_frame.columns if c.startswith(prefix) and c != missing_col
        ]
        for sibling in remaining_siblings:
            # Convert to plain Python object dtype first so that pandas nullable
            # integer (Int64) or float (Float64) types don't raise a TypeError
            # in fillna() when the fill value can't be safely cast.
            raw_col = feature_frame[sibling].astype(object)
            coerced = pd.to_numeric(raw_col, errors="coerce").fillna(0.0)
            feature_frame[sibling] = (coerced != 0).astype("int64")
            feature_frame.loc[missing_mask.astype(bool), sibling] = 0

        report["normalized_columns"].append(col)
        report["actions"].append(
            {
                "column": col,
                "action": "normalized_missing_sentinel_and_zeroed_siblings",
                "dropped_columns": dropped_cols,
                "sibling_count": len(remaining_siblings),
                "missing_rate": float(missing_mask.mean()),
            }
        )

    for col, spec in transforms.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("action") in {"drop", "quarantine"}:
            continue
        if spec.get("semantic_role") not in {"numeric_continuous", "numeric_count"}:
            continue
        if col not in feature_frame.columns:
            continue

        lower, upper = _declared_numeric_bounds(spec)
        if lower is None and upper is None:
            continue

        original_feature = _coerce_numeric(feature_frame[col])
        bounded_feature = original_feature.clip(lower=lower, upper=upper)
        feature_changed = not original_feature.equals(bounded_feature)
        if feature_changed:
            feature_frame[col] = bounded_feature.astype(float)
            changed_feature = True

        cleaned_changed = False
        if cleaned_frame is not None and col in cleaned_frame.columns:
            original_cleaned = _coerce_numeric(cleaned_frame[col])
            bounded_cleaned = original_cleaned.clip(lower=lower, upper=upper)
            cleaned_changed = not original_cleaned.equals(bounded_cleaned)
            if cleaned_changed:
                cleaned_frame[col] = bounded_cleaned.astype(float)
                changed_cleaned = True

        if feature_changed or cleaned_changed:
            report["normalized_columns"].append(col)
            report["actions"].append(
                {
                    "column": col,
                    "action": "enforced_declared_numeric_bounds",
                    "bounds": [lower, upper],
                    "source_cleaning": spec.get("cleaning"),
                }
            )

    if changed_feature:
        feature_frame.to_csv(feature_frame_path, index=False)
    if cleaned_frame is not None and changed_cleaned:
        cleaned_frame.to_csv(cleaned_frame_path, index=False)

    if report["normalized_columns"]:
        report["applied"] = True
        report["normalized_columns"] = sorted(set(report["normalized_columns"]))

    return report


# Check that generated artifacts are usable and do not violate split or leakage rules.
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
    if "target_column" not in dataset_policy_spec:
        return {
            "passed": False,
            "checks": {},
            "errors": [{"rule": "missing_spec_field", "message": "dataset_policy_spec is missing 'target_column'."}],
        }
    target_column = dataset_policy_spec["target_column"]
    group_column = dataset_policy_spec.get("group_column")
    split_strategy = dataset_policy_spec.get("split_strategy", {})

    checks: dict[str, bool] = {}
    errors: list[dict] = []
    try:
        deterministic_normalization = normalize_preprocessing_artifacts(
            execution_result,
            column_transform_spec,
        )
    except Exception as exc:
        # Normalization is best-effort; a crash here must not kill the validation
        # contract — the repair loop depends on validate returning a structured
        # report, not raising an exception.
        deterministic_normalization = {
            "applied": False,
            "skipped": [f"normalization_error:{type(exc).__name__}:{exc}"],
        }

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

    # Deterministic check: target.csv must contain raw string labels, not integer codes.
    # If the preprocessing code encoded the target (e.g. Good→0, Standard→1, Poor→2),
    # the graph will reconstruct class_names as ['0','1','2'] instead of the real labels,
    # breaking every downstream step that references class names (confusion matrix, SHAP,
    # local casebook, explain-risk).  We detect this by comparing target.csv values
    # against the original raw target column in raw_frame.csv.
    if target_path.is_file() and raw_frame_path.is_file() and target_column:
        try:
            target_df = pd.read_csv(target_path)
            target_series_raw = target_df.iloc[:, 0]
            raw_frame_check = pd.read_csv(raw_frame_path, usecols=[target_column], low_memory=False)
            original_unique = set(raw_frame_check[target_column].dropna().astype(str).unique())
            saved_unique = set(target_series_raw.dropna().astype(str).unique())
            # If the saved values are all numeric integers and the original values are
            # not (e.g. 'Good','Standard','Poor' → 0,1,2), the target was encoded.
            original_looks_numeric = all(
                v.lstrip("-").replace(".", "", 1).isdigit() for v in original_unique
            )
            saved_looks_integer = all(
                v.lstrip("-").isdigit() for v in saved_unique
            )
            is_encoded = not original_looks_numeric and saved_looks_integer
            # Secondary check: if saved values are contiguous integers 0..K-1 while
            # original values differ completely, it is definitely encoded.
            if not is_encoded and saved_looks_integer:
                try:
                    int_vals = sorted(int(v) for v in saved_unique)
                    is_encoded = int_vals == list(range(len(int_vals))) and not saved_unique.issubset(original_unique)
                except ValueError:
                    pass
            add_check(
                "target_labels_not_encoded",
                not is_encoded,
                (
                    f"target.csv contains integer codes {sorted(saved_unique)} instead of raw "
                    f"category labels {sorted(original_unique)}. The preprocessing function must "
                    "save the original target values (e.g. 'Good', 'Standard', 'Poor') directly — "
                    "do NOT label-encode or map the target to integers before saving. "
                    "The graph re-encodes the target after validation; encoding it twice produces "
                    "class_names=['0','1','2'] which breaks confusion matrices, SHAP, and explain-risk."
                ),
            )
        except Exception as _exc:
            # Non-fatal: if we can't read either file the target_file_exists check already flagged it.
            checks.setdefault("target_labels_not_encoded", True)

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
            mangled_duplicate_like = [
                col for col in feature_frame.columns
                if re.search(r"\.\d+$", str(col))
            ]
            add_check(
                "no_mangled_duplicate_columns",
                not mangled_duplicate_like,
                (
                    "Feature frame contains duplicate-like columns created by pandas "
                    f"name mangling: {mangled_duplicate_like[:10]}. "
                    "This usually means a one-hot/multi-hot encoder produced the same "
                    "logical category twice (often after whitespace normalization)."
                ),
            )
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
                # Validate group leakage from the raw data and saved indices instead of trusting generated summaries.
                manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
                train_indices = manifest["train_indices"]
                test_indices = manifest["test_indices"]
                raw_frame = pd.read_csv(raw_frame_path, low_memory=False)
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

    # --- Data quality checks (warnings feed into repair loop but do not block pass) ---
    warnings: list[dict] = []
    cleaned_frame_path = Path(artifacts.get("cleaned_frame.csv", ""))

    if feature_frame is not None:
        # Remaining NaNs — imputation should leave zero
        nan_counts = feature_frame.isna().sum()
        total_nans = int(nan_counts.sum())
        if total_nans > 0:
            nan_cols = {col: int(v) for col, v in nan_counts.items() if v > 0}
            warnings.append({
                "rule": "remaining_nans",
                "message": f"Feature frame has {total_nans} NaN values across {len(nan_cols)} columns.",
                "details": nan_cols,
            })

        # Feature count — flag cardinality explosion from one-hot encoding
        n_features = len(feature_frame.columns)
        if n_features > 200:
            warnings.append({
                "rule": "high_feature_count",
                "message": f"Feature frame has {n_features} columns — possible cardinality explosion from encoding. Consider using label encoding or grouping rare categories for high-cardinality columns.",
            })

        # Constant features — zero variance adds noise, no signal
        constant_cols = [col for col in feature_frame.columns if feature_frame[col].nunique(dropna=False) <= 1]
        if constant_cols:
            warnings.append({
                "rule": "constant_features",
                "message": f"{len(constant_cols)} constant feature(s) detected: {constant_cols[:10]}. These add no predictive value.",
            })

    # Cleaned frame NaN check — should also be zero after imputation
    if cleaned_frame_path.is_file():
        try:
            cleaned_frame = pd.read_csv(cleaned_frame_path, nrows=1000)
            cleaned_nans = int(cleaned_frame.isna().sum().sum())
            if cleaned_nans > 0:
                warnings.append({
                    "rule": "cleaned_frame_has_nans",
                    "message": f"cleaned_frame.csv still has NaN values (sampled first 1000 rows: {cleaned_nans} NaNs). Imputation may be incomplete.",
                })
        except Exception:
            pass  # Non-critical — skip if unreadable

    # --- Semantic role contract checks (deterministic, per-column) ---
    role_violations: list[dict] = []
    if feature_frame is not None and column_transform_spec:
        try:
            role_violations = validate_semantic_roles(feature_frame, column_transform_spec)
        except Exception as exc:  # pragma: no cover - defensive: validator must never crash the loop
            role_violations = []
            warnings.append({
                "rule": "semantic_role_validator_error",
                "message": f"semantic role validator raised {type(exc).__name__}: {exc}",
            })

    # --- Cross-field semantic invariants (Age, Credit_History_Age, missingness) ---
    invariant_violations: list[dict] = []
    if feature_frame is not None:
        raw_frame_for_invariants = None
        if raw_frame_path.is_file():
            try:
                raw_frame_for_invariants = pd.read_csv(raw_frame_path, low_memory=False)
            except Exception:
                raw_frame_for_invariants = None
        try:
            invariant_violations = validate_semantic_invariants(
                feature_frame, raw_frame_for_invariants, column_transform_spec or {}
            )
        except Exception as exc:  # pragma: no cover
            invariant_violations = []
            warnings.append({
                "rule": "semantic_invariant_validator_error",
                "message": f"semantic invariant validator raised {type(exc).__name__}: {exc}",
            })

    passed = all(checks.values()) and not errors and not role_violations and not invariant_violations

    # --- Persist preprocessing contract report artifact ---
    if workspace_path and workspace_path.exists():
        contract_report = {
            "workspace": str(workspace_path),
            "passed": passed,
            "deterministic_normalization": deterministic_normalization,
            "structural_checks": checks,
            "errors": errors,
            "warnings": warnings,
            "role_violations": role_violations,
            "cross_field_violations": invariant_violations,
        }
        try:
            (workspace_path / "preprocessing_contract_report.json").write_text(
                json.dumps(contract_report, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover - defensive: never crash on artifact write
            pass

    return {
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "role_violations": role_violations,
        "cross_field_violations": invariant_violations,
        "deterministic_normalization": deterministic_normalization,
    }


def _build_column_stats(df: pd.DataFrame) -> list[dict]:
    """Build per-column statistics for the LLM quality reviewer."""
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
        stats.append(entry)
    return stats


# Ask the LLM to review the quality of preprocessing output.
def review_preprocessing_quality(
    execution_result: dict,
    dataset_policy_spec: dict,
    column_transform_spec: dict,
    previous_audit_report: dict | None = None,
) -> dict:
    system_prompt = load_skill_prompt("audit-preprocessing")
    artifacts = execution_result.get("artifacts", {})

    feature_frame_path = Path(artifacts.get("feature_frame.csv", ""))
    target_path = Path(artifacts.get("target.csv", ""))
    report_path = Path(artifacts.get("preprocessing_report.json", ""))

    feature_frame = pd.read_csv(feature_frame_path)

    # 5 rows for pattern-checking; full-frame stats carry the distribution picture.
    feature_sample = feature_frame.head(5).to_dict(orient="records")
    feature_stats = _build_column_stats(feature_frame)

    target_distribution = None
    if target_path.is_file():
        target_df = pd.read_csv(target_path)
        target_distribution = target_df.iloc[:, 0].value_counts(dropna=False).to_dict()

    preprocessing_report = None
    if report_path.is_file():
        try:
            preprocessing_report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    payload = {
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
        "feature_sample": feature_sample,
        "feature_stats": feature_stats,
        "target_distribution": target_distribution,
        "feature_column_count": len(feature_frame.columns),
        "preprocessing_report": preprocessing_report,
    }
    if previous_audit_report:
        payload["previous_audit_report"] = previous_audit_report

    return _call_preprocess_agent(system_prompt, payload, caller="audit-preprocessing")
