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
from bt5151_credit_risk.skill_prompts import load_skill_prompt


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
    }
    if eda_report:
        payload["eda_insights"] = {
            "top_discriminative_features": eda_report.get("top_discriminative_features", [])[:10],
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
    }
    return _call_preprocess_codegen_agent(system_prompt, payload, caller="generate-preprocessing-code")


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
        "previous_generated_code": previous_generated_code,
        "code_review": code_review,
        "execution_log": execution_log,
        "validation_report": validation_report,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
        "column_transform_spec": column_transform_spec,
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
    return _call_preprocess_codegen_agent(system_prompt, payload, caller=caller)


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
    raw_spec = column_transform_spec or {}
    # Normalize: accept both "transforms" (current schema) and "columns" (deprecated).
    transforms = raw_spec.get("transforms") or raw_spec.get("columns") or {}
    frame_cols = list(feature_frame.columns)
    frame_col_set = set(frame_cols)
    violations: list[dict] = []

    # Fail loudly if no column declares a semantic_role — the validator is a no-op
    # without roles, and we need to know when the spec regresses.
    has_any_role = any(
        isinstance(spec, dict) and spec.get("semantic_role")
        for spec in transforms.values()
    )
    if transforms and not has_any_role:
        violations.append({
            "column": None,
            "declared_role": None,
            "violation": "schema_missing_semantic_roles",
            "observed": "column_transform_spec contains columns but none declare semantic_role",
            "expected": "every column should have a semantic_role for contract enforcement",
            "likely_cause": (
                "the column-transform-spec skill may have regressed to the old schema. "
                "Check that the reasoning model is using the updated prompt with the "
                "12-role taxonomy. This is a spec issue, not a codegen issue."
            ),
        })
        return violations

    def record(column, role, violation, observed, expected, likely_cause):
        violations.append({
            "column": column,
            "declared_role": role,
            "violation": violation,
            "observed": observed,
            "expected": expected,
            "likely_cause": likely_cause,
        })

    def matching_columns(col: str, role: str) -> list[str]:
        exact = [col] if col in frame_col_set else []
        if role in _PREFIX_ROLES:
            prefixed = [c for c in frame_cols if c.startswith(f"{col}_")]
            return exact + prefixed
        return exact

    for col, spec in transforms.items():
        if not isinstance(spec, dict):
            continue
        role = spec.get("semantic_role")
        action = spec.get("action", "keep")
        intent = spec.get("representation_intent")

        if not role:
            # If other columns in this spec already declare roles, a missing role
            # is a partial regression — the reasoning model dropped the field for
            # this column. Emit a violation so repair / the operator can see it.
            if has_any_role and action not in ("drop", "quarantine"):
                record(
                    column=col,
                    role=None,
                    violation="missing_semantic_role",
                    observed=f"column spec has no semantic_role (action={action})",
                    expected="every kept column must declare a semantic_role",
                    likely_cause=(
                        "the reasoning model declared semantic_role on some columns "
                        "but missed this one. This is a spec issue — re-run "
                        "column-transform-spec or patch the spec manually."
                    ),
                )
            continue

        # Absence checks: identifier / group / target / leakage / dropped / quarantined
        if role in _ABSENT_ROLES or action in ("drop", "quarantine"):
            leaked = matching_columns(col, role)
            if leaked:
                record(
                    column=col,
                    role=role,
                    violation="must_be_absent",
                    observed=f"present in feature frame as {leaked[:5]}",
                    expected=f"absent (role={role}, action={action})",
                    likely_cause=(
                        "drop step missed this column, or an encoded derivative of "
                        "it remains. Remove the column (and any prefix-derived indicators) "
                        "before saving feature_frame.csv."
                    ),
                )
            continue

        # Kept features: check each matching column against role invariant.
        matched = matching_columns(col, role)
        if not matched:
            record(
                column=col,
                role=role,
                violation="missing",
                observed="no column or prefixed indicators found in feature frame",
                expected=f"kept column (role={role}) must appear in feature frame",
                likely_cause="column was dropped despite action=keep, or was renamed without updating the spec",
            )
            continue

        for m in matched:
            series = feature_frame[m]
            numeric = pd.api.types.is_numeric_dtype(series)
            uniq = _unique_non_null(series) if numeric else set()

            if role == "binary_flag":
                if not numeric or not uniq.issubset({0, 1, 0.0, 1.0}):
                    bad = sorted([v for v in uniq if float(v) not in (0.0, 1.0)])[:5]
                    record(
                        column=m, role=role,
                        violation="not_binary",
                        observed=f"values include {bad}" if bad else f"dtype={series.dtype}",
                        expected="values ⊆ {0, 1}",
                        likely_cause="binary_flag must be mapped to {0,1} numerically; check for unmapped string values or dtype=object.",
                    )

            elif role == "multi_value_set":
                # Exact-name column still present means the raw list wasn't encoded.
                if m == col:
                    record(
                        column=m, role=role,
                        violation="not_exploded",
                        observed="original delimited column still present",
                        expected="exploded into prefixed binary indicators, original dropped",
                        likely_cause="use str.get_dummies(sep=...) on the cleaned column, then drop the original",
                    )
                    continue
                if intent == "binary_membership" or intent is None:
                    if numeric and not _is_binary_set(uniq):
                        bad = sorted([v for v in uniq if float(v) not in (0.0, 1.0)])[:5]
                        record(
                            column=m, role=role,
                            violation="indicator_not_binary",
                            observed=f"values include {bad}",
                            expected="values ⊆ {0, 1} (presence indicator)",
                            likely_cause=(
                                "multi-hot encoder counted occurrences instead of presence. "
                                "Use str.get_dummies(sep=', ') which produces {0,1} indicators, "
                                "or apply `int(token in set_of_tokens)` — never str.count or sum of dummies."
                            ),
                        )

            elif role == "ordered_categorical":
                if numeric and uniq:
                    non_int = [v for v in uniq if float(v) != int(float(v))]
                    negatives = [v for v in uniq if float(v) < 0]
                    if non_int or negatives:
                        record(
                            column=m, role=role,
                            violation="not_ordinal_encoded",
                            observed=f"values={sorted(uniq)[:10]}",
                            expected="integer codes 0..K-1 preserving declared order",
                            likely_cause="ordered_categorical should be mapped to integer ordinal codes preserving semantic order, not scaled or one-hot encoded.",
                        )
                    else:
                        # Values are non-negative integers — verify contiguous 0..K-1.
                        int_vals = sorted(int(float(v)) for v in uniq)
                        expected_seq = list(range(len(int_vals)))
                        if int_vals != expected_seq:
                            record(
                                column=m, role=role,
                                violation="ordinal_not_contiguous",
                                observed=f"values={int_vals[:10]}",
                                expected=f"contiguous 0..{len(int_vals)-1}",
                                likely_cause=(
                                    "ordinal encoding must map categories to contiguous "
                                    "integers starting at 0. Gaps (e.g. {0,2,5}) or "
                                    "1-based indexing (e.g. {1,2,3}) break tree splits "
                                    "and linear model assumptions. Re-map to 0..K-1."
                                ),
                            )

            elif role == "unordered_categorical":
                if intent == "deferred":
                    # Deferred columns must remain as object-dtype strings in the
                    # canonical base frame.  FE encodes them per model view.
                    if m == col and numeric:
                        record(
                            column=m, role=role,
                            violation="deferred_column_encoded_prematurely",
                            observed=f"dtype={series.dtype} (numeric)",
                            expected="object dtype (string) — deferred encoding must be applied by FE, not preprocessing",
                            likely_cause=(
                                "The column-transform-spec marked this column "
                                "representation_intent='deferred' but the preprocessing code "
                                "encoded it anyway. Remove the encoding step for deferred columns — "
                                "they must reach the FE node as cleaned string columns."
                            ),
                        )
                else:
                    # Non-deferred: base column must be gone (replaced by encoded form).
                    if m == col and not numeric:
                        record(
                            column=m, role=role,
                            violation="unordered_categorical_not_encoded",
                            observed=f"dtype=object (raw string)",
                            expected=f"encoded as numeric per intent='{intent}'",
                            likely_cause=(
                                f"unordered_categorical with intent='{intent}' must be encoded "
                                "to numeric before saving feature_frame.csv. "
                                "For one_hot: use str.get_dummies and drop the original. "
                                "For frequency_encoded: replace with frequency counts. "
                                "Only intent='deferred' may remain as object dtype."
                            ),
                        )
                    # Check one-hot dummies are binary.
                    if intent == "one_hot" and m != col and numeric:
                        if not _is_binary_set(uniq):
                            bad = sorted([v for v in uniq if float(v) not in (0.0, 1.0)])[:5]
                            record(
                                column=m, role=role,
                                violation="one_hot_not_binary",
                                observed=f"values include {bad}",
                                expected="values ⊆ {0, 1}",
                                likely_cause="one-hot indicator is not binary; duplicate categories may have been summed. Strip category labels and deduplicate before pivoting.",
                            )

            elif role in {"numeric_count", "numeric_continuous"}:
                # Dtype gate: numeric roles MUST have numeric dtype. One-sided
                # `if numeric and ...` would silently skip an Age-as-string column.
                if not numeric:
                    record(
                        column=m, role=role,
                        violation="not_numeric_dtype",
                        observed=f"dtype={series.dtype}",
                        expected="numeric dtype (int/float)",
                        likely_cause=(
                            "column remained non-numeric after cleaning. Verify order: "
                            "(1) replace garbage tokens with NaN, (2) strip non-numeric "
                            "artifacts from strings, (3) pd.to_numeric(errors='coerce'), "
                            "(4) impute. Do not run imputation on an object-dtype series."
                        ),
                    )
                    continue

                if role == "numeric_count":
                    if (series.dropna() < 0).any():
                        record(
                            column=m, role=role,
                            violation="negative_count",
                            observed=f"min={float(series.min())}",
                            expected="values >= 0",
                            likely_cause="count values must be non-negative; clip lower bound to 0 or check parsing (a '-' in the raw string may have been preserved).",
                        )

                else:  # numeric_continuous
                    if series.isna().any():
                        record(
                            column=m, role=role,
                            violation="has_nan",
                            observed=f"{int(series.isna().sum())} NaN values",
                            expected="no NaN (imputation should fill)",
                            likely_cause="imputation step did not cover this column; verify the imputation branch runs after cleaning.",
                        )
                    if np.isinf(series).any():
                        n_inf = int(np.isinf(series).sum())
                        record(
                            column=m, role=role,
                            violation="has_inf",
                            observed=f"{n_inf} inf/-inf values",
                            expected="finite values only (no inf/-inf)",
                            likely_cause=(
                                "a transform produced infinity — common causes: "
                                "division by zero, log(0), or inverse of near-zero values. "
                                "Replace inf with np.nan then impute, or clip before the transform."
                            ),
                        )

            elif role == "temporal_feature":
                # Temporal columns are typically decomposed; if the original string column
                # still appears as object dtype, decomposition did not run.
                if m == col and not numeric and series.dtype == object:
                    record(
                        column=m, role=role,
                        violation="not_decomposed",
                        observed=f"original temporal column present as dtype=object",
                        expected="decomposed into numeric features (year/month/dow/etc.) per representation_intent",
                        likely_cause="temporal_feature must be parsed with pd.to_datetime and decomposed into numeric components; drop the original string column after decomposition.",
                    )

    return violations


# Cross-field semantic invariants that are not captured by per-column role contracts.
# Each check emits a structured finding using the same shape as validate_semantic_roles.
# Thresholds are data-aware (relative to raw frame) where a magic-number absolute
# would be brittle across datasets.
def validate_semantic_invariants(
    feature_frame: pd.DataFrame,
    raw_frame: pd.DataFrame | None,
    column_transform_spec: dict,
) -> list[dict]:
    violations: list[dict] = []

    def record(column, violation, observed, expected, likely_cause):
        violations.append({
            "column": column,
            "declared_role": None,
            "violation": violation,
            "observed": observed,
            "expected": expected,
            "likely_cause": likely_cause,
        })

    cols = set(feature_frame.columns)

    if "Age" in cols:
        age = feature_frame["Age"]
        if pd.api.types.is_numeric_dtype(age):
            non_null = age.dropna()
            if len(non_null):
                amin = float(non_null.min())
                amax = float(non_null.max())
                if amin < 18 or amax > 100:
                    record(
                        column="Age",
                        violation="age_out_of_range",
                        observed=f"range=[{amin:.1f}, {amax:.1f}]",
                        expected="18 <= Age <= 100",
                        likely_cause=(
                            "Age contains values outside human plausibility. "
                            "Check parsing (negative signs in strings), clip to [18, 100], "
                            "or coerce garbage to NaN and impute."
                        ),
                    )
                # Cardinality check is only meaningful on production-sized frames;
                # skip for tiny (toy / test) frames where nunique is bounded by row count.
                if len(feature_frame) >= 100 and int(age.nunique()) < 10:
                    record(
                        column="Age",
                        violation="age_low_cardinality",
                        observed=f"nunique={int(age.nunique())} on {len(feature_frame)} rows",
                        expected="nunique >= 10",
                        likely_cause=(
                            "Age collapsed to a handful of values — likely a failed "
                            "parse that imputed the same fallback across most rows."
                        ),
                    )

    if "Credit_History_Age" in cols:
        cha = feature_frame["Credit_History_Age"]
        if pd.api.types.is_numeric_dtype(cha):
            non_null = cha.dropna()
            if len(non_null):
                cmin = float(non_null.min())
                cmax = float(non_null.max())
                if cmin < 0 or cmax > 1000:
                    record(
                        column="Credit_History_Age",
                        violation="credit_history_age_out_of_range",
                        observed=f"range=[{cmin:.1f}, {cmax:.1f}] months",
                        expected="0 <= months <= 1000",
                        likely_cause=(
                            "Credit_History_Age is expected in months. Values outside "
                            "[0, 1000] suggest unit confusion (years vs months) or "
                            "bad parsing."
                        ),
                    )

                if raw_frame is not None and "Credit_History_Age" in raw_frame.columns:
                    raw_nunique = int(raw_frame["Credit_History_Age"].dropna().nunique())
                    out_nunique = int(cha.nunique())
                    if raw_nunique >= 50:
                        ratio = out_nunique / raw_nunique if raw_nunique else 1.0
                        if ratio < 0.3:
                            record(
                                column="Credit_History_Age",
                                violation="credit_history_age_low_cardinality",
                                observed=f"nunique_out={out_nunique}, nunique_raw={raw_nunique}, ratio={ratio:.2f}",
                                expected="nunique_out / nunique_raw >= 0.3",
                                likely_cause=(
                                    "Output cardinality is a small fraction of raw — "
                                    "the regex likely captured only the year component "
                                    "and discarded months."
                                ),
                            )

                non_year_mult = int(((non_null.astype(float) % 12) != 0).sum())
                frac_non_year = non_year_mult / len(non_null)
                if frac_non_year < 0.05:
                    record(
                        column="Credit_History_Age",
                        violation="credit_history_age_year_only",
                        observed=f"only {frac_non_year*100:.1f}% of values have months%12!=0",
                        expected=">= 5% of non-null values should have a non-zero month remainder",
                        likely_cause=(
                            "The parser captured years only (every value is a multiple of 12). "
                            "Use a regex that extracts both year and month, e.g. "
                            r"r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?'."
                        ),
                    )

                if "Age" in cols and pd.api.types.is_numeric_dtype(feature_frame["Age"]):
                    paired = pd.concat([feature_frame["Age"], cha], axis=1).dropna()
                    if len(paired):
                        max_months = (paired["Age"] - 18).clip(lower=0) * 12 * 1.1
                        violating = (paired["Credit_History_Age"] > max_months).sum()
                        frac_violating = violating / len(paired)
                        if frac_violating > 0.05:
                            record(
                                column="Credit_History_Age",
                                violation="credit_history_age_exceeds_adulthood",
                                observed=f"{frac_violating*100:.1f}% of rows have CHA > (Age-18)*12*1.1",
                                expected="<= 5% violating rows",
                                likely_cause=(
                                    "Credit_History_Age exceeds the applicant's possible "
                                    "credit tenure. Likely a unit confusion, or Age was collapsed."
                                ),
                            )

    base_cols_with_missing: dict[str, list[str]] = {}
    for c in feature_frame.columns:
        if c.endswith("_missing"):
            base = c[: -len("_missing")]
            base_cols_with_missing[base] = []
    for base in base_cols_with_missing:
        siblings = [
            c for c in feature_frame.columns
            if c.startswith(f"{base}_") and c != f"{base}_missing"
        ]
        base_cols_with_missing[base] = siblings

    for base, siblings in base_cols_with_missing.items():
        miss_col = f"{base}_missing"
        miss_series = feature_frame[miss_col]
        if not pd.api.types.is_numeric_dtype(miss_series):
            continue
        missing_mask = miss_series.fillna(0).astype(float) == 1
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            continue

        not_specified = [s for s in siblings if s.endswith("_Not Specified") or s.endswith("_Not_Specified")]
        for ns in not_specified:
            record(
                column=ns,
                violation="duplicate_missingness_encoding",
                observed=f"'{ns}' exists alongside '{miss_col}'",
                expected=f"only one representation of missingness — keep '{miss_col}', drop '{ns}'",
                likely_cause=(
                    "Missingness is encoded twice: once via the _missing sentinel and once "
                    "via a 'Not Specified' one-hot column. Drop the one-hot variant."
                ),
            )

        for sib in siblings:
            sib_series = feature_frame[sib]
            if not pd.api.types.is_numeric_dtype(sib_series):
                continue
            collision = ((missing_mask) & (sib_series.fillna(0).astype(float) != 0)).sum()
            if collision > 0:
                record(
                    column=sib,
                    violation="missing_sentinel_collision",
                    observed=f"{int(collision)} rows have '{miss_col}'=1 and '{sib}'!=0",
                    expected=f"when '{miss_col}'=1, all '{base}_*' siblings must be 0",
                    likely_cause=(
                        "Missingness and a category indicator are both set on the same row. "
                        "Zero out sibling indicators when the _missing sentinel fires."
                    ),
                )
                break

    return violations


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
