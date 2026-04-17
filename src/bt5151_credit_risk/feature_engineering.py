import ast
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
        "feature_lineage.json",
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
    deferred_categorical_columns: dict | None = None,
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
    if deferred_categorical_columns:
        payload["deferred_categorical_columns"] = deferred_categorical_columns
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
    deferred_categorical_columns: dict | None = None,
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
    if deferred_categorical_columns:
        payload["deferred_categorical_columns"] = deferred_categorical_columns
    return _call_fe_codegen_agent(system_prompt, payload, caller="repair-feature-engineering-code")


def deterministic_feature_engineering_fallback_code(
    *,
    reason: str | None = None,
    fallback_used: bool = True,
) -> dict:
    """Return conservative FE code that preserves rows and encodes categoricals.

    This is the pipeline seatbelt: if LLM-generated FE code repeatedly crashes,
    continue with a simple deterministic representation rather than failing the
    whole run. It intentionally avoids clever ratios/drops; preprocessing already
    produced the main signal-bearing columns.
    """
    reason = reason or (
        "LLM feature-engineering repair attempts were exhausted; using deterministic "
        "pass-through + categorical encoding."
    )
    code = r'''
import json
import numpy as np
import pandas as pd
from pathlib import Path


def engineer_features(train_df, test_df, workspace_path):
    workspace_path = Path(workspace_path)
    report = {
        "dropped": [],
        "transformed": [],
        "added": [],
        "fallback": {
            "used": __FE_FALLBACK_USED__,
            "reason": __FE_FALLBACK_REASON__
        },
    }
    lineage = {"derived_features": [], "dropped_features": [], "passthrough_features": []}

    train_df = train_df.copy()
    test_df = test_df.copy()
    deferred = globals().get("deferred_categorical_columns", {}) or {}
    object_cols = list(train_df.select_dtypes(exclude=["number", "bool"]).columns)
    for col in object_cols:
        nunique = int(deferred.get(col, train_df[col].nunique(dropna=True)))
        if nunique <= 20:
            train_values = train_df[col].fillna("__MISSING__").astype(str)
            test_values = test_df[col].fillna("__MISSING__").astype(str)
            dummies = pd.get_dummies(train_values, prefix=col, dtype=int)
            test_dummies = pd.get_dummies(test_values, prefix=col, dtype=int)
            test_dummies = test_dummies.reindex(columns=dummies.columns, fill_value=0)
            train_df = pd.concat([train_df.drop(columns=[col]), dummies], axis=1)
            test_df = pd.concat([test_df.drop(columns=[col]), test_dummies], axis=1)
            report["transformed"].append({
                "column": col,
                "transform": "one_hot",
                "rationale": "deterministic fallback encoding for deferred categorical"
            })
            lineage["derived_features"].append({
                "feature": col,
                "operation": "one_hot",
                "inputs": [col],
                "input_stage": "pre_fe_raw_categorical",
            })
        else:
            freq = train_df[col].fillna("__MISSING__").astype(str).value_counts(normalize=True).to_dict()
            train_df[col] = train_df[col].fillna("__MISSING__").astype(str).map(freq).fillna(0.0)
            test_df[col] = test_df[col].fillna("__MISSING__").astype(str).map(freq).fillna(0.0)
            report["transformed"].append({
                "column": col,
                "transform": "frequency_encode",
                "rationale": "deterministic fallback compact encoding for high-cardinality categorical"
            })
            lineage["derived_features"].append({
                "feature": col,
                "operation": "frequency_encode",
                "inputs": [col],
                "input_stage": "pre_fe_raw_categorical",
            })

    # Coerce any remaining bools/numerics safely and fill using train statistics.
    all_columns = list(train_df.columns)
    for col in all_columns:
        if col not in test_df.columns:
            test_df[col] = 0
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)
        else:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        clean_train = train_df[col].replace([np.inf, -np.inf], np.nan)
        median = clean_train.median()
        if pd.isna(median):
            median = 0.0
        train_df[col] = clean_train.fillna(float(median))
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan).fillna(float(median))

    # Align test to train exactly; drop unexpected test-only columns.
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    one_hot_parents = {
        entry["feature"]
        for entry in lineage["derived_features"]
        if entry.get("operation") == "one_hot"
    }
    derived_exact = {
        entry["feature"]
        for entry in lineage["derived_features"]
        if entry.get("operation") != "one_hot"
    }
    passthrough = []
    for col in train_df.columns:
        if col in derived_exact:
            continue
        if any(col.startswith(f"{parent}_") for parent in one_hot_parents):
            continue
        passthrough.append(col)
    lineage["passthrough_features"] = passthrough

    non_num_train = train_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    non_num_test = test_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    assert not non_num_train, f"Non-numeric columns in train: {non_num_train}"
    assert not non_num_test, f"Non-numeric columns in test: {non_num_test}"

    train_df.to_csv(workspace_path / "engineered_train.csv", index=False)
    test_df.to_csv(workspace_path / "engineered_test.csv", index=False)
    (workspace_path / "feature_engineering_report.json").write_text(json.dumps(report, indent=2))
    (workspace_path / "feature_lineage.json").write_text(json.dumps(lineage, indent=2))
'''
    code = code.replace("__FE_FALLBACK_USED__", repr(bool(fallback_used)))
    code = code.replace("__FE_FALLBACK_REASON__", json.dumps(reason))
    return {
        "code": code,
        "entrypoint": "engineer_features",
        "hypothesis": {
            "interactions_rationale": (
                "Deterministic feature engineering: no new interactions are added; the "
                "priority is to preserve validated preprocessing features and unblock "
                "model training."
            ),
            "dropped_features_rationale": "No features are dropped by the fallback.",
            "expected_impact": (
                "Expected to be a stable baseline, possibly lower-performing than successful "
                "LLM FE, but preferable to failing the end-to-end pipeline."
            ),
        },
    }


# Run feature engineering code in a subprocess, writing output CSVs.
def execute_feature_engineering(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    generated_code: dict,
    run_root,
    deferred_categorical_columns: dict | None = None,
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
    deferred_path = workspace_path / "deferred_categorical_columns.json"

    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    code_path.write_text(generated_code.get("code", ""), encoding="utf-8")
    deferred_path.write_text(json.dumps(deferred_categorical_columns or {}), encoding="utf-8")

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
                "    train_path = Path(sys.argv[2])",
                "    test_path = Path(sys.argv[3])",
                "    workspace_path = Path(sys.argv[4])",
                "    deferred_path = Path(sys.argv[5])",
                f"    entrypoint_name = {entrypoint_name!r}",
                '    spec = importlib.util.spec_from_file_location("generated_feature_engineering", code_path)',
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    module.deferred_categorical_columns = json.loads(deferred_path.read_text(encoding='utf-8'))",
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
            [
                sys.executable,
                str(runner_path),
                str(code_path),
                str(train_path),
                str(test_path),
                str(workspace_path),
                str(deferred_path),
            ],
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


# Allowed enumerations for lineage manifest — rejected if anything else appears.
_LINEAGE_ALLOWED_OPERATIONS = {
    "ratio", "product", "sum", "difference", "log1p", "bin", "interaction",
    "passthrough", "one_hot", "frequency_encode", "frequency_encoding",
}
_LINEAGE_ALLOWED_INPUT_STAGES = {
    "pre_fe_raw_numeric",
    "pre_fe_encoded",
    "pre_fe_raw_categorical",
    "fe_derived",
}
_LINEAGE_ALLOWED_DROP_REASONS = {
    "leakage", "deterministic_duplicate", "constant", "correlation_with_higher_mi_feature",
}
# Operations where `input_stage` must be pre-FE raw numeric (no log-transformed parents).
_LINEAGE_RAW_PARENT_OPS = {"ratio", "product", "sum", "difference", "interaction"}
# Operations the replay check actually recomputes row-by-row.
_LINEAGE_REPLAY_OPS = {"ratio", "product", "sum", "difference", "log1p"}


class _FormulaReplayError(ValueError):
    """Raised when a report formula cannot be safely replayed."""


def _feature_formula_map(report: dict | None) -> dict[str, str]:
    """Extract exact engineered-feature formulas from the FE report."""
    if not isinstance(report, dict):
        return {}
    formulas: dict[str, str] = {}
    for section in ("added", "transformed"):
        entries = report.get(section) or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            column = entry.get("column")
            formula = entry.get("formula")
            if isinstance(column, str) and isinstance(formula, str) and formula.strip():
                formulas[column] = formula.strip()
    return formulas


def _subscript_key(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise _FormulaReplayError("only string column subscripts are supported")


def _eval_formula_ast(node: ast.AST, frame: pd.DataFrame):
    """Safely evaluate a small arithmetic expression over DataFrame columns.

    Supported formulas intentionally cover the FE contract language only:
    column names, df["column"] lookups, numeric constants, +, -, *, /, unary
    signs, and log1p. No arbitrary attribute access or function calls.
    """
    if isinstance(node, ast.Expression):
        return _eval_formula_ast(node.body, frame)

    if isinstance(node, ast.Name):
        if node.id in frame.columns:
            return pd.to_numeric(frame[node.id], errors="coerce")
        raise _FormulaReplayError(f"unknown formula symbol {node.id!r}")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise _FormulaReplayError(f"unsupported constant {node.value!r}")

    if isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name) or node.value.id not in {
            "df", "data", "frame", "train_df", "base_train",
        }:
            raise _FormulaReplayError("only df['column']-style subscripts are supported")
        key = _subscript_key(node.slice)
        if key not in frame.columns:
            raise _FormulaReplayError(f"formula references missing column {key!r}")
        return pd.to_numeric(frame[key], errors="coerce")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_formula_ast(node.operand, frame)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise _FormulaReplayError("unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_formula_ast(node.left, frame)
        right = _eval_formula_ast(node.right, frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise _FormulaReplayError("unsupported binary operator")

    if isinstance(node, ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
        ):
            func_name = node.func.attr
        if func_name != "log1p" or len(node.args) != 1 or node.keywords:
            raise _FormulaReplayError("only log1p(x) formulas are supported")
        arg = _eval_formula_ast(node.args[0], frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log1p(arg)

    raise _FormulaReplayError(f"unsupported formula syntax {type(node).__name__}")


def _evaluate_feature_formula(formula: str, frame: pd.DataFrame) -> pd.Series:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise _FormulaReplayError(f"invalid formula syntax: {exc}") from exc
    value = _eval_formula_ast(tree, frame)
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").reset_index(drop=True)
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise _FormulaReplayError("formula did not produce numeric values") from exc
    return pd.Series([scalar] * len(frame))


def _lineage_expected_candidates(expected: pd.Series, entry: dict) -> list[pd.Series]:
    """Build acceptable replay candidates for formulas with documented cleanup.

    Generated FE often records a mathematically exact formula in
    feature_engineering_report.json, then applies the lineage manifest's
    fill/clip strategy for stability. The validator accepts the raw formula
    and the documented stabilized variants, but still rejects unrelated math
    such as log-transformed parents.
    """
    numeric = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan)
    candidates = [numeric]

    fill_strategy = str(entry.get("fill_strategy") or "").lower()
    clip_strategy = str(entry.get("clip_strategy") or "").lower()
    if fill_strategy or clip_strategy:
        median = numeric.median()
        if pd.isna(median):
            median = 0.0
        filled = numeric.fillna(float(median))
        candidates.append(filled)

        if "p99" in clip_strategy or "quantile" in clip_strategy or "winsor" in clip_strategy:
            upper = filled.quantile(0.99)
            if not pd.isna(upper):
                candidates.append(filled.clip(upper=float(upper)))
            lower = filled.quantile(0.01)
            if not pd.isna(lower) and not pd.isna(upper):
                candidates.append(filled.clip(lower=float(lower), upper=float(upper)))

    return candidates


def _values_close(expected: float, actual: float) -> bool:
    if pd.isna(expected) or pd.isna(actual):
        return False
    return abs(expected - actual) <= max(1e-4, 1e-4 * abs(expected))


def _apply_lineage_operation(operation: str, input_values: list[float]) -> float | None:
    """Recompute a derived feature value from its raw inputs. Returns None if
    the operation is not deterministically replayable (e.g. bin, interaction
    without a declared operator)."""
    if operation == "ratio":
        if len(input_values) != 2:
            return None
        a, b = input_values
        if b == 0 or pd.isna(b):
            return None  # fill_strategy makes replay ambiguous
        return a / b
    if operation == "product":
        out = 1.0
        for v in input_values:
            out *= v
        return out
    if operation == "sum":
        return float(sum(input_values))
    if operation == "difference":
        if len(input_values) != 2:
            return None
        return input_values[0] - input_values[1]
    if operation == "log1p":
        if len(input_values) != 1:
            return None
        v = input_values[0]
        if v < 0 or pd.isna(v):
            return None
        return float(np.log1p(v))
    return None


def validate_feature_lineage(
    lineage: dict,
    train_frame_pre_fe: pd.DataFrame | None,
    engineered_train: pd.DataFrame,
    top_mi_features: list[str] | None,
    feature_formulas: dict[str, str] | None = None,
) -> list[dict]:
    """Replay the declared lineage against the actual engineered train frame.

    Returns a list of violation dicts. Empty means lineage is consistent.
    """
    violations: list[dict] = []

    def record(rule: str, message: str, **extra) -> None:
        violations.append({"rule": rule, "message": message, **extra})

    if not isinstance(lineage, dict):
        record("lineage_artifact_present", "feature_lineage.json is not a valid JSON object.")
        return violations

    derived = lineage.get("derived_features") or []
    dropped = lineage.get("dropped_features") or []
    passthrough = set(lineage.get("passthrough_features") or [])

    if not isinstance(derived, list):
        record("lineage_schema", "`derived_features` must be a list.")
        return violations

    # --- Enum validation ---
    derived_names: set[str] = set()
    one_hot_parents: set[str] = set()
    for i, entry in enumerate(derived):
        if not isinstance(entry, dict):
            record("lineage_schema", f"derived_features[{i}] is not an object.")
            continue
        feature = entry.get("feature")
        op = entry.get("operation")
        input_stage = entry.get("input_stage")
        inputs = entry.get("inputs")
        if not feature or not isinstance(feature, str):
            record("lineage_schema", f"derived_features[{i}] missing `feature`.")
            continue
        if op not in _LINEAGE_ALLOWED_OPERATIONS:
            record(
                "lineage_operation_enum",
                f"derived_features[{feature!r}] has invalid operation {op!r}. "
                f"Allowed: {sorted(_LINEAGE_ALLOWED_OPERATIONS)}.",
                feature=feature,
            )
            continue
        if input_stage not in _LINEAGE_ALLOWED_INPUT_STAGES:
            record(
                "lineage_input_stage_enum",
                f"derived_features[{feature!r}] has invalid input_stage {input_stage!r}. "
                f"Allowed: {sorted(_LINEAGE_ALLOWED_INPUT_STAGES)}.",
                feature=feature,
            )
            continue
        if op != "passthrough" and (not isinstance(inputs, list) or not inputs):
            record(
                "lineage_inputs_missing",
                f"derived_features[{feature!r}] (operation={op}) has empty inputs.",
                feature=feature,
            )
            continue

        # Raw-parent rule: ratios/products/sums/differences/interactions must use raw parents.
        if op in _LINEAGE_RAW_PARENT_OPS and input_stage != "pre_fe_raw_numeric":
            record(
                "ratios_use_raw_parents",
                f"derived_features[{feature!r}] (operation={op}) has "
                f"input_stage={input_stage!r}. Rule: ratios/products/sums/differences/"
                f"interactions must be built from pre_fe_raw_numeric inputs — no log-transformed "
                f"or otherwise pre-transformed parents allowed.",
                feature=feature,
            )

        derived_names.add(feature)
        if op == "one_hot":
            one_hot_parents.add(feature)

    # --- Dropped features enum + top-MI drop rule ---
    for i, entry in enumerate(dropped):
        if not isinstance(entry, dict):
            record("lineage_schema", f"dropped_features[{i}] is not an object.")
            continue
        feature = entry.get("feature")
        reason = entry.get("drop_reason")
        if reason not in _LINEAGE_ALLOWED_DROP_REASONS:
            record(
                "lineage_drop_reason_enum",
                f"dropped_features[{feature!r}] has invalid drop_reason {reason!r}. "
                f"Allowed: {sorted(_LINEAGE_ALLOWED_DROP_REASONS)}.",
                feature=feature,
            )
            continue
        if top_mi_features and feature in set(top_mi_features):
            if reason not in {"leakage", "deterministic_duplicate"}:
                record(
                    "top_mi_drop_requires_justification",
                    f"Top-MI feature {feature!r} was dropped with reason {reason!r}. "
                    f"Rule: top-5 MI raw features can only be dropped by leakage or "
                    f"deterministic_duplicate — correlation heuristic is not sufficient. "
                    f"Restore the feature or escalate the drop_reason.",
                    feature=feature,
                )

    # --- Coverage: every engineered column must be accounted for ---
    output_cols = set(engineered_train.columns)
    unaccounted = []
    for col in output_cols:
        if col in derived_names or col in passthrough:
            continue
        # Accept one-hot expansions: `<parent>_<value>` where parent has operation=one_hot
        # OR parent is a passthrough categorical in the input frame.
        parent_match = False
        for parent in one_hot_parents | passthrough:
            if col.startswith(f"{parent}_"):
                parent_match = True
                break
        if parent_match:
            continue
        unaccounted.append(col)

    if unaccounted:
        # Cap the list in the error message.
        sample = unaccounted[:15]
        record(
            "lineage_coverage_complete",
            f"{len(unaccounted)} engineered column(s) have no lineage entry: {sample}. "
            f"Every output column must appear in `derived_features`, `passthrough_features`, "
            f"or be an expansion of a `one_hot` parent.",
            unaccounted_sample=sample,
            unaccounted_count=len(unaccounted),
        )

    # --- Replay check: sample 20 rows, recompute arithmetic ops ---
    feature_formulas = feature_formulas or {}
    if train_frame_pre_fe is not None and len(engineered_train) > 0:
        n_sample = min(20, len(engineered_train))
        # Align indices: engineered_train and train_frame_pre_fe must share
        # positional order (same rows in same order). We read by position.
        common_len = min(len(train_frame_pre_fe), len(engineered_train))
        if common_len < n_sample:
            n_sample = common_len
        if n_sample > 0:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(common_len, size=n_sample, replace=False)
            for entry in derived:
                if not isinstance(entry, dict):
                    continue
                feature = entry.get("feature")
                op = entry.get("operation")
                inputs = entry.get("inputs")
                input_stage = entry.get("input_stage")
                if op not in _LINEAGE_REPLAY_OPS:
                    continue
                if input_stage != "pre_fe_raw_numeric":
                    continue
                if feature not in output_cols:
                    continue
                if not isinstance(inputs, list):
                    continue

                formula = feature_formulas.get(str(feature))
                if formula:
                    try:
                        expected_series = _evaluate_feature_formula(formula, train_frame_pre_fe)
                    except _FormulaReplayError as exc:
                        record(
                            "lineage_formula_replayable",
                            f"Replay of {feature!r} could not parse/evaluate report formula "
                            f"{formula!r}: {exc}. Use simple arithmetic over pre-FE column names.",
                            feature=feature,
                            formula=formula,
                        )
                        continue

                    candidates = _lineage_expected_candidates(expected_series, entry)
                    mismatches = []
                    for pos in sample_idx:
                        try:
                            actual = float(engineered_train.iloc[int(pos)][feature])
                        except (TypeError, ValueError, KeyError):
                            continue
                        if pd.isna(actual):
                            continue
                        expected_values = [
                            float(candidate.iloc[int(pos)])
                            for candidate in candidates
                            if int(pos) < len(candidate)
                        ]
                        if not expected_values:
                            continue
                        if any(_values_close(expected, actual) for expected in expected_values):
                            continue
                        mismatches.append({
                            "row_pos": int(pos),
                            "expected": round(float(expected_values[0]), 6),
                            "actual": round(float(actual), 6),
                        })
                    if mismatches:
                        record(
                            "lineage_replay_matches",
                            f"Replay of {feature!r} using report formula {formula!r} disagrees "
                            f"with the engineered output on {len(mismatches)}/{n_sample} "
                            f"sampled rows. The code is likely computing something other than "
                            f"the declared formula, or applying an undeclared transform.",
                            feature=feature,
                            operation=op,
                            formula=formula,
                            mismatch_sample=mismatches[:3],
                        )
                    continue

                # All parents must exist in the pre-FE frame.
                if any(parent not in train_frame_pre_fe.columns for parent in inputs):
                    continue
                mismatches = []
                for pos in sample_idx:
                    try:
                        parent_vals = [
                            float(train_frame_pre_fe.iloc[int(pos)][p]) for p in inputs
                        ]
                    except (TypeError, ValueError):
                        continue
                    expected = _apply_lineage_operation(op, parent_vals)
                    if expected is None:
                        continue
                    try:
                        actual = float(engineered_train.iloc[int(pos)][feature])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if pd.isna(actual):
                        continue
                    if not _values_close(float(expected), actual):
                        mismatches.append({
                            "row_pos": int(pos),
                            "expected": round(float(expected), 6),
                            "actual": round(float(actual), 6),
                            "parents": {p: round(v, 6) for p, v in zip(inputs, parent_vals)},
                        })
                if mismatches:
                    record(
                        "lineage_replay_matches",
                        f"Replay of {feature!r} (operation={op}) disagrees with the "
                        f"declared lineage on {len(mismatches)}/{n_sample} sampled rows. "
                        f"The code is likely computing something other than what the lineage claims. "
                        f"Check for log-transformed parents, wrong order of inputs, or different fill strategies.",
                        feature=feature,
                        operation=op,
                        mismatch_sample=mismatches[:3],
                    )

    return violations


# Validate feature engineering output: row counts, NaNs, feature count.
def validate_feature_engineering_output(
    execution_result: dict,
    original_train_rows: int,
    original_test_rows: int,
    original_feature_count: int,
    deferred_categoricals: dict | None = None,
    top_mi_features: list[str] | None = None,
    train_frame_pre_fe: pd.DataFrame | None = None,
) -> dict:
    """
    deferred_categoricals: {col_name: nunique} for object-dtype columns in the preprocessed
    feature frame. Used to detect frequency-encoding of low-cardinality identity columns in
    tree_view (which collapses category identity and must be rejected).
    """
    artifacts = execution_result.get("artifacts", {})
    view_metadata = execution_result.get("view_metadata") or _load_view_metadata(artifacts)
    checks: dict[str, bool] = {}
    errors: list[dict] = []
    views: dict[str, dict] = {}

    def add_check(rule: str, passed: bool, message: str) -> None:
        checks[rule] = passed
        if not passed:
            errors.append({"rule": rule, "message": message})

    def check_top_mi_not_dropped(output_columns: list[str], view_label: str = "") -> None:
        """Top-5 MI raw features must survive FE — exact name or as a column prefix.

        A top-MI feature is considered present if:
        - It appears exactly as-is, OR
        - At least one output column starts with '<feature>_' (e.g. log-transform, bin).
        This protects against the LLM dropping high-signal raw features in favor of
        lower-MI proxies when resolving high-correlation pairs (|r| > 0.95).
        """
        if not top_mi_features:
            return
        out_set = set(output_columns)
        label = f"[{view_label}] " if view_label else ""
        for feat in top_mi_features:
            present = feat in out_set or any(c == feat or c.startswith(f"{feat}_") for c in out_set)
            add_check(
                f"top_mi_feature_preserved_{feat.lower().replace(' ', '_')[:40]}",
                present,
                f"{label}Top-MI feature '{feat}' is absent from the output. "
                f"Rule: when |r| > 0.95, keep the higher-MI feature, not the higher-variance "
                f"feature. Never drop a top-5 MI raw feature in favor of a lower-MI proxy "
                f"unless it is a deterministic duplicate or leakage feature. "
                f"Restore '{feat}' or an explicit derived form (e.g. '{feat}_log', '{feat}_bin').",
            )

    report_path = Path(artifacts.get("feature_engineering_report.json", ""))
    feature_formulas: dict[str, str] = {}

    add_check("report_file_exists", report_path.is_file(), "feature_engineering_report.json not found.")
    if report_path.is_file():
        try:
            feature_formulas = _feature_formula_map(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
        except Exception:
            feature_formulas = {}

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
                train_non_numeric = list(train_df.select_dtypes(exclude=["number", "bool"]).columns)
                add_check(
                    f"{view_name}_train_fully_numeric",
                    not train_non_numeric,
                    f"{train_artifact} has non-numeric/non-bool columns {train_non_numeric[:10]} — FE must encode deferred categoricals per view.",
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
                test_non_numeric = list(test_df.select_dtypes(exclude=["number", "bool"]).columns)
                add_check(
                    f"{view_name}_test_fully_numeric",
                    not test_non_numeric,
                    f"{test_artifact} has non-numeric/non-bool columns {test_non_numeric[:10]} — FE must encode deferred categoricals per view.",
                )

            if train_df is not None and test_df is not None:
                add_check(
                    f"{view_name}_column_alignment",
                    list(train_df.columns) == list(test_df.columns),
                    f"{view_name} train and test columns do not match after feature engineering.",
                )

                # Top-MI protection: top-5 MI raw features must not be silently dropped.
                check_top_mi_not_dropped(list(train_df.columns), view_label=view_name)

                # Identity-preservation check for tree_view: low-cardinality deferred
                # categoricals (≤20 unique values) must NOT appear as a single float column
                # (frequency encoding signature). They must be one-hot expanded into multiple
                # {col}_{value} columns so tree models can split on category identity.
                if view_name == "tree_view" and deferred_categoricals:
                    output_cols = set(train_df.columns)
                    for col, n_unique in deferred_categoricals.items():
                        if n_unique > 20:
                            continue  # higher cardinality — frequency encoding acceptable
                        if col in output_cols:
                            # Single column with the original name = frequency/ordinal encoding
                            add_check(
                                f"tree_view_identity_{col}",
                                False,
                                f"{col} has {n_unique} unique categories (≤20) but appears as a "
                                f"single numeric column in tree_view — this is frequency encoding "
                                f"which collapses category identity. Two categories with the same "
                                f"prevalence become numerically identical even if their risk profiles "
                                f"differ. One-hot encode it instead (produces {col}_<value> columns). "
                                f"Fix: replace frequency encoding with pd.get_dummies(train_df['{col}'], "
                                f"prefix='{col}') for both views.",
                            )
                        else:
                            checks[f"tree_view_identity_{col}"] = True

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
            train_non_numeric = list(train_df.select_dtypes(exclude=["number", "bool"]).columns)
            add_check(
                "train_fully_numeric",
                not train_non_numeric,
                f"engineered_train.csv has non-numeric/non-bool columns {train_non_numeric[:10]} — FE must encode deferred categoricals.",
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
            test_non_numeric = list(test_df.select_dtypes(exclude=["number", "bool"]).columns)
            add_check(
                "test_fully_numeric",
                not test_non_numeric,
                f"engineered_test.csv has non-numeric/non-bool columns {test_non_numeric[:10]} — FE must encode deferred categoricals.",
            )

        if train_df is not None and test_df is not None:
            add_check(
                "column_alignment",
                list(train_df.columns) == list(test_df.columns),
                "Train and test columns do not match after feature engineering.",
            )
            # Top-MI protection: top-5 MI raw features must not be silently dropped.
            check_top_mi_not_dropped(list(train_df.columns))
            views["default"] = {
                "train_artifact": "engineered_train.csv",
                "test_artifact": "engineered_test.csv",
                "feature_count": len(train_df.columns),
            }

    # --- Feature lineage validation ---
    lineage_violations: list[dict] = []
    lineage_path = Path(artifacts.get("feature_lineage.json", ""))
    lineage: dict | None = None
    if lineage_path.is_file():
        try:
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        except Exception as exc:
            lineage_violations.append({
                "rule": "lineage_artifact_present",
                "message": f"feature_lineage.json could not be parsed: {exc}",
            })
    else:
        lineage_violations.append({
            "rule": "lineage_artifact_present",
            "message": (
                f"feature_lineage.json not found in workspace. Every FE run must "
                f"emit a lineage manifest describing how each engineered column "
                f"was produced from pre-FE inputs. See skills/generate-feature-"
                f"engineering-code.md for the required schema."
            ),
        })

    # Pick the canonical engineered frame for replay (first view, or default)
    canonical_train: pd.DataFrame | None = None
    for view_name, view_spec in views.items():
        train_artifact = view_spec.get("train_artifact")
        train_path = Path(artifacts.get(train_artifact, ""))
        if train_path.is_file():
            try:
                canonical_train = pd.read_csv(train_path)
                break
            except Exception:
                continue

    if lineage is not None and canonical_train is not None:
        try:
            extra = validate_feature_lineage(
                lineage,
                train_frame_pre_fe,
                canonical_train,
                top_mi_features,
                feature_formulas=feature_formulas,
            )
            lineage_violations.extend(extra)
        except Exception as exc:  # pragma: no cover — defensive
            lineage_violations.append({
                "rule": "lineage_validator_error",
                "message": f"lineage validator raised {type(exc).__name__}: {exc}",
            })

    passed = all(checks.values()) and not errors and not lineage_violations

    # --- Persist feature contract report artifact ---
    # Use the execution workspace if discoverable; otherwise skip silently.
    workspace_path: Path | None = None
    for artifact_name in ("engineered_train.csv", "engineered_train_tree.csv",
                          "engineered_train_linear.csv", "feature_engineering_report.json"):
        p = Path(artifacts.get(artifact_name, ""))
        if p.is_file():
            workspace_path = p.parent
            break
    if workspace_path is not None:
        try:
            contract_report = {
                "workspace": str(workspace_path),
                "passed": passed,
                "structural_checks": checks,
                "errors": errors,
                "lineage_violations": lineage_violations,
                "view_mode": "dual" if view_metadata else "single",
                "available_views": list(views.keys()),
            }
            (workspace_path / "feature_contract_report.json").write_text(
                json.dumps(contract_report, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover
            pass

    return {
        "passed": passed,
        "checks": checks,
        "errors": errors,
        "lineage_violations": lineage_violations,
        "view_mode": "dual" if view_metadata else "single",
        "available_views": list(views.keys()),
        "views": views,
    }
