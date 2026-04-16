import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from bt5151_credit_risk.business import explain_risk
from bt5151_credit_risk.eda import build_eda_report
from bt5151_credit_risk.evaluate import choose_best_model, compute_multiclass_metrics, reason_model_selection
from bt5151_credit_risk.hypotheses import (
    generate_eda_hypotheses,
    generate_training_diagnostics,
    interpret_global_xai,
    interpret_local_xai,
)
from bt5151_credit_risk.xai import (
    compute_global_shap,
    compute_permutation_importance,
    compute_partial_dependence,
    compute_ale,
    compute_shap_contributions_for_case,
    annotate_pdp_position,
    select_classification_cases,
)
from bt5151_credit_risk.feature_engineering import (
    execute_feature_engineering,
    generate_feature_engineering_code,
    repair_feature_engineering_code,
    validate_feature_engineering_output,
)
from bt5151_credit_risk.preprocess import cleanup_old_workspaces
from bt5151_credit_risk.shortcut_audit import run_shortcut_audit
from bt5151_credit_risk.preprocess import execute_generated_preprocessing
from bt5151_credit_risk.preprocess import generate_column_transform_spec
from bt5151_credit_risk.preprocess import generate_dataset_policy_spec
from bt5151_credit_risk.preprocess import generate_preprocessing_code
from bt5151_credit_risk.preprocess import inspect_preprocessing_code
from bt5151_credit_risk.preprocess import repair_preprocessing_code
from bt5151_credit_risk.preprocess import review_preprocessing_quality
from bt5151_credit_risk.preprocess import validate_preprocessing_output
from bt5151_credit_risk.profile import build_dataset_profile
from bt5151_credit_risk.state import CreditRiskState
from bt5151_credit_risk.train import build_candidate_models, extract_learning_curves, reason_hyperparameter_grids, tune_models


MAX_REPAIR_ATTEMPTS = 5
MAX_FE_REPAIR_ATTEMPTS = 3


def _default_feature_view_name(view_dict):
    if not view_dict:
        return None
    if "tree_view" in view_dict:
        return "tree_view"
    if "default" in view_dict:
        return "default"
    return next(iter(view_dict))


def _default_model_view_map(view_dict):
    if not view_dict:
        return {}
    default_view = _default_feature_view_name(view_dict)
    mapping = {
        "logistic_regression": "linear_view" if "linear_view" in view_dict else default_view,
        "random_forest": "tree_view" if "tree_view" in view_dict else default_view,
        "xgboost": "tree_view" if "tree_view" in view_dict else default_view,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _get_model_view_name(state, model_name):
    explicit_map = getattr(state, "model_view_map", None) or {}
    if model_name in explicit_map:
        return explicit_map[model_name]
    available_views = getattr(state, "train_views", None) or {}
    return _default_model_view_map(available_views).get(model_name)


def _get_train_frame_for_model(state, model_name):
    train_views = getattr(state, "train_views", None) or {}
    view_name = _get_model_view_name(state, model_name)
    if view_name and view_name in train_views:
        return train_views[view_name]
    return state.train_frame


def _get_test_frame_for_model(state, model_name):
    test_views = getattr(state, "test_views", None) or {}
    view_name = _get_model_view_name(state, model_name)
    if view_name and view_name in test_views:
        return test_views[view_name]
    return state.test_frame


def _get_full_frame_for_model(state, model_name):
    full_views = getattr(state, "full_feature_frames_by_view", None) or {}
    view_name = _get_model_view_name(state, model_name)
    if view_name and view_name in full_views:
        return full_views[view_name]
    return state.full_feature_frame


def _get_feature_columns_for_model(state, model_name):
    columns_by_view = getattr(state, "feature_columns_by_view", None) or {}
    view_name = _get_model_view_name(state, model_name)
    if view_name and view_name in columns_by_view:
        return columns_by_view[view_name]
    return state.feature_columns


# Load the raw dataset once and get the high-level preprocessing policy.
def dataset_policy_spec_node(state: CreditRiskState):
    logger.info(">>> dataset-policy-spec: loading %s", state.raw_dataset_path)
    raw_frame = pd.read_csv(state.raw_dataset_path, low_memory=False)
    logger.info("    loaded %d rows x %d columns", len(raw_frame), len(raw_frame.columns))
    dataset_profile = build_dataset_profile(raw_frame)
    dataset_policy_spec = generate_dataset_policy_spec(raw_frame, dataset_profile)
    logger.info("    policy spec keys: %s", list(dataset_policy_spec.keys()))
    target_column = dataset_policy_spec.get("target_column")
    if target_column:
        dataset_profile = build_dataset_profile(raw_frame, target_column=target_column)
    logger.info("    profile: %s", json.dumps({k: v for k, v in dataset_profile.items() if k != "missing_counts"}, default=str))

    return {
        "raw_frame": raw_frame,
        "dataset_profile": dataset_profile,
        "dataset_policy_spec": dataset_policy_spec,
    }


# Programmatic EDA: correlations, class separability, skewness, missing patterns, cardinality.
def exploratory_data_analysis_node(state: CreditRiskState):
    logger.info(">>> exploratory-data-analysis")
    target_column = (state.dataset_policy_spec or {}).get("target_column")
    if not target_column:
        logger.warning("    no target column in policy spec, skipping EDA")
        return {"eda_report": {}}

    eda_report = build_eda_report(state.raw_frame, target_column)

    # Log key findings
    high_pairs = eda_report.get("correlations", {}).get("high_pairs", [])
    if high_pairs:
        logger.info("    high-correlation pairs (|r| > 0.8): %d found", len(high_pairs))
        for pair in high_pairs[:5]:
            logger.info("      %s ↔ %s: r=%.4f", pair["col_a"], pair["col_b"], pair["correlation"])

    top_features = eda_report.get("top_discriminative_features", [])
    if top_features:
        logger.info("    top discriminative features (mutual information):")
        for feat in top_features[:10]:
            logger.info("      %s: MI=%.4f", feat["column"], feat["mutual_information"])

    skewed = eda_report.get("skewness", {}).get("highly_skewed", {})
    if skewed:
        logger.info("    highly skewed columns (|skew| > 2): %d", len(skewed))

    high_card = eda_report.get("cardinality", {}).get("high_cardinality", [])
    if high_card:
        logger.info("    high-cardinality categoricals (>20 unique):")
        for entry in high_card:
            logger.info("      %s: %d unique", entry["column"], entry["nunique"])

    mnar = eda_report.get("missing_patterns", {}).get("mnar_suspects", [])
    if mnar:
        logger.info("    MNAR suspects (missingness correlates with target): %d", len(mnar))
        for entry in mnar:
            logger.info("      %s: %.1f%% missing, target dist diff=%.4f",
                        entry["column"], entry["missing_pct"] * 100, entry["max_target_dist_diff"])

    anova_top = eda_report.get("class_separability", {}).get("anova_top_features", [])
    if anova_top:
        logger.info("    ANOVA top-5 features:")
        for feat in anova_top[:5]:
            logger.info("      %s: F=%.2f p=%.6f", feat["column"], feat["f_statistic"], feat["p_value"])

    return {"eda_report": eda_report}


# LLM interprets EDA statistics into bold, directional, three-tier hypotheses.
def generate_eda_hypotheses_node(state: CreditRiskState):
    logger.info(">>> generate-eda-hypotheses")
    if not state.eda_report:
        logger.warning("    no EDA report available, skipping hypothesis generation")
        return {"eda_hypotheses": None}

    try:
        hypotheses = generate_eda_hypotheses(state.eda_report, state.dataset_profile)
    except Exception as exc:
        logger.warning("    EDA hypothesis generation failed (%s), continuing without hypotheses", exc)
        return {"eda_hypotheses": None}

    tested = hypotheses.get("tested_predictions", [])
    supported = hypotheses.get("supported_conjectures", [])
    exploratory = hypotheses.get("exploratory_leads", [])
    logger.info("    tested predictions: %d", len(tested))
    for h in tested:
        logger.info("      [tested] %s", h.get("hypothesis", "")[:150])
    logger.info("    supported conjectures: %d", len(supported))
    for h in supported:
        logger.info("      [supported] %s", h.get("hypothesis", "")[:150])
    logger.info("    exploratory leads: %d", len(exploratory))
    for h in exploratory:
        logger.info("      [exploratory] %s", h.get("hypothesis", "")[:150])
    logger.info("    model selection prediction: %s", hypotheses.get("model_selection_prediction", ""))
    logger.info("    class struggle prediction: %s", hypotheses.get("class_struggle_prediction", ""))

    return {"eda_hypotheses": hypotheses}


# Ask for the detailed column-level transformation rules.
def column_transform_spec_node(state: CreditRiskState):
    logger.info(">>> column-transform-spec")
    column_transform_spec = generate_column_transform_spec(
        state.raw_frame,
        state.dataset_policy_spec,
        eda_report=state.eda_report,
    )
    logger.info("    transform spec keys: %s", list(column_transform_spec.keys()))
    transforms = column_transform_spec.get("transforms", {})
    for col_name, spec in transforms.items():
        action = spec.get("action", "?")
        cleaning = spec.get("cleaning") or ""
        encoding = spec.get("encoding") or ""
        detail = ", ".join(filter(None, [cleaning, encoding]))
        logger.info("    col %-25s action=%-5s %s", col_name, action, detail)
    reasoning = column_transform_spec.get("reasoning", {})
    if reasoning:
        logger.info("    column-transform reasoning:")
        for col_name, reason in reasoning.items():
            logger.info("      %s: %s", col_name, reason[:120])
    return {"column_transform_spec": column_transform_spec}


# Generate preprocessing code from the two specs.
def generate_preprocessing_code_node(state: CreditRiskState):
    logger.info(">>> generate-preprocessing-code")
    generated_code = generate_preprocessing_code(
        state.raw_frame,
        state.dataset_profile,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    logger.info("    entrypoint: %s, code length: %d chars", generated_code.get("entrypoint"), len(generated_code.get("code", "")))
    return {
        "preprocessing_code": generated_code,
        "preprocessing_codegen_metadata": {
            "entrypoint": generated_code.get("entrypoint"),
        },
        "preprocessing_attempt_count": 1,
    }


# Inspect generated code before we try to run it.
def inspect_preprocessing_code_node(state: CreditRiskState):
    logger.info(">>> inspect-preprocessing-code (attempt %d/%d)", state.preprocessing_attempt_count or 0, MAX_REPAIR_ATTEMPTS)
    code_review = inspect_preprocessing_code(state.preprocessing_code or {})
    logger.info("    inspection passed: %s, issues: %d", code_review.get("passed"), len(code_review.get("issues", [])))
    if not code_review.get("passed"):
        for issue in code_review.get("issues", []):
            logger.warning("    inspection issue: [%s] %s", issue.get("rule"), issue.get("message"))
    return {
        "preprocessing_code_review": code_review,
    }


# Execute the generated preprocessing code and save its artifact metadata.
def execute_generated_preprocessing_node(state: CreditRiskState):
    logger.info(">>> execute-generated-preprocessing")
    run_root = Path(state.raw_dataset_path).resolve().parent / "generated_preprocessing_runs"
    execution_result = execute_generated_preprocessing(
        state.raw_frame,
        state.preprocessing_code,
        run_root,
    )
    exec_log = execution_result.get("execution_log", {})
    logger.info("    returncode: %s, timed_out: %s, success: %s",
                exec_log.get("returncode"), exec_log.get("timed_out"), execution_result.get("success"))
    if exec_log.get("stderr"):
        logger.warning("    subprocess stderr:\n%s", exec_log["stderr"][:2000])
    missing = execution_result.get("missing_artifacts", [])
    if missing:
        logger.warning("    missing artifacts: %s", missing)
    return {
        "preprocessing_workspace": execution_result["workspace_path"],
        "preprocessing_raw_frame_path": execution_result["raw_frame_path"],
        "preprocessing_artifacts": execution_result["artifacts"],
        "preprocessing_execution_log": execution_result["execution_log"],
        "preprocessing_execution_report": execution_result,
    }


# Validate artifacts, then rebuild train/test data for deterministic modeling.
def validate_preprocessing_output_node(state: CreditRiskState):
    logger.info(">>> validate-preprocessing-output")
    execution_result = {
        "workspace_path": state.preprocessing_workspace,
        "artifacts": state.preprocessing_artifacts,
        "execution_log": state.preprocessing_execution_log or {},
        "raw_frame_path": state.preprocessing_raw_frame_path,
    }
    validation_report = validate_preprocessing_output(
        execution_result,
        state.dataset_policy_spec,
        state.column_transform_spec,
    )
    logger.info("    validation passed: %s, checks: %s", validation_report.get("passed"), validation_report.get("checks"))
    if validation_report.get("errors"):
        for err in validation_report["errors"]:
            logger.warning("    validation error: [%s] %s", err.get("rule"), err.get("message"))
    if validation_report.get("warnings"):
        for warn in validation_report["warnings"]:
            logger.warning("    quality warning: [%s] %s", warn.get("rule"), warn.get("message"))
    if validation_report.get("role_violations"):
        for v in validation_report["role_violations"]:
            logger.warning("    role violation: [%s] %s (role=%s) — %s",
                           v.get("violation"), v.get("column"), v.get("declared_role"),
                           v.get("likely_cause"))
    if validation_report.get("cross_field_violations"):
        for v in validation_report["cross_field_violations"]:
            logger.warning("    invariant violation: [%s] %s — %s",
                           v.get("violation"), v.get("column"), v.get("likely_cause"))

    # Capability-ceiling escalation: if the same (column, violation) pair appears in
    # this attempt AND the previous one, the repair model is not acting on the contract.
    # Flag escalation so the next repair call routes to a stronger model.
    current_violations = (validation_report.get("role_violations") or []) + (
        validation_report.get("cross_field_violations") or []
    )
    current_sig = {(v.get("column"), v.get("violation")) for v in current_violations}
    previous_violations = state.preprocessing_last_role_violations or []
    previous_sig = {(v.get("column"), v.get("violation")) for v in previous_violations}
    repeat_sig = current_sig & previous_sig
    escalation_triggered = bool(repeat_sig)
    if escalation_triggered:
        logger.warning(
            "    CAPABILITY CEILING: %d role violation(s) repeated across attempts: %s. "
            "Next repair will escalate to stronger model (OPENAI_MODEL_REPAIR_PREPROCESSING_CODE_ESCALATED if set).",
            len(repeat_sig), sorted(repeat_sig),
        )

    if not validation_report["passed"]:
        return {
            "preprocessing_validation_report": validation_report,
            "preprocessing_last_role_violations": current_violations,
            "preprocessing_escalation_triggered": escalation_triggered,
        }

    artifacts = state.preprocessing_artifacts or {}
    feature_frame = pd.read_csv(artifacts["feature_frame.csv"])
    target_frame = pd.read_csv(artifacts["target.csv"])
    target_series = target_frame.iloc[:, 0]

    # --- Deterministic MNAR missing-indicator injection ---
    # The LLM may not add {col}_missing for MNAR multi-value-set columns. Instead of
    # relying on prompt guidance, we inject it here from the multi-hot expansion itself:
    # a row where ALL multi-hot indicators for a column are 0 was originally missing/empty.
    mnar_suspects = (state.eda_report or {}).get("missing_patterns", {}).get("mnar_suspects", [])
    _injected_mnar: list[str] = []
    for entry in mnar_suspects:
        col = entry["column"]
        missing_col = f"{col}_missing"
        if missing_col in feature_frame.columns:
            continue  # already present — LLM handled it
        # Find multi-hot expansion columns for this MNAR column
        prefix = col + "_"
        multihot_cols = [c for c in feature_frame.columns if c.startswith(prefix)]
        if multihot_cols:
            # All-zero row = original value was missing/empty
            feature_frame[missing_col] = (feature_frame[multihot_cols].sum(axis=1) == 0).astype(int)
            _injected_mnar.append(missing_col)
            logger.info(
                "    MNAR inject: added %s (%.1f%% flagged as missing, from %d multi-hot cols)",
                missing_col,
                feature_frame[missing_col].mean() * 100,
                len(multihot_cols),
            )
    if _injected_mnar:
        # Write the updated feature frame back so FE sees the new column
        feature_frame.to_csv(artifacts["feature_frame.csv"], index=False)

    # --- Record deferred categorical columns (object-dtype) for FE identity check ---
    deferred_categorical_columns = {
        col: int(feature_frame[col].nunique())
        for col in feature_frame.select_dtypes(include="object").columns
    }

    logger.info("    feature_frame: %d rows x %d cols", len(feature_frame), len(feature_frame.columns))
    # Sort labels so the encoded class mapping stays stable across runs.
    class_names = sorted(pd.Series(target_series.dropna()).astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    encoded_target = target_series.astype(str).map(label_to_id)
    split_manifest = json.loads(Path(artifacts["split_manifest.json"]).read_text(encoding="utf-8"))
    train_indices = split_manifest["train_indices"]
    test_indices = split_manifest["test_indices"]
    validation_policy = (state.dataset_policy_spec or {}).get("validation_policy") or {}
    group_column = validation_policy.get("group_column") or (state.dataset_policy_spec or {}).get("group_column")
    time_column = validation_policy.get("time_column")

    train_group_values = None
    test_group_values = None
    train_time_values = None
    test_time_values = None
    raw_frame_for_policy = None
    if (group_column or time_column) and state.preprocessing_raw_frame_path:
        try:
            raw_frame_for_policy = pd.read_csv(state.preprocessing_raw_frame_path, low_memory=False)
        except Exception as exc:
            logger.warning("    could not reload raw frame for validation-policy metadata: %s", exc)

    if raw_frame_for_policy is not None and group_column and group_column in raw_frame_for_policy.columns:
        train_group_values = raw_frame_for_policy.iloc[train_indices][group_column].tolist()
        test_group_values = raw_frame_for_policy.iloc[test_indices][group_column].tolist()

    if raw_frame_for_policy is not None and time_column and time_column in raw_frame_for_policy.columns:
        train_time_values = raw_frame_for_policy.iloc[train_indices][time_column].tolist()
        test_time_values = raw_frame_for_policy.iloc[test_indices][time_column].tolist()

    # Clean up older run folders only after we have rebuilt everything we still need from this run.
    run_root = Path(state.preprocessing_workspace).parent
    cleanup_old_workspaces(run_root, keep_latest=1)

    logger.info("    class_names: %s, train: %d, test: %d", class_names, len(train_indices), len(test_indices))
    return {
        "preprocessing_validation_report": validation_report,
        "full_feature_frame": feature_frame,
        "train_frame": feature_frame.iloc[train_indices],
        "test_frame": feature_frame.iloc[test_indices],
        "train_target": encoded_target.iloc[train_indices],
        "test_target": encoded_target.iloc[test_indices],
        "train_group_values": train_group_values,
        "test_group_values": test_group_values,
        "train_time_values": train_time_values,
        "test_time_values": test_time_values,
        "feature_columns": feature_frame.columns.tolist(),
        "deferred_categorical_columns": deferred_categorical_columns,
        "class_names": class_names,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
    }


# LLM-based quality review of the preprocessing output.
def review_preprocessing_quality_node(state: CreditRiskState):
    logger.info(">>> review-preprocessing-quality")

    # Skip LLM call if structural validation already failed (files may not exist).
    validation = state.preprocessing_validation_report or {}
    if not validation.get("passed", False):
        logger.info("    skipping LLM quality review — structural validation failed")
        return {"preprocessing_audit_report": {"verdict": "skip", "issues": [], "summary": "Skipped — structural validation failed."}}

    execution_result = {
        "workspace_path": state.preprocessing_workspace,
        "artifacts": state.preprocessing_artifacts,
        "raw_frame_path": state.preprocessing_raw_frame_path,
    }
    # Pass previous audit report so the reviewer can do a focused re-check, not a fresh review.
    previous_audit = state.preprocessing_audit_report if state.preprocessing_audit_report else None
    audit_report = review_preprocessing_quality(
        execution_result,
        state.dataset_policy_spec,
        state.column_transform_spec,
        previous_audit_report=previous_audit,
    )
    verdict = audit_report.get("verdict", "pass")
    issues = audit_report.get("issues", [])
    # Deterministic override: if the LLM said "pass" but left critical/major issues in the
    # report, force needs_repair.  This prevents the reviewer from being too lenient and
    # allows major issues to slip through as "pass" with logged warnings.
    if verdict != "needs_repair":
        actionable = [i for i in issues if i.get("severity") in ("critical", "major")]
        if actionable:
            logger.warning(
                "    audit verdict overridden to needs_repair — %d major/critical issue(s) present despite '%s' verdict",
                len(actionable), verdict,
            )
            verdict = "needs_repair"
            audit_report = dict(audit_report)
            audit_report["verdict"] = "needs_repair"
            audit_report.setdefault("override_reason", f"Deterministic override: {len(actionable)} major/critical issue(s) present.")
    logger.info("    verdict: %s, issues: %d", verdict, len(issues))
    for issue in issues:
        level = logging.WARNING if issue.get("severity") in ("critical", "major") else logging.INFO
        logger.log(level, "    [%s/%s] %s: %s",
                   issue.get("severity"), issue.get("category"),
                   issue.get("column") or "general", issue.get("description"))
    logger.info("    summary: %s", audit_report.get("summary"))

    # Feed quality issues into the validation report so the repair loop can see them.
    # Preserve the original structural pass/fail separately so the routing function
    # can distinguish "structure OK but quality failed" from "structure failed".
    merged_validation = dict(validation)
    merged_validation["structural_passed"] = validation.get("passed", False)
    if verdict == "needs_repair":
        merged_validation["passed"] = False
        merged_validation.setdefault("errors", [])
        for issue in issues:
            if issue.get("severity") in ("critical", "major"):
                merged_validation["errors"].append({
                    "rule": f"quality_{issue.get('category', 'unknown')}",
                    "message": f"[{issue.get('column') or 'general'}] {issue.get('description')} Suggestion: {issue.get('suggestion', 'N/A')}",
                })

    return {
        "preprocessing_audit_report": audit_report,
        "preprocessing_validation_report": merged_validation,
    }


# Ask the LLM to repair preprocessing code after a failed attempt.
def repair_preprocessing_code_node(state: CreditRiskState):
    attempt_count = state.preprocessing_attempt_count or 0
    logger.warning(">>> repair-preprocessing-code (attempt %d/%d)", attempt_count, MAX_REPAIR_ATTEMPTS)
    # attempt_count includes the initial generation, so 3 means initial try plus up to 2 repairs.
    if attempt_count >= MAX_REPAIR_ATTEMPTS:
        raise RuntimeError(
            f"Preprocessing repair failed after {attempt_count} attempts."
        )

    escalate = bool(state.preprocessing_escalation_triggered)
    if escalate:
        logger.warning("    escalating repair call (caller=repair-preprocessing-code-escalated)")
    repaired_code = repair_preprocessing_code(
        previous_generated_code=state.preprocessing_code,
        code_review=state.preprocessing_code_review or {},
        execution_log=state.preprocessing_execution_log or {},
        validation_report=state.preprocessing_validation_report or {},
        dataset_profile=state.dataset_profile,
        dataset_policy_spec=state.dataset_policy_spec,
        column_transform_spec=state.column_transform_spec,
        escalate=escalate,
    )
    return {
        "preprocessing_code": repaired_code,
        "preprocessing_codegen_metadata": {
            "entrypoint": repaired_code.get("entrypoint"),
        },
        "preprocessing_attempt_count": attempt_count + 1,
        "preprocessing_escalation_triggered": False,  # consume trigger; re-arm only if repeat persists
    }


# Generate feature engineering code via LLM.
def generate_feature_engineering_code_node(state: CreditRiskState):
    logger.info(">>> generate-feature-engineering-code")
    generated_code = generate_feature_engineering_code(
        state.train_frame,
        state.test_frame,
        state.feature_columns,
        state.dataset_profile,
        eda_report=state.eda_report,
        eda_hypotheses=state.eda_hypotheses,
    )
    logger.info("    entrypoint: %s, code length: %d chars",
                generated_code.get("entrypoint"), len(generated_code.get("code", "")))

    # Extract and log hypothesis if provided by the LLM.
    hypothesis = generated_code.pop("hypothesis", None)
    if hypothesis:
        logger.info("    FE hypothesis: %s", json.dumps(hypothesis, default=str))

    return {
        "feature_engineering_code": generated_code,
        "feature_engineering_hypothesis": hypothesis,
        "feature_engineering_attempt_count": 1,
    }


# Reuse the same AST inspector as preprocessing — same safety rules apply.
def inspect_feature_engineering_code_node(state: CreditRiskState):
    attempt_count = state.feature_engineering_attempt_count or 1
    logger.info(">>> inspect-feature-engineering-code (attempt %d/%d)", attempt_count, MAX_FE_REPAIR_ATTEMPTS)
    code_review = inspect_preprocessing_code(state.feature_engineering_code)
    passed = code_review.get("passed", False)
    logger.info("    inspection passed: %s, issues: %d", passed, len(code_review.get("issues", [])))
    if not passed:
        for issue in code_review.get("issues", []):
            logger.warning("    inspection issue: [%s] %s", issue.get("rule"), issue.get("message"))
    return {"feature_engineering_code_review": code_review}


# Execute feature engineering code in a subprocess.
def execute_feature_engineering_node(state: CreditRiskState):
    logger.info(">>> execute-feature-engineering")
    run_root = Path(state.raw_dataset_path).resolve().parent / "feature_engineering_runs"
    execution_result = execute_feature_engineering(
        state.train_frame,
        state.test_frame,
        state.feature_engineering_code,
        run_root,
    )
    exec_log = execution_result.get("execution_log", {})
    logger.info("    returncode: %s, timed_out: %s, success: %s",
                exec_log.get("returncode"), exec_log.get("timed_out"), execution_result.get("success"))
    if exec_log.get("stderr"):
        logger.warning("    subprocess stderr:\n%s", exec_log["stderr"][:2000])
    missing = execution_result.get("missing_artifacts", [])
    if missing:
        logger.warning("    missing artifacts: %s", missing)
    return {"feature_engineering_execution_log": execution_result}


# Validate FE output and overwrite train/test frames on success.
def validate_feature_engineering_node(state: CreditRiskState):
    logger.info(">>> validate-feature-engineering")
    execution_result = state.feature_engineering_execution_log or {}
    # Extract top-5 MI raw feature names for the drop-protection validator.
    _top_mi = None
    if state.eda_report and state.eda_report.get("top_discriminative_features"):
        _top_mi = [f["column"] for f in state.eda_report["top_discriminative_features"][:5]]
    validation_report = validate_feature_engineering_output(
        execution_result,
        original_train_rows=len(state.train_frame),
        original_test_rows=len(state.test_frame),
        original_feature_count=len(state.feature_columns),
        deferred_categoricals=state.deferred_categorical_columns,
        top_mi_features=_top_mi,
        train_frame_pre_fe=state.train_frame,
    )
    logger.info("    validation passed: %s, checks: %s", validation_report.get("passed"), validation_report.get("checks"))
    if validation_report.get("errors"):
        for err in validation_report["errors"]:
            logger.warning("    validation error: [%s] %s", err.get("rule"), err.get("message"))

    if not validation_report["passed"]:
        return {"feature_engineering_validation_report": validation_report}

    artifacts = execution_result.get("artifacts", {})
    train_indices = state.train_frame.index.tolist()
    test_indices = state.test_frame.index.tolist()
    train_views = {}
    test_views = {}
    full_feature_frames_by_view = {}
    feature_columns_by_view = {}

    for view_name, view_spec in (validation_report.get("views") or {}).items():
        train_artifact = view_spec["train_artifact"]
        test_artifact = view_spec["test_artifact"]
        engineered_train = pd.read_csv(artifacts[train_artifact])
        engineered_test = pd.read_csv(artifacts[test_artifact])
        full_feature_frame = pd.concat([engineered_train, engineered_test], ignore_index=False)
        full_feature_frame.index = train_indices + test_indices
        full_feature_frame = full_feature_frame.sort_index()

        train_views[view_name] = engineered_train
        test_views[view_name] = engineered_test
        full_feature_frames_by_view[view_name] = full_feature_frame
        feature_columns_by_view[view_name] = engineered_train.columns.tolist()

    default_view = _default_feature_view_name(train_views)
    default_train = train_views[default_view]
    default_test = test_views[default_view]
    default_full = full_feature_frames_by_view[default_view]
    default_columns = feature_columns_by_view[default_view]
    model_view_map = _default_model_view_map(train_views)

    logger.info(
        "    feature views: %s",
        {view_name: len(cols) for view_name, cols in feature_columns_by_view.items()},
    )
    logger.info("    default active view: %s", default_view)
    if model_view_map:
        logger.info("    model view map: %s", model_view_map)

    # Log the FE report if available
    report_path = artifacts.get("feature_engineering_report.json")
    if report_path:
        try:
            fe_report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            for key in ("dropped", "transformed", "added"):
                items = fe_report.get(key, [])
                # Deduplicate by column name — dual-view code sometimes appends the same
                # entry twice (once per view) into the shared report list.
                seen: set[str] = set()
                unique_items = []
                for item in items:
                    col_key = item.get("column") if isinstance(item, dict) else str(item)
                    if col_key not in seen:
                        seen.add(col_key)
                        unique_items.append(item)
                if unique_items:
                    logger.info("    FE %s (%d):", key, len(unique_items))
                    for item in unique_items[:10]:
                        logger.info("      %s", json.dumps(item, default=str)[:150])
        except Exception:
            pass

    return {
        "feature_engineering_validation_report": validation_report,
        "train_frame": default_train,
        "test_frame": default_test,
        "full_feature_frame": default_full,
        "feature_columns": default_columns,
        "train_views": train_views,
        "test_views": test_views,
        "full_feature_frames_by_view": full_feature_frames_by_view,
        "feature_columns_by_view": feature_columns_by_view,
        "model_view_map": model_view_map,
    }


# Repair failed feature engineering code.
def repair_feature_engineering_code_node(state: CreditRiskState):
    attempt_count = state.feature_engineering_attempt_count or 0
    logger.warning(">>> repair-feature-engineering-code (attempt %d/%d)", attempt_count, MAX_FE_REPAIR_ATTEMPTS)
    if attempt_count >= MAX_FE_REPAIR_ATTEMPTS:
        raise RuntimeError(
            f"Feature engineering repair failed after {attempt_count} attempts."
        )

    exec_result = state.feature_engineering_execution_log or {}
    repaired_code = repair_feature_engineering_code(
        previous_generated_code=state.feature_engineering_code,
        code_review=state.feature_engineering_code_review or {},
        execution_log=exec_result.get("execution_log") or {},
        validation_report=state.feature_engineering_validation_report or {},
        feature_columns=state.feature_columns,
        dataset_profile=state.dataset_profile,
    )
    return {
        "feature_engineering_code": repaired_code,
        "feature_engineering_attempt_count": attempt_count + 1,
    }


# Fit the candidate sklearn models on the generated training data.
def train_models_node(state: CreditRiskState):
    logger.info(">>> train-models")
    from sklearn.utils.class_weight import compute_sample_weight

    models = build_candidate_models()
    sample_weights = compute_sample_weight("balanced", state.train_target)
    validation_policy = (state.dataset_policy_spec or {}).get("validation_policy") or {}
    if validation_policy:
        logger.info("    validation policy: %s", json.dumps(validation_policy, default=str))

    # Step 1: Quick baseline via cross-validation on training data only.
    # Never touch test_frame here — baseline metrics feed the LLM grid reasoner.
    # IMPORTANT: use the same grouped/temporal policy as tuning so the LLM sees
    # realistic (not inflated) scores when it reasons about hyperparameter grids.
    from sklearn.metrics import f1_score
    from bt5151_credit_risk.train import _build_cv_splits, _normalize_validation_policy
    baseline_metrics = {}
    _policy = _normalize_validation_policy(
        validation_policy=validation_policy,
        group_values=state.train_group_values,
        time_values=state.train_time_values,
    )
    _splits = _build_cv_splits(
        state.train_target,
        _policy,
        group_values=state.train_group_values,
        n_splits=3,
    )
    if _splits:
        import sklearn.base
        _n_splits = len(_splits)
        _all_classes = set(state.train_target.unique())
        for model_name, model in models.items():
            train_view = _get_train_frame_for_model(state, model_name)
            _fold_scores = []
            for tr_idx, val_idx in _splits:
                # Skip folds where a class is absent from train or val — can happen
                # on small datasets with strict group constraints.
                _tr_classes = set(state.train_target.iloc[tr_idx].unique())
                _val_classes = set(state.train_target.iloc[val_idx].unique())
                if not (_all_classes <= _tr_classes and _val_classes <= _tr_classes):
                    continue
                try:
                    _m = sklearn.base.clone(model)
                    _m.fit(train_view.iloc[tr_idx], state.train_target.iloc[tr_idx])
                    _preds = _m.predict(train_view.iloc[val_idx])
                    _fold_scores.append(f1_score(state.train_target.iloc[val_idx], _preds, average="macro"))
                except Exception as exc:
                    logger.warning("    baseline fold skipped (%s): %s", model_name, exc)
            if _fold_scores:
                baseline_metrics[model_name] = {
                    "macro_f1": round(float(np.mean(_fold_scores)), 4),
                }
        logger.info(
            "    baseline metrics (%d-fold %s CV on train): %s",
            _n_splits, _policy["type"], baseline_metrics,
        )
    else:
        logger.warning("    skipping baseline CV — could not build splits")

    # Step 2: LLM reasons hyperparameter grids.
    class_distribution = {}
    if state.dataset_profile and "target_distribution" in state.dataset_profile:
        class_distribution = state.dataset_profile["target_distribution"]

    grids_response = reason_hyperparameter_grids(
        model_names=list(models.keys()),
        train_rows=len(state.train_frame),
        feature_count=len(state.feature_columns),
        class_distribution=class_distribution,
        current_metrics=baseline_metrics,
    )
    grids = grids_response.get("grids", {})
    logger.info("    LLM-reasoned grids:")
    for model_name, grid in grids.items():
        logger.info("      %s: %s", model_name, json.dumps(grid, default=str))
    if grids_response.get("reasoning"):
        logger.info("    tuning reasoning: %s", grids_response["reasoning"])

    # Step 3: Tune with Optuna Bayesian optimization.
    tuned_models = {}
    tuning_results = {}
    trial_histories = {}
    for model_name, model in build_candidate_models().items():
        train_view = _get_train_frame_for_model(state, model_name)
        tuned_subset, tuning_subset, trial_subset = tune_models(
            {model_name: model},
            grids,
            train_view,
            state.train_target,
            sample_weights=sample_weights,
            validation_policy=validation_policy,
            train_group_values=state.train_group_values,
            train_time_values=state.train_time_values,
        )
        tuned_models.update(tuned_subset)
        tuning_results.update(tuning_subset)
        trial_histories.update(trial_subset)

    # Extract learning curves for XGBoost.
    learning_curves = {}
    for model_name, model in tuned_models.items():
        curves = extract_learning_curves(model, model_name)
        if curves:
            learning_curves[model_name] = curves

    candidate_model_specs = {}
    for model_name, model in tuned_models.items():
        candidate_model_specs[model_name] = {
            "estimator": type(model).__name__,
            "tuning": tuning_results.get(model_name),
            "feature_view": _get_model_view_name(state, model_name),
        }
    logger.info("    trained models: %s", list(tuned_models.keys()))
    return {
        "candidate_model_specs": candidate_model_specs,
        "trained_models": tuned_models,
        "learning_curves": learning_curves if learning_curves else None,
        "tuning_trial_history": trial_histories if trial_histories else None,
    }


# Score every candidate model on the held-out test split.
def evaluate_models_node(state: CreditRiskState):
    logger.info(">>> evaluate-models")
    results = {}
    for model_name, model in state.trained_models.items():
        test_view = _get_test_frame_for_model(state, model_name)
        predictions = model.predict(test_view)
        results[model_name] = compute_multiclass_metrics(
            state.test_target,
            predictions,
            state.class_names,
        )
    for name, metrics in results.items():
        logger.info("    %s: accuracy=%.4f macro_f1=%.4f", name, metrics.get("accuracy", 0), metrics.get("macro_f1", 0))
        cm = metrics.get("confusion_matrix")
        if cm:
            logger.info("    %s confusion matrix (rows=true, cols=pred):", name)
            for i, row in enumerate(cm):
                logger.info("      %s: %s", state.class_names[i], row)
        per_class = metrics.get("per_class", {})
        for cls_name, cls_metrics in per_class.items():
            logger.info("    %s  %s: precision=%.3f recall=%.3f f1=%.3f support=%d",
                        name, cls_name, cls_metrics.get("precision", 0),
                        cls_metrics.get("recall", 0), cls_metrics.get("f1-score", 0),
                        cls_metrics.get("support", 0))
    return {"evaluation_results": results}


# LLM-driven training diagnostics: per-class analysis, capacity, hypothesis validation.
def training_diagnostics_node(state: CreditRiskState):
    logger.info(">>> training-diagnostics")

    # Build tuning_results from candidate_model_specs
    tuning_results = {}
    if state.candidate_model_specs:
        for m_name, spec in state.candidate_model_specs.items():
            if spec.get("tuning"):
                tuning_results[m_name] = spec["tuning"]

    # Compute confidence stats programmatically for the LLM to interpret
    confidence_stats = {}
    for model_name, model in state.trained_models.items():
        test_view = _get_test_frame_for_model(state, model_name)
        try:
            probas = model.predict_proba(test_view)
            preds = model.predict(test_view)
            correct = preds == state.test_target.values
            model_conf = {}
            if correct.any():
                correct_max_proba = probas[correct].max(axis=1)
                model_conf.update({
                    "correct_mean_confidence": round(float(correct_max_proba.mean()), 4),
                    "correct_std_confidence": round(float(correct_max_proba.std()), 4),
                })
            if (~correct).any():
                wrong_max_proba = probas[~correct].max(axis=1)
                model_conf.update({
                    "wrong_mean_confidence": round(float(wrong_max_proba.mean()), 4),
                    "wrong_std_confidence": round(float(wrong_max_proba.std()), 4),
                })
            per_class_correct = {}
            per_class_wrong = {}
            for cidx, cname in enumerate(state.class_names or []):
                pred_mask = preds == cidx
                correct_mask = correct & pred_mask
                wrong_mask = (~correct) & pred_mask
                if correct_mask.any():
                    per_class_correct[cname] = round(float(probas[correct_mask].max(axis=1).mean()), 4)
                if wrong_mask.any():
                    per_class_wrong[cname] = round(float(probas[wrong_mask].max(axis=1).mean()), 4)
            if per_class_correct:
                model_conf["per_class_correct_mean_confidence"] = per_class_correct
            if per_class_wrong:
                model_conf["per_class_wrong_mean_confidence"] = per_class_wrong
            confidence_stats[model_name] = model_conf
        except Exception:
            pass

    try:
        diagnostics = generate_training_diagnostics(
            evaluation_results=state.evaluation_results,
            tuning_results=tuning_results,
            learning_curves=state.learning_curves,
            eda_hypotheses=state.eda_hypotheses,
            feature_engineering_hypothesis=state.feature_engineering_hypothesis,
            class_names=state.class_names,
            confidence_stats=confidence_stats,
        )
    except Exception as exc:
        logger.warning("    training diagnostics failed (%s), continuing without", exc)
        return {"training_diagnostics": None}

    raw_confidence_analysis = diagnostics.get("confidence_analysis")
    if isinstance(raw_confidence_analysis, dict):
        confidence_analysis = dict(raw_confidence_analysis)
        confidence_analysis.setdefault("summary", "")
        confidence_analysis["by_model"] = confidence_stats
    else:
        confidence_analysis = {
            "summary": raw_confidence_analysis or "",
            "by_model": confidence_stats,
        }
    diagnostics["confidence_analysis"] = confidence_analysis

    per_class = diagnostics.get("per_class_analysis", {})
    for cls_name, analysis in per_class.items():
        logger.info("    %s: struggle=%s — %s", cls_name,
                     analysis.get("struggle_level", "?"), analysis.get("diagnosis", "")[:120])
    capacity = diagnostics.get("capacity_analysis", {})
    for m_name, cap in capacity.items():
        logger.info("    capacity %s: %s", m_name, str(cap)[:150])
    hyp_val = diagnostics.get("hypothesis_validation", {})
    tested = hyp_val.get("tested", [])
    logger.info("    hypothesis validation: %d tested predictions checked", len(tested))
    for h in tested:
        logger.info("      [%s] %s → %s", h.get("outcome", "?"), h.get("original", "")[:80], h.get("actual", "")[:80])

    return {"training_diagnostics": diagnostics}


# Choose the final model from the evaluation results.
def select_model_node(state: CreditRiskState):
    logger.info(">>> select-model")

    # Step 1: Compute global SHAP for the metric-best model.
    # Uses compute_global_shap (single call) instead of inline code.
    metric_best = choose_best_model(state.evaluation_results)
    best_model_name = metric_best["model_name"]
    global_shap_importance = []
    shap_result = {}  # full SHAP result (importance + beeswarm + dependence)
    best_test_view = _get_test_frame_for_model(state, best_model_name)
    best_feature_columns = _get_feature_columns_for_model(state, best_model_name)
    try:
        shap_result = compute_global_shap(
            state.trained_models[best_model_name], best_model_name,
            best_test_view, best_feature_columns, state.class_names,
        )
        global_shap_importance = shap_result.get("importance", [])
        logger.info("    global SHAP top-10 features:")
        for entry in global_shap_importance[:10]:
            logger.info("      %s: %.4f", entry["feature"], entry["mean_abs_shap"])
    except Exception as exc:
        logger.warning("    global SHAP computation failed: %s", exc)

    # Step 2: LLM-driven model selection with hypothesis validation.
    tuning_results = {}
    if state.candidate_model_specs:
        for m_name, spec in state.candidate_model_specs.items():
            if spec.get("tuning"):
                tuning_results[m_name] = spec["tuning"]

    eda_top_features = None
    if state.eda_report and state.eda_report.get("top_discriminative_features"):
        eda_top_features = state.eda_report["top_discriminative_features"][:10]

    selection = reason_model_selection(
        evaluation_results=state.evaluation_results,
        tuning_results=tuning_results,
        global_shap_importance=global_shap_importance,
        eda_top_features=eda_top_features,
        fe_hypothesis=state.feature_engineering_hypothesis,
        class_names=state.class_names,
    )
    advisory_model_name = selection.get("llm_model_name", selection.get("model_name"))
    selected_name = best_model_name
    if advisory_model_name and advisory_model_name != best_model_name:
        logger.warning(
            "    advisory LLM model_name=%s ignored; choose_best_model() selected %s",
            advisory_model_name,
            best_model_name,
        )
    logger.info("    selected: %s", selected_name)
    logger.info("    justification: %s", selection.get("justification", ""))
    if selection.get("hypothesis_validation"):
        logger.info("    hypothesis validation: %s", selection["hypothesis_validation"])

    # Ensure global_shap_importance belongs to the selected model.
    # Recompute only when LLM picked a different model or first attempt failed.
    needs_recompute = (selected_name != best_model_name) or (not global_shap_importance)
    if needs_recompute:
        reason = ("LLM selected %s (metric-best was %s)" % (selected_name, best_model_name)
                  if selected_name != best_model_name
                  else "initial SHAP failed for %s" % best_model_name)
        logger.info("    %s — computing SHAP for selected model", reason)
        try:
            selected_test_view = _get_test_frame_for_model(state, selected_name)
            selected_feature_columns = _get_feature_columns_for_model(state, selected_name)
            shap_result = compute_global_shap(
                state.trained_models[selected_name], selected_name,
                selected_test_view, selected_feature_columns, state.class_names,
            )
            global_shap_importance = shap_result.get("importance", global_shap_importance)
        except Exception as exc:
            logger.warning("    SHAP computation for %s failed: %s", selected_name, exc)

    return {
        "selected_model_name": selected_name,
        "selection_justification": selection.get("justification", ""),
        "global_shap_importance": global_shap_importance,
        # Pass full SHAP result so global_xai_node can reuse it.
        "global_xai_results": {"shap": shap_result} if shap_result else None,
    }


# Compute global XAI evidence: SHAP beeswarm/dependence, grouped PFI, PDP/ALE (gated).
def global_xai_node(state: CreditRiskState):
    logger.info(">>> global-xai")
    model = state.trained_models[state.selected_model_name]
    model_name = state.selected_model_name
    test_view = _get_test_frame_for_model(state, model_name)
    feature_columns = _get_feature_columns_for_model(state, model_name)
    results = {"methods_used": []}

    # Reuse SHAP result from select_model_node if available, otherwise compute.
    prior_shap = (state.global_xai_results or {}).get("shap")
    if prior_shap and prior_shap.get("importance"):
        shap_full = prior_shap
        logger.info("    SHAP reused from select-model (skipping recomputation)")
    else:
        try:
            shap_full = compute_global_shap(
                model, model_name, test_view,
                feature_columns, state.class_names,
            )
        except Exception as exc:
            logger.warning("    SHAP computation failed: %s", exc)
            shap_full = {}
    if shap_full:
        results["shap"] = shap_full
        results["methods_used"].append("shap")
        logger.info("    SHAP beeswarm: %d features, dependence: %d features",
                     len(shap_full.get("beeswarm_data", {})), len(shap_full.get("dependence_data", {})))

    # Always: grouped PFI
    try:
        pfi = compute_permutation_importance(
            model, model_name, test_view, state.test_target,
            feature_columns,
            column_transform_spec=state.column_transform_spec,
        )
        results["pfi"] = pfi
        results["methods_used"].append("pfi_grouped")
        logger.info("    PFI grouped top-5:")
        for entry in pfi.get("grouped", [])[:5]:
            logger.info("      %s: %.4f (n_cols=%d)", entry["original_feature"],
                         entry["importance_mean"], entry["n_columns"])
    except Exception as exc:
        logger.warning("    PFI computation failed: %s", exc)

    # Identify top continuous-ish features from SHAP importance.
    # We keep the set small for runtime, but compute both PDP and ALE where feasible
    # because they are complementary: PDP gives a direct average effect view, while
    # ALE is more reliable when local feature correlations distort PDP.
    top_shap_features = [e["feature"] for e in (state.global_shap_importance or [])[:10]]
    numeric_cols = set(test_view.select_dtypes(include="number").columns)
    top_continuous = [f for f in top_shap_features if f in numeric_cols][:5]

    if top_continuous:
        try:
            pdp = compute_partial_dependence(
                model, model_name, test_view,
                feature_columns, top_continuous, state.class_names,
            )
            if pdp:
                results["pdp"] = pdp
                results["methods_used"].append("pdp")
                logger.info("    PDP computed for %d features: %s", len(pdp), list(pdp.keys()))
        except Exception as exc:
            logger.warning("    PDP computation failed: %s", exc)

        try:
            ale = compute_ale(
                model, model_name, test_view,
                feature_columns, top_continuous, state.class_names,
            )
            if ale:
                results["ale"] = ale
                results["methods_used"].append("ale")
                logger.info("    ALE computed for %d features: %s", len(ale), list(ale.keys()))
        except Exception as exc:
            logger.warning("    ALE computation failed: %s", exc)

    logger.info("    methods used: %s", results["methods_used"])
    return {"global_xai_results": results}


# Select systematic local XAI cases: representative, borderline, worst misclassification per class.
def local_xai_node(state: CreditRiskState):
    logger.info(">>> local-xai (casebook)")
    model = state.trained_models[state.selected_model_name]
    model_name = state.selected_model_name
    test_view = _get_test_frame_for_model(state, model_name)
    feature_columns = _get_feature_columns_for_model(state, model_name)

    cases = select_classification_cases(
        test_view, state.test_target, model,
        state.class_names, state.id_to_label,
    )

    # Compute SHAP waterfall for each case
    for case in cases:
        try:
            input_frame = test_view.loc[[case["row_index"]]]
            predicted_code = int(model.predict(input_frame)[0])
            shap_contribs = compute_shap_contributions_for_case(
                model, model_name, input_frame,
                predicted_code, feature_columns, top_n=10,
            )
            case["shap_contributions"] = shap_contribs
        except Exception as exc:
            logger.warning("    SHAP for case row=%d failed: %s", case["row_index"], exc)
            case["shap_contributions"] = []

    logger.info("    selected %d local XAI cases", len(cases))
    for case in cases:
        logger.info("      [%s] row=%d true=%s pred=%s conf=%.3f%s",
                     case["case_type"], case["row_index"], case["true_label"],
                     case["predicted_label"], max(case["probabilities"].values()),
                     f" confused_with={case['confused_with_class']}" if case.get("confused_with_class") else "")

    return {"local_xai_cases": cases}


# Deterministic shortcut-feature audit: detect SHAP/PFI divergence, dominance, or
# calendar/index shortcuts in the top-ranked features. Run capped ablation (max 2)
# on the selected model using zero-out on the test view. Writes shortcut_audit.json.
def shortcut_audit_node(state: CreditRiskState):
    logger.info(">>> shortcut-feature-audit")
    model = state.trained_models[state.selected_model_name]
    model_name = state.selected_model_name
    test_view = _get_test_frame_for_model(state, model_name)

    shap_importance = state.global_shap_importance or []
    pfi_grouped = (state.global_xai_results or {}).get("pfi", {}).get("grouped", [])

    if not shap_importance or not pfi_grouped:
        logger.info("    skip: shap_importance or pfi_grouped missing")
        return {"shortcut_audit": {"suspects": [], "ablations": [], "skipped": True}}

    audit = run_shortcut_audit(
        model=model,
        test_view=test_view,
        y_test=state.test_target,
        shap_importance=shap_importance,
        pfi_grouped=pfi_grouped,
        class_names=state.class_names,
        max_ablations=2,
    )

    for s in audit["suspects"]:
        logger.info("    suspect: %s signals=%s shap_rank=%s pfi_rank=%s share=%.3f",
                     s["feature"], s["signals"], s["shap_rank"], s["pfi_rank"], s["shap_share"])
    for a in audit["ablations"]:
        logger.info("    ablation: %s verdict=%s Δmacro_f1=%s",
                     a["feature"], a["verdict"], a.get("delta_macro_f1"))

    # Persist artifact alongside analysis bundle (reuse preprocessing workspace if no explicit run dir)
    artifact_dir = None
    if state.preprocessing_workspace:
        from pathlib import Path as _Path
        artifact_dir = _Path(state.preprocessing_workspace)
    if artifact_dir is not None and artifact_dir.exists():
        try:
            (artifact_dir / "shortcut_audit.json").write_text(
                json.dumps(audit, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("    failed to write shortcut_audit.json: %s", exc)

    return {"shortcut_audit": audit}


# LLM interprets GLOBAL XAI evidence (SHAP/PFI/PDP/ALE) — model-level patterns only.
def interpret_global_xai_node(state: CreditRiskState):
    logger.info(">>> interpret-global-xai")

    interpretation = interpret_global_xai(
        global_xai_results=state.global_xai_results or {},
        class_names=state.class_names,
        training_diagnostics=state.training_diagnostics,
        eda_hypotheses=state.eda_hypotheses,
        feature_engineering_hypothesis=state.feature_engineering_hypothesis,
        shortcut_audit=state.shortcut_audit,
    )

    for obs in (interpretation.get("observations") or [])[:3]:
        logger.info("    observation: %s", obs[:120])
    for ins in (interpretation.get("insights") or [])[:2]:
        logger.info("    insight: %s", ins[:120])

    consensus = interpretation.get("feature_importance_consensus", {})
    if consensus.get("agreement"):
        logger.info("    SHAP/PFI agreement: %s", consensus["agreement"])
    if consensus.get("interpretation"):
        logger.info("    consensus: %s", consensus["interpretation"][:140])

    hyps = interpretation.get("hypotheses", {})
    for tier in ("tested_predictions", "supported_conjectures", "exploratory_leads"):
        for h in (hyps.get(tier) or []):
            logger.info("    global [%s] %s", tier.split("_")[0], h.get("hypothesis", "")[:120])

    return {"global_xai_interpretation": interpretation}


# LLM interprets LOCAL casebook — per-class stories, boundary analysis, case-grounded hypotheses.
def interpret_local_xai_node(state: CreditRiskState):
    logger.info(">>> interpret-local-xai")

    interpretation = interpret_local_xai(
        local_xai_cases=state.local_xai_cases or [],
        class_names=state.class_names,
        global_xai_interpretation=state.global_xai_interpretation,
        global_xai_results=state.global_xai_results,
        training_diagnostics=state.training_diagnostics,
    )

    stories = interpretation.get("per_class_stories", {}) or {}
    for cls, story in stories.items():
        if story.get("worst_misclassification_story"):
            logger.info("    [%s] worst: %s", cls, story["worst_misclassification_story"][:140])
    patterns = interpretation.get("confusion_patterns", {}) or {}
    if patterns.get("dominant_direction"):
        logger.info("    confusion direction: %s", patterns["dominant_direction"][:140])
    boundary = interpretation.get("decision_boundary_analysis", {}) or {}
    if boundary.get("thinnest_boundary"):
        logger.info("    thinnest boundary: %s", boundary["thinnest_boundary"][:140])

    hyps = interpretation.get("hypotheses", {})
    for tier in ("tested_predictions", "supported_conjectures", "exploratory_leads"):
        for h in (hyps.get(tier) or []):
            logger.info("    local [%s] %s", tier.split("_")[0], h.get("hypothesis", "")[:120])

    return {"local_xai_interpretation": interpretation}


# Package all analytical outputs into a serializable analysis bundle.
def package_analysis_bundle_node(state: CreditRiskState):
    logger.info(">>> package-analysis-bundle")
    import hashlib
    from datetime import datetime, timezone

    # The semantic bundle: every LLM-authored interpretation and every hypothesis
    # passes through verbatim. No [:5]/[:3] truncation — explain-risk is the consumer
    # that decides what to surface to the customer and should see the full evidence.
    bundle = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": state.run_id,
            "selected_model": state.selected_model_name,
            "selected_model_view": _get_model_view_name(state, state.selected_model_name),
            "class_names": state.class_names,
            "feature_count": len(_get_feature_columns_for_model(state, state.selected_model_name) or []),
            "dataset_hash": (
                hashlib.md5(
                    open(state.raw_dataset_path, "rb").read(8192)
                ).hexdigest()[:8]
                if state.raw_dataset_path and Path(state.raw_dataset_path).exists()
                else hashlib.md5(str(state.raw_dataset_path).encode()).hexdigest()[:8]
            ),
        },
        "eda_hypotheses": state.eda_hypotheses,
        "training_diagnostics": state.training_diagnostics,
        "global_xai_interpretation": state.global_xai_interpretation,
        "local_xai_interpretation": state.local_xai_interpretation,
        "local_casebook": state.local_xai_cases,
        "feature_engineering_hypothesis": state.feature_engineering_hypothesis,
        "selection_justification": state.selection_justification,
    }

    # Separate artifact for raw numeric arrays — kept out of the semantic bundle
    # so explain-risk isn't drowned in beeswarm matrices.
    numeric_artifacts = {}
    if state.global_xai_results:
        shap_block = state.global_xai_results.get("shap") or {}
        numeric_artifacts["shap_importance"] = shap_block.get("importance")
        numeric_artifacts["shap_dependence_features"] = list((shap_block.get("dependence_data") or {}).keys())
        numeric_artifacts["pfi"] = state.global_xai_results.get("pfi")
        numeric_artifacts["pdp"] = state.global_xai_results.get("pdp")
        numeric_artifacts["ale"] = state.global_xai_results.get("ale")
        numeric_artifacts["methods_used"] = state.global_xai_results.get("methods_used", [])

    # Save both to disk using the stage run_id when available so log and bundle share
    # one stable artifact identity.
    run_root = Path(state.raw_dataset_path).resolve().parent
    timestamp_str = state.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    full_on_disk = dict(bundle)
    full_on_disk["numeric_artifacts"] = numeric_artifacts
    # Save alongside run logs in lab/logs/ so all run artifacts are co-located.
    log_dir = run_root / "lab" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = log_dir / f"analysis_bundle_{timestamp_str}.json"
    bundle_path.write_text(json.dumps(full_on_disk, indent=2, default=str), encoding="utf-8")
    logger.info("    analysis bundle saved to %s", bundle_path)

    logger.info("    bundle keys: %s", list(bundle.keys()))
    return {
        "analysis_bundle": bundle,
        # Back-compat: explain-risk reads this field. Now it IS the full semantic bundle,
        # not a programmatic compression.
        "analysis_bundle_summary": bundle,
    }


# Run one inference example with the selected model.
def run_inference_node(state: CreditRiskState):
    logger.info(">>> run-inference")
    if not state.inference_input or "row_index" not in state.inference_input:
        raise ValueError("inference_input with 'row_index' is required.")

    row_index = int(state.inference_input["row_index"])
    model_name = state.selected_model_name
    input_frame = _get_full_frame_for_model(state, model_name).loc[[row_index]]
    feature_columns = _get_feature_columns_for_model(state, model_name)
    model = state.trained_models[model_name]
    probabilities = model.predict_proba(input_frame)[0]
    predicted_code = int(model.predict(input_frame)[0])
    predicted_label = state.id_to_label[predicted_code]
    probability_map = {
        state.id_to_label[idx]: float(score)
        for idx, score in enumerate(probabilities)
    }
    confidence = max(probability_map.values())
    logger.info("    row %d → predicted: %s, confidence: %.4f", row_index, predicted_label, confidence)

    # Full SHAP waterfall — top-10, all classes, base values.
    shap_result = {}
    try:
        shap_result = compute_shap_contributions_for_case(
            model, model_name, input_frame, predicted_code,
            feature_columns, top_n=10, class_names=state.class_names,
        )
        waterfall = shap_result.get("predicted_class_waterfall", {})
        for feat in (waterfall.get("top_features") or [])[:5]:
            logger.info("    SHAP [%s] %s = %.4f (%s)",
                        waterfall.get("class"), feat["feature"], feat["shap_value"], feat["direction"])
    except Exception as exc:
        logger.warning("    SHAP waterfall failed: %s", exc)

    # PDP position — for top-4 SHAP features, locate this row on the PDP curve.
    pdp_annotations = []
    try:
        pdp_results = (state.global_xai_results or {}).get("pdp") or {}
        if pdp_results and shap_result.get("predicted_class_waterfall"):
            top_shap_features = [
                f["feature"]
                for f in shap_result["predicted_class_waterfall"].get("top_features", [])
            ]
            row_values = {
                feat: float(input_frame[feat].iloc[0])
                for feat in top_shap_features
                if feat in input_frame.columns
            }
            pdp_annotations = annotate_pdp_position(row_values, pdp_results, top_n=4)
            for ann in pdp_annotations:
                logger.info("    PDP [%s] val=%.3f → grid_pct=%.0f%%, pd=%s",
                            ann["feature"], ann["feature_value"],
                            ann["grid_position_pct"], ann["pd_values_at_position"])
    except Exception as exc:
        logger.warning("    PDP annotation failed: %s", exc)

    # Confidence diagnosis — compare this prediction's confidence against per-class
    # calibration from training diagnostics to flag whether caution is warranted.
    confidence_diagnosis = {}
    try:
        diag = state.training_diagnostics or {}
        conf_analysis = diag.get("confidence_analysis") or {}
        per_class = diag.get("per_class_analysis") or {}
        # Typical correct-prediction confidence for predicted class (if available)
        typical_conf = None
        if isinstance(conf_analysis, dict):
            by_model = conf_analysis.get("by_model") or {}
            model_conf = by_model.get(model_name) or by_model.get(state.selected_model_name) or {}
            class_conf = model_conf.get("per_class_correct_mean_confidence") or {}
            if predicted_label in class_conf:
                try:
                    typical_conf = float(class_conf[predicted_label])
                except (TypeError, ValueError):
                    typical_conf = None
            if typical_conf is None:
                class_conf = model_conf.get("correct_mean_confidence")
                if class_conf is None:
                    class_conf = conf_analysis.get(predicted_label) or conf_analysis.get("correct_mean")
                if class_conf is not None:
                    try:
                        typical_conf = float(class_conf)
                    except (TypeError, ValueError):
                        typical_conf = None
        # Predicted class struggle level
        struggle = None
        if predicted_label in per_class:
            struggle = per_class[predicted_label].get("struggle_level") or per_class[predicted_label].get("struggle")
        # Determine caution flag
        if typical_conf is not None:
            gap = typical_conf - confidence
            if confidence < 0.45:
                caution = "high"
                reason = f"confidence {confidence:.2f} is very low — model is near-uniform across classes for this customer"
            elif gap > 0.15:
                caution = "medium"
                reason = f"confidence {confidence:.2f} is {gap:.2f} below the typical correct-prediction confidence ({typical_conf:.2f}) for {predicted_label}"
            else:
                caution = "low"
                reason = f"confidence {confidence:.2f} is in line with typical correct-prediction confidence for {predicted_label}"
        else:
            # Fallback: rule-of-thumb only
            if confidence < 0.45:
                caution = "high"
                reason = "confidence below 0.45 — near-uniform class probabilities"
            elif confidence < 0.65:
                caution = "medium"
                reason = "moderate confidence — prediction may be sensitive to small feature changes"
            else:
                caution = "low"
                reason = "confidence above 0.65"
        confidence_diagnosis = {
            "caution_level": caution,
            "reason": reason,
            "confidence": round(confidence, 4),
            "typical_correct_confidence": typical_conf,
            "predicted_class_struggle": struggle,
        }
        logger.info("    confidence_diagnosis: caution=%s, %.4f — %s", caution, confidence, reason)
    except Exception as exc:
        logger.warning("    Confidence diagnosis failed: %s", exc)

    # Nearest casebook case — find which of the 9 cases most resembles this row by SHAP cosine.
    nearest_case = {}
    try:
        cases = state.local_xai_cases or []
        if cases and shap_result.get("predicted_class_waterfall"):
            this_features = {
                f["feature"]: f["shap_value"]
                for f in shap_result["predicted_class_waterfall"].get("top_features", [])
            }
            best_sim, best_case = -1.0, None
            for case in cases:
                case_shap = case.get("shap_contributions") or {}
                if isinstance(case_shap, dict):
                    case_top = (case_shap.get("predicted_class_waterfall") or {}).get("top_features") or []
                else:
                    case_top = case_shap or []
                case_features = {f["feature"]: f["shap_value"] for f in case_top}
                common = set(this_features) & set(case_features)
                if not common:
                    continue
                a = np.array([this_features[k] for k in common])
                b = np.array([case_features[k] for k in common])
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                sim = float(np.dot(a, b) / denom) if denom > 0 else 0.0
                if sim > best_sim:
                    best_sim, best_case = sim, case
            if best_case:
                nearest_case = {
                    "case_type": best_case["case_type"],
                    "true_label": best_case["true_label"],
                    "predicted_label": best_case["predicted_label"],
                    "confidence": round(max(best_case["probabilities"].values()), 4),
                    "cosine_similarity": round(best_sim, 3),
                    "row_index": best_case["row_index"],
                }
                logger.info("    nearest casebook: [%s] row=%d sim=%.3f",
                            best_case["case_type"], best_case["row_index"], best_sim)
    except Exception as exc:
        logger.warning("    Nearest casebook lookup failed: %s", exc)

    return {
        "prediction_output": {
            "row_index": row_index,
            "predicted_label": predicted_label,
            "probabilities": probability_map,
            "confidence": round(confidence, 4),
            "selected_model_name": state.selected_model_name,
            "selected_model_view": _get_model_view_name(state, model_name),
            "evaluation_metrics": state.evaluation_results[state.selected_model_name],
            "source_record": state.raw_frame.loc[row_index].to_dict(),
            # Full SHAP waterfall (top-10, all classes, base values)
            "shap_waterfall": shap_result,
            # PDP position — where this row sits on the risk curve for top features
            "pdp_position": pdp_annotations,
            # Confidence diagnosis — is caution warranted?
            "confidence_diagnosis": confidence_diagnosis,
            # Nearest casebook case — which model behavior archetype this resembles
            "nearest_casebook_case": nearest_case,
        }
    }


# Explain-risk: synthesises the full analysis bundle + inference-time local diagnostics
# into a business explanation AND recommended action in one reasoning call.
def explain_risk_node(state: CreditRiskState):
    logger.info(">>> explain-risk")
    prediction_output = state.prediction_output

    explanation = explain_risk(
        prediction_output["predicted_label"],
        prediction_output["probabilities"],
        selected_model_name=prediction_output["selected_model_name"],
        evaluation_metrics=prediction_output["evaluation_metrics"],
        source_record=prediction_output["source_record"],
        shap_waterfall=prediction_output.get("shap_waterfall"),
        pdp_position=prediction_output.get("pdp_position"),
        confidence_diagnosis=prediction_output.get("confidence_diagnosis"),
        nearest_casebook_case=prediction_output.get("nearest_casebook_case"),
        global_shap_importance=state.global_shap_importance,
        analysis_bundle_summary=state.analysis_bundle_summary,
        selection_justification=state.selection_justification,
    )
    logger.info("    risk_level: %s, confidence_band: %s, caution: %s, action: %s",
                explanation.get("risk_level"), explanation.get("confidence_band"),
                explanation.get("confidence_assessment", {}).get("caution_level"),
                explanation.get("recommended_action", {}).get("action"))
    # Store recommended_action separately for downstream consumers and logging.
    recommended_action_out = explanation.pop("recommended_action", {})
    return {"risk_explanation": explanation, "recommended_action": recommended_action_out}


# Good code goes to execution; failed inspection goes to repair.
def _route_after_inspection(state: CreditRiskState):
    if state.preprocessing_code_review and state.preprocessing_code_review.get("passed"):
        return "execute-generated-preprocessing"
    return "repair-preprocessing-code"


# Both structural validation and quality review have run — route based on combined result.
def _route_after_quality_review(state: CreditRiskState):
    validation = state.preprocessing_validation_report or {}
    audit = state.preprocessing_audit_report or {}
    # Use the preserved structural result — not the merged "passed" flag which gets
    # overwritten to False by quality issues.
    structural_ok = validation.get("structural_passed", validation.get("passed", False))
    quality_ok = audit.get("verdict") in ("pass", None)
    if structural_ok and quality_ok:
        return "generate-feature-engineering-code"
    # If structural validation passed but quality review keeps failing, only accept
    # residual minor issues after repeated attempts. Critical/major issues should
    # keep the repair loop alive instead of silently flowing into training.
    attempt_count = state.preprocessing_attempt_count or 0
    blocking_issues = [
        issue for issue in audit.get("issues", [])
        if issue.get("severity") in ("critical", "major")
    ]
    if structural_ok and attempt_count >= 3 and not blocking_issues:
        logger.warning(
            "    only minor quality issues remain after %d attempts — accepting and moving on",
            attempt_count,
        )
        return "generate-feature-engineering-code"
    return "repair-preprocessing-code"


# Good FE code goes to execution; failed inspection goes to repair.
def _route_after_fe_inspection(state: CreditRiskState):
    if state.feature_engineering_code_review and state.feature_engineering_code_review.get("passed"):
        return "execute-feature-engineering"
    return "repair-feature-engineering-code"


# Passed FE validation goes to training; failed goes to repair.
def _route_after_fe_validation(state: CreditRiskState):
    validation = state.feature_engineering_validation_report or {}
    if validation.get("passed", False):
        return "train-models"
    return "repair-feature-engineering-code"


# Define the end-to-end LangGraph pipeline.
def build_graph():
    graph = StateGraph(CreditRiskState)
    graph.add_node("dataset-policy-spec", dataset_policy_spec_node)
    graph.add_node("exploratory-data-analysis", exploratory_data_analysis_node)
    graph.add_node("generate-eda-hypotheses", generate_eda_hypotheses_node)
    graph.add_node("column-transform-spec", column_transform_spec_node)
    graph.add_node("generate-preprocessing-code", generate_preprocessing_code_node)
    graph.add_node("inspect-preprocessing-code", inspect_preprocessing_code_node)
    graph.add_node("execute-generated-preprocessing", execute_generated_preprocessing_node)
    graph.add_node("validate-preprocessing-output", validate_preprocessing_output_node)
    graph.add_node("review-preprocessing-quality", review_preprocessing_quality_node)
    graph.add_node("repair-preprocessing-code", repair_preprocessing_code_node)
    graph.add_node("generate-feature-engineering-code", generate_feature_engineering_code_node)
    graph.add_node("inspect-feature-engineering-code", inspect_feature_engineering_code_node)
    graph.add_node("execute-feature-engineering", execute_feature_engineering_node)
    graph.add_node("validate-feature-engineering", validate_feature_engineering_node)
    graph.add_node("repair-feature-engineering-code", repair_feature_engineering_code_node)
    graph.add_node("train-models", train_models_node)
    graph.add_node("evaluate-models", evaluate_models_node)
    graph.add_node("training-diagnostics", training_diagnostics_node)
    graph.add_node("select-model", select_model_node)
    graph.add_node("global-xai", global_xai_node)
    graph.add_node("local-xai", local_xai_node)
    graph.add_node("shortcut-feature-audit", shortcut_audit_node)
    graph.add_node("interpret-global-xai", interpret_global_xai_node)
    graph.add_node("interpret-local-xai", interpret_local_xai_node)
    graph.add_node("package-analysis-bundle", package_analysis_bundle_node)
    graph.add_node("run-inference", run_inference_node)
    graph.add_node("explain-risk", explain_risk_node)
    # recommend-action merged into explain-risk — single reasoning call

    graph.add_edge(START, "dataset-policy-spec")
    graph.add_edge("dataset-policy-spec", "exploratory-data-analysis")
    graph.add_edge("exploratory-data-analysis", "generate-eda-hypotheses")
    graph.add_edge("generate-eda-hypotheses", "column-transform-spec")
    graph.add_edge("column-transform-spec", "generate-preprocessing-code")
    graph.add_edge("generate-preprocessing-code", "inspect-preprocessing-code")
    graph.add_conditional_edges(
        "inspect-preprocessing-code",
        _route_after_inspection,
        {
            "execute-generated-preprocessing": "execute-generated-preprocessing",
            "repair-preprocessing-code": "repair-preprocessing-code",
        },
    )
    graph.add_edge("execute-generated-preprocessing", "validate-preprocessing-output")
    graph.add_edge("validate-preprocessing-output", "review-preprocessing-quality")
    graph.add_conditional_edges(
        "review-preprocessing-quality",
        _route_after_quality_review,
        {
            "generate-feature-engineering-code": "generate-feature-engineering-code",
            "repair-preprocessing-code": "repair-preprocessing-code",
        },
    )
    graph.add_edge("repair-preprocessing-code", "inspect-preprocessing-code")
    graph.add_edge("generate-feature-engineering-code", "inspect-feature-engineering-code")
    graph.add_conditional_edges(
        "inspect-feature-engineering-code",
        _route_after_fe_inspection,
        {
            "execute-feature-engineering": "execute-feature-engineering",
            "repair-feature-engineering-code": "repair-feature-engineering-code",
        },
    )
    graph.add_edge("execute-feature-engineering", "validate-feature-engineering")
    graph.add_conditional_edges(
        "validate-feature-engineering",
        _route_after_fe_validation,
        {
            "train-models": "train-models",
            "repair-feature-engineering-code": "repair-feature-engineering-code",
        },
    )
    graph.add_edge("repair-feature-engineering-code", "inspect-feature-engineering-code")
    graph.add_edge("train-models", "evaluate-models")
    graph.add_edge("evaluate-models", "training-diagnostics")
    graph.add_edge("training-diagnostics", "select-model")
    graph.add_edge("select-model", "global-xai")
    graph.add_edge("global-xai", "local-xai")
    graph.add_edge("local-xai", "shortcut-feature-audit")
    graph.add_edge("shortcut-feature-audit", "interpret-global-xai")
    graph.add_edge("interpret-global-xai", "interpret-local-xai")
    graph.add_edge("interpret-local-xai", "package-analysis-bundle")
    graph.add_edge("package-analysis-bundle", "run-inference")
    graph.add_edge("run-inference", "explain-risk")
    graph.add_edge("explain-risk", END)
    return graph


# Compile the graph into a runnable pipeline object.
def compile_graph():
    return build_graph().compile()
