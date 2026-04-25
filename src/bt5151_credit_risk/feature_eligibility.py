from __future__ import annotations

from typing import Any


_IDENTIFIER_LIKE_NAME_TOKENS = (
    "customer",
    "account",
    "email",
    "phone",
    "address",
    "name",
    "ssn",
    "passport",
)


def apply_feature_eligibility(
    *,
    raw_top_features: list[dict[str, Any]],
    feature_profiles: dict[str, dict[str, Any]],
    dataset_policy_spec: dict | None = None,
) -> dict[str, Any]:
    """Split raw discriminative rankings into audit vs model-eligible views.

    Hard blocks come from explicit policy fields and strong near-unique
    identifier behavior. Name-based heuristics remain review-first: they create
    leakage alerts but do not automatically remove the feature unless another
    stronger rule applies.
    """

    policy = dataset_policy_spec or {}
    target_column = policy.get("target_column")
    group_column = policy.get("group_column")
    identifier_columns = {str(v) for v in (policy.get("identifier_columns") or [])}
    leakage_drop_columns = {
        str(v) for v in ((policy.get("leakage_policy") or {}).get("columns_to_drop") or [])
    }
    leakage_drop_columns.update(
        str(v) for v in ((policy.get("leakage_rules") or {}).get("drop_columns") or [])
    )

    model_eligible: list[dict[str, Any]] = []
    leakage_alerts: list[dict[str, Any]] = []
    feature_decisions: dict[str, dict[str, Any]] = {}

    for entry in raw_top_features or []:
        column = str(entry.get("column") or "")
        if not column:
            continue
        profile = feature_profiles.get(column, {})
        column_type = str(entry.get("column_type") or profile.get("column_type") or "").lower()
        decision = _classify_feature(
            column=column,
            column_type=column_type,
            profile=profile,
            target_column=target_column,
            group_column=group_column,
            identifier_columns=identifier_columns,
            leakage_drop_columns=leakage_drop_columns,
        )
        feature_decisions[column] = decision
        if decision["severity"] == "block":
            leakage_alerts.append(_build_alert(entry, decision))
            continue
        model_eligible.append(entry)
        if decision["severity"] == "review":
            leakage_alerts.append(_build_alert(entry, decision))

    return {
        "raw_top_discriminative_features": list(raw_top_features or []),
        "model_eligible_top_discriminative_features": model_eligible,
        "leakage_alerts": leakage_alerts,
        "feature_decisions": feature_decisions,
    }


def _classify_feature(
    *,
    column: str,
    column_type: str,
    profile: dict[str, Any],
    target_column: str | None,
    group_column: str | None,
    identifier_columns: set[str],
    leakage_drop_columns: set[str],
) -> dict[str, str]:
    if target_column and column == target_column:
        return {
            "severity": "block",
            "reason": "target_column",
            "message": "Column is the declared target and must not appear in model-facing rankings.",
        }
    if group_column and column == group_column:
        return {
            "severity": "block",
            "reason": "group_column",
            "message": "Column is the declared grouping key and should be excluded from model-facing rankings.",
        }
    if column in identifier_columns:
        return {
            "severity": "block",
            "reason": "identifier_column",
            "message": "Column is explicitly listed as an identifier in dataset policy and should be excluded from model-facing rankings.",
        }
    if column in leakage_drop_columns:
        return {
            "severity": "block",
            "reason": "leakage_drop_rule",
            "message": "Column is explicitly listed in leakage/drop policy and should be excluded from model-facing rankings.",
        }
    if _is_near_unique_identifier(profile, column_type=column_type):
        return {
            "severity": "block",
            "reason": "near_unique_identifier_behavior",
            "message": "Column behaves like a near-unique identifier and should be excluded from model-facing rankings.",
        }
    if _looks_identifier_like(column):
        return {
            "severity": "review",
            "reason": "identifier_like_name",
            "message": "Column name looks identifier-like but was kept because policy/spec did not block it and uniqueness was not near-identifier level.",
        }
    return {
        "severity": "eligible",
        "reason": "eligible",
        "message": "No policy or strong heuristic block was triggered.",
    }


def _build_alert(entry: dict[str, Any], decision: dict[str, str]) -> dict[str, Any]:
    return {
        "column": entry["column"],
        "mutual_information": entry.get("mutual_information"),
        "severity": decision["severity"],
        "reason": decision["reason"],
        "message": decision["message"],
    }


def _looks_identifier_like(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in _IDENTIFIER_LIKE_NAME_TOKENS)


def _is_near_unique_identifier(profile: dict[str, Any], *, column_type: str) -> bool:
    if column_type == "numeric":
        return False
    nunique = int(profile.get("nunique") or 0)
    non_null_count = int(profile.get("non_null_count") or 0)
    if non_null_count <= 0:
        return False
    unique_ratio = nunique / non_null_count
    return nunique >= 50 and unique_ratio >= 0.98
