from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_SAFE_RATIO_SPECS = (
    ("EMI_to_Salary_Ratio", "Total_EMI_per_month", "Monthly_Inhand_Salary"),
    ("Debt_to_Income_Ratio", "Outstanding_Debt", "Annual_Income"),
    ("Balance_to_Salary_Ratio", "Monthly_Balance", "Monthly_Inhand_Salary"),
)


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = _numeric_series(denominator).replace(0, np.nan)
    numer = _numeric_series(numerator)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (numer / denom).replace([np.inf, -np.inf], np.nan)


def _apply_blocked_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: dict,
    lineage: dict,
    blocked_columns: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocked = [col for col in (blocked_columns or []) if col in train_df.columns or col in test_df.columns]
    for col in blocked:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
        report["dropped"].append({
            "column": col,
            "reason": "blocked_by_preprocessing_contract",
        })
        lineage["dropped_features"].append({
            "feature": col,
            "drop_reason": "leakage",
        })
    return train_df, test_df


def _add_ratio_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: dict,
    lineage: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for feature_name, numerator_col, denominator_col in _SAFE_RATIO_SPECS:
        if numerator_col not in train_df.columns or denominator_col not in train_df.columns:
            continue
        if numerator_col not in test_df.columns or denominator_col not in test_df.columns:
            continue

        train_ratio = _safe_ratio(train_df[numerator_col], train_df[denominator_col])
        median = train_ratio.median()
        if pd.isna(median):
            median = 0.0
        test_ratio = _safe_ratio(test_df[numerator_col], test_df[denominator_col])

        train_df[feature_name] = train_ratio.fillna(float(median))
        test_df[feature_name] = test_ratio.fillna(float(median))
        report["added"].append({
            "column": feature_name,
            "transform": "ratio",
            "formula": f"{numerator_col} / {denominator_col}",
            "rationale": "conservative deterministic ratio from validated preprocessing columns",
        })
        lineage["derived_features"].append({
            "feature": feature_name,
            "operation": "ratio",
            "inputs": [numerator_col, denominator_col],
            "input_stage": "pre_fe_raw_numeric",
            "fill_strategy": "median_fill_after_zero_guard",
        })
    return train_df, test_df


def engineer_deterministic_feature_views(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    deferred_categorical_columns: dict | None = None,
    blocked_columns: list[str] | None = None,
    fallback_used: bool = True,
    reason: str | None = None,
) -> dict:
    report = {
        "dropped": [],
        "transformed": [],
        "added": [],
        "fallback": {
            "used": bool(fallback_used),
            "reason": reason or (
                "Deterministic feature-engineering mode is active; using a validated "
                "pass-through + categorical encoding baseline."
            ),
        },
    }
    lineage = {"derived_features": [], "dropped_features": [], "passthrough_features": []}

    train_df = train_df.copy()
    test_df = test_df.copy()
    deferred = deferred_categorical_columns or {}

    train_df, test_df = _apply_blocked_columns(train_df, test_df, report, lineage, blocked_columns)
    train_df, test_df = _add_ratio_features(train_df, test_df, report, lineage)

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
                "rationale": "deterministic fallback encoding for deferred categorical",
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
                "rationale": "deterministic fallback compact encoding for high-cardinality categorical",
            })
            lineage["derived_features"].append({
                "feature": col,
                "operation": "frequency_encode",
                "inputs": [col],
                "input_stage": "pre_fe_raw_categorical",
            })

    for col in list(train_df.columns):
        if col not in test_df.columns:
            test_df[col] = 0
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)
        else:
            train_df[col] = _numeric_series(train_df[col])
            test_df[col] = _numeric_series(test_df[col])
        median = train_df[col].median()
        if pd.isna(median):
            median = 0.0
        train_df[col] = train_df[col].fillna(float(median))
        test_df[col] = test_df[col].fillna(float(median))

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

    return {
        "train_frame": train_df,
        "test_frame": test_df,
        "report": report,
        "lineage": lineage,
    }


def write_deterministic_feature_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    workspace_path,
    *,
    deferred_categorical_columns: dict | None = None,
    blocked_columns: list[str] | None = None,
    fallback_used: bool = True,
    reason: str | None = None,
) -> None:
    workspace = Path(workspace_path)
    result = engineer_deterministic_feature_views(
        train_df,
        test_df,
        deferred_categorical_columns=deferred_categorical_columns,
        blocked_columns=blocked_columns,
        fallback_used=fallback_used,
        reason=reason,
    )
    result["train_frame"].to_csv(workspace / "engineered_train.csv", index=False)
    result["test_frame"].to_csv(workspace / "engineered_test.csv", index=False)
    (workspace / "feature_engineering_report.json").write_text(
        json.dumps(result["report"], indent=2),
        encoding="utf-8",
    )
    (workspace / "feature_lineage.json").write_text(
        json.dumps(result["lineage"], indent=2),
        encoding="utf-8",
    )
