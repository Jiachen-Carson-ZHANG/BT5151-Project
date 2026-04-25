from __future__ import annotations

import re

import numpy as np
import pandas as pd


_ABSENT_ROLES = {"identifier", "group_identifier", "target", "leakage_risk_feature"}
_PREFIX_ROLES = {"multi_value_set", "unordered_categorical", "free_text"}


def _unique_non_null(series: pd.Series) -> set:
    vals = series.dropna().unique()
    return set(vals.tolist()) if len(vals) else set()


def _is_binary_set(values: set) -> bool:
    return all(float(v) in (0.0, 1.0) for v in values if not isinstance(v, bool))


def validate_semantic_roles(
    feature_frame: pd.DataFrame,
    column_transform_spec: dict,
) -> list[dict]:
    raw_spec = column_transform_spec or {}
    transforms = raw_spec.get("transforms") or raw_spec.get("columns") or {}
    frame_cols = list(feature_frame.columns)
    frame_col_set = set(frame_cols)
    violations: list[dict] = []

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
                        column=m,
                        role=role,
                        violation="not_binary",
                        observed=f"values include {bad}" if bad else f"dtype={series.dtype}",
                        expected="values ⊆ {0, 1}",
                        likely_cause="binary_flag must be mapped to {0,1} numerically; check for unmapped string values or dtype=object.",
                    )

            elif role == "multi_value_set":
                if m == col:
                    record(
                        column=m,
                        role=role,
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
                            column=m,
                            role=role,
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
                            column=m,
                            role=role,
                            violation="not_ordinal_encoded",
                            observed=f"values={sorted(uniq)[:10]}",
                            expected="integer codes 0..K-1 preserving declared order",
                            likely_cause="ordered_categorical should be mapped to integer ordinal codes preserving semantic order, not scaled or one-hot encoded.",
                        )
                    else:
                        int_vals = sorted(int(float(v)) for v in uniq)
                        expected_seq = list(range(len(int_vals)))
                        if int_vals != expected_seq:
                            record(
                                column=m,
                                role=role,
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
                    if m == col and numeric:
                        record(
                            column=m,
                            role=role,
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
                    if m == col and not numeric:
                        record(
                            column=m,
                            role=role,
                            violation="unordered_categorical_not_encoded",
                            observed="dtype=object (raw string)",
                            expected=f"encoded as numeric per intent='{intent}'",
                            likely_cause=(
                                f"unordered_categorical with intent='{intent}' must be encoded "
                                "to numeric before saving feature_frame.csv. "
                                "For one_hot: use str.get_dummies and drop the original. "
                                "For frequency_encoded: replace with frequency counts. "
                                "Only intent='deferred' may remain as object dtype."
                            ),
                        )
                    if intent == "one_hot" and m != col and numeric and not _is_binary_set(uniq):
                        bad = sorted([v for v in uniq if float(v) not in (0.0, 1.0)])[:5]
                        record(
                            column=m,
                            role=role,
                            violation="one_hot_not_binary",
                            observed=f"values include {bad}",
                            expected="values ⊆ {0, 1}",
                            likely_cause="one-hot indicator is not binary; duplicate categories may have been summed. Strip category labels and deduplicate before pivoting.",
                        )

            elif role in {"numeric_count", "numeric_continuous"}:
                if not numeric:
                    record(
                        column=m,
                        role=role,
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

                if role == "numeric_count" and (series.dropna() < 0).any():
                    record(
                        column=m,
                        role=role,
                        violation="negative_count",
                        observed=f"min={float(series.min())}",
                        expected="values >= 0",
                        likely_cause="count values must be non-negative; clip lower bound to 0 or check parsing (a '-' in the raw string may have been preserved).",
                    )

                if role == "numeric_continuous":
                    if series.isna().any():
                        record(
                            column=m,
                            role=role,
                            violation="has_nan",
                            observed=f"{int(series.isna().sum())} NaN values",
                            expected="no NaN (imputation should fill)",
                            likely_cause="imputation step did not cover this column; verify the imputation branch runs after cleaning.",
                        )
                    if np.isinf(series).any():
                        record(
                            column=m,
                            role=role,
                            violation="has_inf",
                            observed=f"{int(np.isinf(series).sum())} inf/-inf values",
                            expected="finite values only (no inf/-inf)",
                            likely_cause=(
                                "a transform produced infinity — common causes: "
                                "division by zero, log(0), or inverse of near-zero values. "
                                "Replace inf with np.nan then impute, or clip before the transform."
                            ),
                        )

            elif role == "temporal_feature":
                if m == col and not numeric and series.dtype == object:
                    record(
                        column=m,
                        role=role,
                        violation="not_decomposed",
                        observed="original temporal column present as dtype=object",
                        expected="decomposed into numeric features (year/month/dow/etc.) per representation_intent",
                        likely_cause="temporal_feature must be parsed with pd.to_datetime and decomposed into numeric components; drop the original string column after decomposition.",
                    )

    return violations


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
                                violation="credit_history_exceeds_adulthood_cap",
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
            base = c[:-len("_missing")]
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
