from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MISSING_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "not specified",
    "not_specified",
}


def allowed_cleaning_primitives() -> dict[str, dict[str, str | list[str]]]:
    return {
        "parse_dirty_numeric": {
            "purpose": "Strip non-numeric artifacts and coerce to float.",
            "use_when": ["money-like strings", "counts with underscores", "rates with symbols"],
        },
        "parse_age_series": {
            "purpose": "Extract the first numeric age token from dirty age strings.",
            "use_when": ["human age stored as text or mixed-format strings"],
        },
        "parse_duration_months": {
            "purpose": "Convert 'X Years and Y Months' style durations into total months.",
            "use_when": ["credit history age", "tenure or duration strings"],
        },
        "missing_string_mask": {
            "purpose": "Normalize common missing string sentinels into a boolean missing mask.",
            "use_when": ["text columns with '', 'nan', 'null', 'Not Specified'"],
        },
        "multi_hot_membership": {
            "purpose": "Split delimited multi-value strings into binary membership indicators without changing row count.",
            "use_when": ["loan type lists", "symptom lists", "tag-like categorical sets"],
        },
        "fill_numeric_by_group_then_global": {
            "purpose": "Impute numeric values from group medians before a global fallback.",
            "use_when": ["entity-stable numeric fields with group structure"],
        },
        "cap_credit_history_by_adulthood": {
            "purpose": "Enforce that credit-history months do not exceed plausible adult credit tenure.",
            "use_when": ["credit history age paired with applicant age"],
        },
    }


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def parse_dirty_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"[^0-9.\-]+", "", regex=True)
        .replace("", pd.NA)
    )
    return coerce_numeric(cleaned)


def parse_money_series(series: pd.Series) -> pd.Series:
    return parse_dirty_numeric(series)


def parse_age_series(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    return coerce_numeric(extracted)


def parse_count_series(series: pd.Series) -> pd.Series:
    return parse_dirty_numeric(series)


def parse_rate_series(series: pd.Series) -> pd.Series:
    return parse_dirty_numeric(series)


def missing_string_mask(series: pd.Series, extra_tokens: Iterable[str] | None = None) -> pd.Series:
    as_string = series.astype("string")
    normalized = as_string.str.strip().str.lower()
    tokens = set(DEFAULT_MISSING_TOKENS)
    if extra_tokens:
        tokens.update(str(token).strip().lower() for token in extra_tokens)
    return as_string.isna() | normalized.isin(tokens)


def parse_duration_months(series: pd.Series) -> pd.Series:
    as_string = series.astype("string")
    extracted = as_string.str.extract(
        r"(?i)^\s*(?P<years>\d+)\s*years?(?:\s*(?:and|&)?\s*(?P<months>\d+)\s*months?)?\s*$"
    )
    years = pd.to_numeric(extracted["years"], errors="coerce")
    months = pd.to_numeric(extracted["months"], errors="coerce").fillna(0)
    parsed = years * 12 + months
    numeric_fallback = pd.to_numeric(as_string, errors="coerce")
    return coerce_numeric(parsed.where(parsed.notna(), numeric_fallback))


def fill_numeric_by_group_then_global(
    values: pd.Series,
    groups: pd.Series | None,
    default: float,
) -> pd.Series:
    base = coerce_numeric(values)
    filled = base.copy()
    if groups is not None:
        grouped_median = base.groupby(groups).transform("median").astype("float64")
        filled = filled.fillna(grouped_median)
    global_fill = float(base.median()) if base.notna().any() else float(default)
    return filled.fillna(global_fill)


def cap_credit_history_by_adulthood(months: pd.Series, age_series: pd.Series | None) -> pd.Series:
    if age_series is None:
        return months
    age_numeric = coerce_numeric(age_series)
    adulthood_cap = np.floor((age_numeric - 18).clip(lower=0) * 12 * 1.1)
    capped = months.copy()
    mask = capped.notna() & adulthood_cap.notna()
    capped.loc[mask] = np.minimum(capped.loc[mask], adulthood_cap.loc[mask])
    return capped


def multi_hot_membership(
    series: pd.Series,
    *,
    column_name: str,
    lowercase: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    missing = missing_string_mask(series).astype(int)
    cleaned = series.astype("string").fillna("")
    cleaned = cleaned.str.replace(r"\band\b", ",", regex=True, flags=re.IGNORECASE)
    dummies = cleaned.str.get_dummies(sep=",")
    dummies.columns = [c.strip() for c in dummies.columns]
    if lowercase:
        dummies.columns = [c.lower() for c in dummies.columns]
    if not dummies.empty:
        dummies = dummies.T.groupby(level=0).max().T
        dummies = dummies.loc[:, dummies.columns.str.strip() != ""]
        dummies.columns = [f"{column_name}_{c}" for c in dummies.columns]
        dummies = dummies.astype("int64")
        dummies.loc[missing.astype(bool), :] = 0
    return dummies, missing
