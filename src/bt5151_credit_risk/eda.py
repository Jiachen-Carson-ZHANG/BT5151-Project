import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from bt5151_credit_risk.feature_eligibility import apply_feature_eligibility

logger = logging.getLogger(__name__)


def build_eda_report(
    df: pd.DataFrame,
    target_column: str,
    dataset_policy_spec: dict | None = None,
) -> dict:
    """Programmatic EDA producing a structured report for downstream LLM nodes."""
    report = {}

    target = df[target_column] if target_column in df.columns else None
    feature_cols = [c for c in df.columns if c != target_column]

    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist()

    # 1. Correlation matrix — extract high-correlation pairs (|r| > 0.8)
    report["correlations"] = _compute_correlations(df, numeric_cols)

    # 2. Class-conditional means + ANOVA F-stat (numeric) + class-conditional mode (categorical)
    report["class_separability"] = _compute_class_separability(
        df, numeric_cols, categorical_cols, target
    )

    # 3. Skewness per numeric column
    report["skewness"] = _compute_skewness(df, numeric_cols)

    # 4. Missing value patterns
    report["missing_patterns"] = _compute_missing_patterns(df, feature_cols, target)

    # 5. Cardinality per categorical
    report["cardinality"] = _compute_cardinality(df, categorical_cols)

    # 6. Top discriminative features (mutual information + F-classif)
    #    Includes label-encoded categoricals so ordinal features like Credit_Mix
    #    and Month are visible alongside numerics.
    raw_top_discriminative_features = _compute_discriminative_features(
        df, numeric_cols, categorical_cols, target
    )
    eligibility = apply_feature_eligibility(
        raw_top_features=raw_top_discriminative_features,
        feature_profiles=_build_feature_profiles(df, feature_cols),
        dataset_policy_spec=dataset_policy_spec,
    )
    report["raw_top_discriminative_features"] = eligibility["raw_top_discriminative_features"]
    report["model_eligible_top_discriminative_features"] = eligibility["model_eligible_top_discriminative_features"]
    report["top_discriminative_features"] = eligibility["model_eligible_top_discriminative_features"]
    report["leakage_alerts"] = eligibility["leakage_alerts"]
    report["feature_eligibility"] = eligibility["feature_decisions"]

    # 7. Categorical association with target (Cramér's V + chi-squared)
    report["categorical_association"] = _compute_categorical_association(
        df, categorical_cols, target
    )

    return report


def _build_feature_profiles(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for col in feature_cols:
        series = df[col]
        profiles[col] = {
            "nunique": int(series.nunique(dropna=True)),
            "non_null_count": int(series.notna().sum()),
        }
    return profiles


def _compute_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    if len(numeric_cols) < 2:
        return {"high_pairs": [], "matrix_shape": [0, 0]}

    corr = df[numeric_cols].corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and abs(val) > 0.8:
                high_pairs.append({
                    "col_a": idx,
                    "col_b": col,
                    "correlation": round(float(val), 4),
                })
    high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return {
        "high_pairs": high_pairs[:30],
        "matrix_shape": list(corr.shape),
    }


def _compute_class_separability(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: pd.Series | None,
) -> dict:
    if target is None:
        return {"class_means": {}, "anova_top_features": [], "categorical_class_modes": {}}

    # Class-conditional means for numerics
    class_means = {}
    for cls in target.dropna().unique():
        mask = target == cls
        class_means[str(cls)] = {
            col: round(float(df.loc[mask, col].mean()), 4)
            for col in numeric_cols
            if not df.loc[mask, col].isna().all()
        }

    # Class-conditional mode distribution for categoricals (top-2 values per class)
    categorical_class_modes: dict = {}
    for col in categorical_cols:
        col_dist: dict = {}
        for cls in target.dropna().unique():
            mask = target == cls
            vc = df.loc[mask, col].value_counts(normalize=True)
            col_dist[str(cls)] = {
                str(k): round(float(v), 4) for k, v in vc.head(3).items()
            }
        categorical_class_modes[col] = col_dist

    # ANOVA F-stat for numeric features vs target
    anova_features = []
    valid_cols = [c for c in numeric_cols if df[c].notna().sum() > 1]
    if valid_cols:
        X = df[valid_cols].fillna(df[valid_cols].median())
        y = target.fillna(target.mode().iloc[0] if not target.mode().empty else "Unknown")
        try:
            f_stats, p_values = f_classif(X, y)
            for col, f_val, p_val in zip(valid_cols, f_stats, p_values):
                anova_features.append({
                    "column": col,
                    "f_statistic": round(float(f_val), 2),
                    "p_value": round(float(p_val), 6),
                })
            anova_features.sort(key=lambda x: x["f_statistic"], reverse=True)
        except Exception:
            pass

    return {
        "class_means": class_means,
        "anova_top_features": anova_features[:20],
        "categorical_class_modes": categorical_class_modes,
    }


def _compute_skewness(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    skewness = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 1:
            skewness[col] = round(float(s.skew()), 4)
    highly_skewed = {k: v for k, v in skewness.items() if abs(v) > 2}
    return {
        "all": skewness,
        "highly_skewed": highly_skewed,
    }


def _compute_missing_patterns(
    df: pd.DataFrame, feature_cols: list[str], target: pd.Series | None
) -> dict:
    missing_pct = {}
    for col in feature_cols:
        pct = float(df[col].isna().mean())
        if pct > 0:
            missing_pct[col] = round(pct, 4)

    # Check if missingness correlates with target (MNAR detection)
    mnar_suspects = []
    if target is not None:
        for col, pct in missing_pct.items():
            if pct < 0.01 or pct > 0.5:
                continue
            missing_mask = df[col].isna()
            target_dist_missing = target[missing_mask].value_counts(normalize=True)
            target_dist_present = target[~missing_mask].value_counts(normalize=True)
            # Compare distributions — large KL divergence suggests MNAR
            all_classes = set(target_dist_missing.index) | set(target_dist_present.index)
            max_diff = 0.0
            for cls in all_classes:
                p_miss = target_dist_missing.get(cls, 0)
                p_pres = target_dist_present.get(cls, 0)
                max_diff = max(max_diff, abs(p_miss - p_pres))
            if max_diff > 0.05:
                mnar_suspects.append({
                    "column": col,
                    "missing_pct": pct,
                    "max_target_dist_diff": round(max_diff, 4),
                })

    return {
        "missing_pct": missing_pct,
        "mnar_suspects": mnar_suspects,
    }


def _compute_cardinality(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    cardinality = {}
    high_cardinality = []
    for col in categorical_cols:
        nunique = int(df[col].nunique())
        cardinality[col] = nunique
        if nunique > 20:
            high_cardinality.append({"column": col, "nunique": nunique})
    high_cardinality.sort(key=lambda x: x["nunique"], reverse=True)
    return {
        "all": cardinality,
        "high_cardinality": high_cardinality,
    }


def _compute_discriminative_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: pd.Series | None,
) -> list[dict]:
    """Compute mutual information for both numeric and categorical columns.

    Categorical columns are label-encoded before MI computation so that ordinal
    features (Credit_Mix, Month, Payment_of_Min_Amount, etc.) appear in the
    same unified ranking as numerics. Each result is tagged with its column type
    so the LLM knows which columns needed encoding.
    """
    if target is None:
        return []

    y = target.copy()
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.fillna("Unknown")), index=y.index)
    else:
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

    # Build feature matrix: numerics filled with median, categoricals label-encoded
    X_parts = {}
    col_types = {}

    valid_numeric = [c for c in numeric_cols if df[c].notna().sum() > 1]
    for col in valid_numeric:
        X_parts[col] = df[col].fillna(df[col].median())
        col_types[col] = "numeric"

    for col in categorical_cols:
        if df[col].notna().sum() < 2:
            continue
        le_col = LabelEncoder()
        encoded = le_col.fit_transform(df[col].fillna("__missing__").astype(str))
        X_parts[col] = pd.Series(encoded, index=df.index)
        col_types[col] = "categorical"

    if not X_parts:
        return []

    X = pd.DataFrame(X_parts)
    results = []
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        for col, mi in zip(X.columns, mi_scores):
            results.append({
                "column": col,
                "mutual_information": round(float(mi), 4),
                "column_type": col_types[col],
            })
        results.sort(key=lambda x: x["mutual_information"], reverse=True)
    except Exception:
        pass

    return results[:30]


def _compute_categorical_association(
    df: pd.DataFrame, categorical_cols: list[str], target: pd.Series | None
) -> list[dict]:
    """Cramér's V association between each categorical column and the target.

    Complements MI with a scale-free association measure that does not require
    encoding. Useful for the LLM to understand which categorical features are
    most strongly linked to the target distribution.
    """
    if target is None or len(categorical_cols) == 0:
        return []

    results = []
    for col in categorical_cols:
        try:
            ct = pd.crosstab(df[col].fillna("__missing__"), target)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, _, _ = chi2_contingency(ct)
            n = ct.values.sum()
            k = min(ct.shape)
            cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if n > 0 and k > 1 else 0.0
            results.append({
                "column": col,
                "cramers_v": round(cramers_v, 4),
                "chi2": round(float(chi2), 2),
                "p_value": round(float(p), 6),
                "n_categories": int(df[col].nunique()),
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["cramers_v"], reverse=True)
    return results
