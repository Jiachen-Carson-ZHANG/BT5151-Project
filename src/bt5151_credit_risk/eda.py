import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def build_eda_report(df: pd.DataFrame, target_column: str) -> dict:
    """Programmatic EDA producing a structured report for downstream LLM nodes."""
    report = {}

    target = df[target_column] if target_column in df.columns else None
    feature_cols = [c for c in df.columns if c != target_column]

    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist()

    # 1. Correlation matrix — extract high-correlation pairs (|r| > 0.8)
    report["correlations"] = _compute_correlations(df, numeric_cols)

    # 2. Class-conditional means + ANOVA F-stat
    report["class_separability"] = _compute_class_separability(df, numeric_cols, target)

    # 3. Skewness per numeric column
    report["skewness"] = _compute_skewness(df, numeric_cols)

    # 4. Missing value patterns
    report["missing_patterns"] = _compute_missing_patterns(df, feature_cols, target)

    # 5. Cardinality per categorical
    report["cardinality"] = _compute_cardinality(df, categorical_cols)

    # 6. Top discriminative features (mutual information + F-classif)
    report["top_discriminative_features"] = _compute_discriminative_features(
        df, numeric_cols, target
    )

    return report


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
    df: pd.DataFrame, numeric_cols: list[str], target: pd.Series | None
) -> dict:
    if target is None or len(numeric_cols) == 0:
        return {"class_means": {}, "anova_top_features": []}

    # Class-conditional means
    class_means = {}
    for cls in target.dropna().unique():
        mask = target == cls
        class_means[str(cls)] = {
            col: round(float(df.loc[mask, col].mean()), 4)
            for col in numeric_cols
            if not df.loc[mask, col].isna().all()
        }

    # ANOVA F-stat for numeric features vs target
    anova_features = []
    valid_cols = [c for c in numeric_cols if df[c].notna().sum() > 1]
    if valid_cols:
        # Fill NaN for f_classif (it requires no missing values)
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
    df: pd.DataFrame, numeric_cols: list[str], target: pd.Series | None
) -> list[dict]:
    if target is None or len(numeric_cols) == 0:
        return []

    valid_cols = [c for c in numeric_cols if df[c].notna().sum() > 1]
    if not valid_cols:
        return []

    X = df[valid_cols].fillna(df[valid_cols].median())
    y = target.copy()
    # Encode target if string
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.fillna("Unknown")), index=y.index)
    else:
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

    results = []
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        for col, mi in zip(valid_cols, mi_scores):
            results.append({
                "column": col,
                "mutual_information": round(float(mi), 4),
            })
        results.sort(key=lambda x: x["mutual_information"], reverse=True)
    except Exception:
        pass

    return results[:20]
