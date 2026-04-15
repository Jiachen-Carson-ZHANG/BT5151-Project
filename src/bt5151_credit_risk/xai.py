import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global SHAP (refactored from graph.py select_model_node)
# ---------------------------------------------------------------------------

def compute_global_shap(
    model,
    model_name: str,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    class_names: list[str],
    sample_size: int = 500,
    top_n: int = 15,
) -> dict:
    """Compute mean |SHAP| importance, beeswarm data, and dependence data.

    Returns dict with keys: importance, beeswarm_data, dependence_data.
    """
    import shap

    sample_frame = test_frame.sample(n=min(sample_size, len(test_frame)), random_state=42)

    if model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_frame)
    elif model_name == "logistic_regression":
        scaler = model.named_steps["scaler"]
        lr_model = model.named_steps["model"]
        scaled = scaler.transform(sample_frame)
        explainer = shap.LinearExplainer(lr_model, scaled)
        shap_values = explainer.shap_values(scaled)
    else:
        return {"importance": [], "beeswarm_data": {}, "dependence_data": {}}

    shap_arr = np.array(shap_values)

    # Normalise shape to (n_samples, n_features, n_classes)
    if shap_arr.ndim == 3:
        if shap_arr.shape[0] == len(class_names):
            # (n_classes, n_samples, n_features) -> transpose
            shap_arr = shap_arr.transpose(1, 2, 0)
        elif shap_arr.shape[2] == len(class_names):
            pass  # already (n_samples, n_features, n_classes)
        else:
            # (n_samples, n_features, n_classes) assumed
            pass
    elif shap_arr.ndim == 2:
        # Binary or single-output — expand to 3D
        shap_arr = shap_arr[:, :, np.newaxis]

    # Importance: mean |SHAP| across samples and classes
    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))
    importance_pairs = sorted(zip(feature_columns, mean_abs.tolist()), key=lambda x: x[1], reverse=True)
    importance = [{"feature": f, "mean_abs_shap": round(v, 4)} for f, v in importance_pairs[:top_n]]

    # Beeswarm data: per-sample SHAP values for top features
    top_feature_names = [e["feature"] for e in importance[:10]]
    beeswarm_data = {}
    feature_values = sample_frame.values if model_name != "logistic_regression" else scaled
    for fname in top_feature_names:
        fidx = feature_columns.index(fname)
        beeswarm_data[fname] = {
            "shap_values": shap_arr[:, fidx, :].mean(axis=1).tolist(),  # avg across classes
            "feature_values": feature_values[:, fidx].tolist() if hasattr(feature_values, '__getitem__') else [],
        }

    # Dependence data: for top-3 features, per-class
    dependence_data = {}
    for fname in top_feature_names[:3]:
        fidx = feature_columns.index(fname)
        dep = {}
        for cidx, cname in enumerate(class_names):
            if cidx < shap_arr.shape[2]:
                dep[cname] = {
                    "feature_values": feature_values[:, fidx].tolist() if hasattr(feature_values, '__getitem__') else [],
                    "shap_values": shap_arr[:, fidx, cidx].tolist(),
                }
        dependence_data[fname] = dep

    return {"importance": importance, "beeswarm_data": beeswarm_data, "dependence_data": dependence_data}


# ---------------------------------------------------------------------------
# Grouped Permutation Feature Importance
# ---------------------------------------------------------------------------

def _group_onehot_columns(
    feature_columns: list[str],
    column_transform_spec: dict | None = None,
) -> dict[str, list[int]]:
    """Group one-hot encoded columns by original feature.

    When column_transform_spec is provided, uses authoritative lineage:
    columns marked encoding=one_hot in the spec are grouped by their
    original column name. This correctly handles binary categoricals
    (2 one-hot columns) without false-merging engineered features.

    Fallback (no spec): prefix heuristic with 3+ threshold.
    """
    # Build authoritative one-hot set from the spec
    onehot_originals = set()
    if column_transform_spec:
        transforms = column_transform_spec.get("transforms", column_transform_spec.get("columns", {}))
        for col_name, spec in transforms.items():
            if isinstance(spec, dict) and spec.get("encoding") == "one_hot":
                onehot_originals.add(col_name)

    if onehot_originals:
        # Authoritative grouping: match output columns to known one-hot originals
        groups = {}
        claimed = set()
        for orig in sorted(onehot_originals):
            indices = [
                idx for idx, col in enumerate(feature_columns)
                if col.startswith(orig + "_") or col == orig
            ]
            if len(indices) >= 2:
                groups[orig] = indices
                claimed.update(indices)
            elif len(indices) == 1:
                # Original column survived without encoding (e.g., dropped by FE)
                groups[feature_columns[indices[0]]] = indices
                claimed.update(indices)
        # Remaining columns: each is its own group
        for idx, col in enumerate(feature_columns):
            if idx not in claimed:
                groups[col] = [idx]
        return groups

    # Fallback: prefix heuristic (3+ columns threshold)
    prefix_map = defaultdict(list)
    for idx, col in enumerate(feature_columns):
        if "_" in col:
            prefix = col.rsplit("_", 1)[0]
            prefix_map[prefix].append(idx)
        else:
            prefix_map[col].append(idx)

    groups = {}
    for prefix, indices in prefix_map.items():
        if len(indices) >= 3:
            groups[prefix] = indices
        else:
            for idx in indices:
                groups[feature_columns[idx]] = [idx]
    return groups


def compute_permutation_importance(
    model,
    model_name: str,
    test_frame: pd.DataFrame,
    test_target: pd.Series,
    feature_columns: list[str],
    column_transform_spec: dict | None = None,
    n_repeats: int = 10,
    max_samples: int = 5_000,
) -> dict:
    """Compute raw PFI and true grouped PFI (permutes all one-hot columns in a group together).

    Raw PFI uses sklearn's permutation_importance (per-column).
    Grouped PFI permutes all columns sharing an original feature simultaneously,
    which correctly measures the joint importance of the original feature.

    When column_transform_spec is provided, uses authoritative lineage from
    the spec to identify one-hot groups (including binary categoricals).
    Falls back to prefix heuristic otherwise.

    Returns dict with keys: raw, grouped.
    """
    from sklearn.metrics import f1_score

    # Subsample large test sets — PFI rankings are stable at 5k rows.
    if len(test_frame) > max_samples:
        sample_positions = np.random.RandomState(42).choice(len(test_frame), size=max_samples, replace=False)
        test_frame = test_frame.iloc[sample_positions].copy()
        test_target = test_target.iloc[sample_positions].copy()

    result = permutation_importance(
        model, test_frame, test_target,
        n_repeats=n_repeats, scoring="f1_macro", random_state=42, n_jobs=-1,
    )

    raw = []
    for i, col in enumerate(feature_columns):
        raw.append({
            "feature": col,
            "importance_mean": round(float(result.importances_mean[i]), 4),
            "importance_std": round(float(result.importances_std[i]), 4),
        })
    raw.sort(key=lambda x: x["importance_mean"], reverse=True)

    # True grouped PFI: permute all columns in each group simultaneously
    groups = _group_onehot_columns(feature_columns, column_transform_spec)
    baseline_score = f1_score(test_target, model.predict(test_frame), average="macro", zero_division=0)
    rng = np.random.RandomState(42)

    grouped = []
    for group_name, indices in groups.items():
        repeat_scores = []
        for _ in range(n_repeats):
            X_permuted = test_frame.values.copy()
            perm_idx = rng.permutation(len(X_permuted))
            for col_idx in indices:
                X_permuted[:, col_idx] = X_permuted[perm_idx, col_idx]
            permuted_frame = pd.DataFrame(X_permuted, columns=feature_columns, index=test_frame.index)
            perm_score = f1_score(test_target, model.predict(permuted_frame), average="macro", zero_division=0)
            repeat_scores.append(baseline_score - perm_score)
        grouped.append({
            "original_feature": group_name,
            "importance_mean": round(float(np.mean(repeat_scores)), 4),
            "importance_std": round(float(np.std(repeat_scores)), 4),
            "n_columns": len(indices),
        })
    grouped.sort(key=lambda x: x["importance_mean"], reverse=True)

    return {"raw": raw[:20], "grouped": grouped[:20]}


# ---------------------------------------------------------------------------
# Partial Dependence Plots (per-class probability curves)
# ---------------------------------------------------------------------------

def compute_partial_dependence(
    model,
    model_name: str,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    top_features: list[str],
    class_names: list[str],
    grid_resolution: int = 50,
) -> dict:
    """PDP for top continuous features. Returns per-class probability curves.

    Returns {feature: {grid: [...], pd_values: {class_name: [...]}}}
    """
    results = {}
    for feat in top_features:
        if feat not in feature_columns:
            continue
        feat_idx = feature_columns.index(feat)
        try:
            pdp_frame = test_frame.copy()
            if pd.api.types.is_integer_dtype(pdp_frame.iloc[:, feat_idx]):
                pdp_frame[feat] = pdp_frame.iloc[:, feat_idx].astype(float)
            pd_result = partial_dependence(
                model if model_name != "logistic_regression" else model,
                pdp_frame, features=[feat_idx],
                kind="average", grid_resolution=grid_resolution,
            )
            grid = pd_result["grid_values"][0].tolist()
            avg = pd_result["average"]  # shape (n_classes, n_grid_points) or (1, n_grid_points)
            pd_values = {}
            if avg.ndim == 2 and avg.shape[0] == len(class_names):
                for cidx, cname in enumerate(class_names):
                    pd_values[cname] = [round(float(v), 6) for v in avg[cidx]]
            else:
                pd_values["prediction"] = [round(float(v), 6) for v in avg[0]]
            results[feat] = {"grid": [round(float(v), 4) for v in grid], "pd_values": pd_values}
        except Exception as exc:
            logger.warning("    PDP failed for %s: %s", feat, exc)
    return results


# ---------------------------------------------------------------------------
# Accumulated Local Effects (custom implementation, no alibi)
# ---------------------------------------------------------------------------

def compute_ale(
    model,
    model_name: str,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    top_features: list[str],
    class_names: list[str],
    n_bins: int = 50,
) -> dict:
    """ALE for top continuous features. Custom implementation.

    Algorithm:
    1. Sort data by feature, bin into quantile bins.
    2. For each bin, set feature to upper/lower boundary, compute prediction difference.
    3. Accumulate differences and centre.

    Returns {feature: {bin_centres: [...], ale_values: {class_name: [...]}}}
    """
    results = {}
    for feat in top_features:
        if feat not in feature_columns:
            continue
        feat_idx = feature_columns.index(feat)
        col_values = test_frame.iloc[:, feat_idx].values.astype(float)

        # Quantile bin edges
        unique_vals = np.unique(col_values[~np.isnan(col_values)])
        if len(unique_vals) < 3:
            continue
        percentiles = np.linspace(0, 100, min(n_bins + 1, len(unique_vals)))
        bin_edges = np.unique(np.percentile(unique_vals, percentiles))
        if len(bin_edges) < 3:
            continue

        bin_indices = np.digitize(col_values, bin_edges[1:-1])  # assigns to bins 0..n_bins-1

        try:
            X = test_frame.values.copy()
            n_effective_bins = len(bin_edges) - 1
            # For each bin, compute mean prediction difference
            ale_per_class = np.zeros((n_effective_bins, len(class_names)))

            for b in range(n_effective_bins):
                mask = bin_indices == b
                if mask.sum() == 0:
                    continue
                X_lower = X[mask].copy()
                X_upper = X[mask].copy()
                X_lower[:, feat_idx] = bin_edges[b]
                X_upper[:, feat_idx] = bin_edges[b + 1]

                df_lower = pd.DataFrame(X_lower, columns=feature_columns)
                df_upper = pd.DataFrame(X_upper, columns=feature_columns)

                pred_lower = model.predict_proba(df_lower)
                pred_upper = model.predict_proba(df_upper)
                diff = pred_upper - pred_lower  # (n_samples_in_bin, n_classes)
                ale_per_class[b, :] = diff.mean(axis=0)[:len(class_names)]

            # Accumulate
            ale_cumulative = np.cumsum(ale_per_class, axis=0)
            # Centre: subtract mean ALE
            ale_centred = ale_cumulative - ale_cumulative.mean(axis=0, keepdims=True)

            bin_centres = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_effective_bins)]

            ale_values = {}
            for cidx, cname in enumerate(class_names):
                if cidx < ale_centred.shape[1]:
                    ale_values[cname] = [round(float(v), 6) for v in ale_centred[:, cidx]]

            results[feat] = {
                "bin_centres": [round(float(v), 4) for v in bin_centres],
                "ale_values": ale_values,
            }
        except Exception as exc:
            logger.warning("    ALE failed for %s: %s", feat, exc)

    return results


# ---------------------------------------------------------------------------
# Per-case SHAP (refactored from graph.py _compute_shap_contributions)
# ---------------------------------------------------------------------------

def compute_shap_contributions_for_case(
    model,
    model_name: str,
    input_frame: pd.DataFrame,
    predicted_class_idx: int,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Compute SHAP values for a single prediction row. Returns top-N features."""
    import shap

    if model_name == "logistic_regression":
        scaler = model.named_steps["scaler"]
        lr_model = model.named_steps["model"]
        scaled_input = scaler.transform(input_frame)
        explainer = shap.LinearExplainer(lr_model, scaled_input)
        shap_values = explainer.shap_values(scaled_input)
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        elif np.array(shap_values).ndim == 3:
            sv = np.array(shap_values)
            class_shap = sv[0, :, predicted_class_idx] if sv.shape[0] == 1 else sv[predicted_class_idx][0]
        else:
            class_shap = np.array(shap_values)[0]
    else:
        if model_name in ("xgboost", "random_forest"):
            explainer = shap.TreeExplainer(model)
        else:
            return []
        shap_values = explainer.shap_values(input_frame)
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            class_shap = shap_arr[0, :, predicted_class_idx]
        elif isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx][0]
        else:
            class_shap = shap_arr[0]

    abs_shap = [(feature_names[i], float(class_shap[i])) for i in range(len(feature_names))]
    abs_shap.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {"feature": f, "shap_value": round(v, 4), "direction": "positive" if v > 0 else "negative"}
        for f, v in abs_shap[:top_n]
    ]


# ---------------------------------------------------------------------------
# Classification Casebook Strategy
# ---------------------------------------------------------------------------

def select_classification_cases(
    test_frame: pd.DataFrame,
    test_target: pd.Series,
    model,
    class_names: list[str],
    id_to_label: dict,
) -> list[dict]:
    """Select representative, worst misclassification, and borderline cases per class.

    Returns list of dicts with: row_index, true_label, predicted_label,
    probabilities, case_type, confused_with_class.
    """
    probas = model.predict_proba(test_frame)
    preds = model.predict(test_frame)
    cases = []

    for class_idx, class_name in enumerate(class_names):
        true_mask = (test_target.values == class_idx)
        if not true_mask.any():
            continue

        class_probas = probas[true_mask, class_idx]
        class_indices = test_frame.index[true_mask]
        class_preds = preds[true_mask]

        correct_mask = (class_preds == class_idx)
        wrong_mask = ~correct_mask

        # Representative: most confident correct prediction
        if correct_mask.any():
            best_idx = np.argmax(class_probas[correct_mask])
            row_idx = int(class_indices[correct_mask][best_idx])
            cases.append(_build_case_metadata(
                row_idx, class_name, id_to_label[int(preds[true_mask][correct_mask][best_idx])],
                probas[true_mask][correct_mask][best_idx], class_names, id_to_label,
                "representative", None,
            ))

        # Borderline: least confident correct prediction
        if correct_mask.any() and correct_mask.sum() > 1:
            worst_correct_idx = np.argmin(class_probas[correct_mask])
            row_idx = int(class_indices[correct_mask][worst_correct_idx])
            cases.append(_build_case_metadata(
                row_idx, class_name, id_to_label[int(preds[true_mask][correct_mask][worst_correct_idx])],
                probas[true_mask][correct_mask][worst_correct_idx], class_names, id_to_label,
                "borderline", None,
            ))

        # Worst misclassification: most confident wrong prediction
        # Select by highest predicted-wrong-class probability, not lowest true-class probability.
        # In multiclass, these differ: a sample can have low true-class prob spread across
        # many wrong classes, vs one with high confidence in a specific wrong class.
        if wrong_mask.any():
            wrong_preds = preds[true_mask][wrong_mask]
            wrong_pred_confidence = np.array([
                probas[true_mask][wrong_mask][i, int(wrong_preds[i])]
                for i in range(wrong_mask.sum())
            ])
            worst_idx = np.argmax(wrong_pred_confidence)
            row_idx = int(class_indices[wrong_mask][worst_idx])
            predicted_class = int(wrong_preds[worst_idx])
            cases.append(_build_case_metadata(
                row_idx, class_name, id_to_label[predicted_class],
                probas[true_mask][wrong_mask][worst_idx], class_names, id_to_label,
                "worst_misclassification", id_to_label[predicted_class],
            ))

    return cases


def _build_case_metadata(
    row_index, true_label, predicted_label, proba_array,
    class_names, id_to_label, case_type, confused_with,
):
    probability_map = {id_to_label[idx]: round(float(p), 4) for idx, p in enumerate(proba_array)}
    return {
        "row_index": row_index,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "probabilities": probability_map,
        "case_type": case_type,
        "confused_with_class": confused_with,
    }
