"""Shortcut-feature audit: deterministic verdicts on suspect top-ranked features.

A feature is flagged when any of three signals fire:
  1. SHAP rank <= 5 AND grouped-PFI rank > 10 (divergence)
  2. mean|SHAP| share of a single feature exceeds 20% of the top-10 SHAP total
     (dominance)
  3. Feature name matches a calendar / row-index pattern and appears in SHAP top-10
     (calendar shortcut)

For up to two suspects per run we run a capped ablation using the fitted model
with the feature zeroed out on the test view (no retraining). Deltas against
the full-model macro_f1 produce one of three verdicts:
  - weak_signal   (|delta| < 0.005)
  - real_signal   (delta < -0.02)
  - inconclusive  (everything in between)

The node writes shortcut_audit.json and stores a compact summary in state.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


CALENDAR_PATTERNS = [
    re.compile(r"^Month$", re.IGNORECASE),
    re.compile(r"^Day.*$", re.IGNORECASE),
    re.compile(r"^Year$", re.IGNORECASE),
    re.compile(r"^Index$", re.IGNORECASE),
    re.compile(r"^Week$", re.IGNORECASE),
    re.compile(r"^Quarter$", re.IGNORECASE),
]


def _calendar_match(name: str) -> bool:
    return any(p.match(name) for p in CALENDAR_PATTERNS)


def detect_shortcut_suspects(
    shap_importance: list[dict],
    pfi_grouped: list[dict],
) -> list[dict]:
    """Return suspect entries with {feature, signals, shap_rank, pfi_rank, shap_share}.

    shap_importance: list of {"feature": str, "mean_abs_shap": float}, pre-sorted desc.
    pfi_grouped:     list of {"original_feature": str, "importance_mean": float}, pre-sorted desc.
    """
    shap_rank = {e["feature"]: i + 1 for i, e in enumerate(shap_importance)}
    pfi_rank = {e["original_feature"]: i + 1 for i, e in enumerate(pfi_grouped)}

    top10_total = sum(float(e["mean_abs_shap"]) for e in shap_importance[:10])

    suspects: dict[str, dict] = {}

    def add(feature: str, signal: str) -> None:
        if feature not in suspects:
            s_rank = shap_rank.get(feature)
            p_rank = pfi_rank.get(feature)
            shap_val = next(
                (float(e["mean_abs_shap"]) for e in shap_importance if e["feature"] == feature),
                0.0,
            )
            share = (shap_val / top10_total) if top10_total > 0 else 0.0
            suspects[feature] = {
                "feature": feature,
                "signals": [],
                "shap_rank": s_rank,
                "pfi_rank": p_rank,
                "shap_share": round(share, 4),
            }
        if signal not in suspects[feature]["signals"]:
            suspects[feature]["signals"].append(signal)

    # Signal 1: SHAP/PFI divergence among SHAP top-5
    for entry in shap_importance[:5]:
        f = entry["feature"]
        pr = pfi_rank.get(f)
        if pr is None or pr > 10:
            add(f, "shap_pfi_divergence")

    # Signal 2: dominance — single feature >20% of top-10 mean|SHAP|
    if top10_total > 0:
        for entry in shap_importance[:10]:
            share = float(entry["mean_abs_shap"]) / top10_total
            if share > 0.20:
                add(entry["feature"], "dominance")

    # Signal 3: calendar/index shortcut
    for entry in shap_importance[:10]:
        if _calendar_match(entry["feature"]):
            add(entry["feature"], "calendar_shortcut")

    # Severity: divergence + calendar + dominance all contribute; prefer suspects
    # with more signals and lower PFI rank (=worse divergence).
    def severity(s: dict) -> tuple:
        n_signals = len(s["signals"])
        pfi_penalty = s["pfi_rank"] if s["pfi_rank"] is not None else 999
        return (-n_signals, -pfi_penalty, -s["shap_share"])

    return sorted(suspects.values(), key=severity)


def _zero_feature_in_view(view: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Return a copy of `view` with `feature` replaced by its median (numeric)
    or 0 (non-numeric). Tree and linear models both accept this shape; we avoid
    dropping columns so the fitted model's feature order is preserved."""
    out = view.copy()
    if feature not in out.columns:
        return out
    col = out[feature]
    if pd.api.types.is_numeric_dtype(col):
        out[feature] = float(col.median()) if col.notna().any() else 0.0
    else:
        out[feature] = 0
    return out


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def ablate_suspects(
    model: Any,
    test_view: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    suspects: list[dict],
    class_names: list[str] | None = None,
    max_ablations: int = 2,
) -> list[dict]:
    """Zero-out ablation for up to max_ablations suspects. Returns list of
    {feature, delta_macro_f1, delta_per_class_recall, verdict, baseline_macro_f1, ablated_macro_f1}.
    """
    if not suspects:
        return []

    baseline_pred = _predict(model, test_view)
    baseline_macro = float(f1_score(y_test, baseline_pred, average="macro"))
    baseline_recall = recall_score(y_test, baseline_pred, average=None, zero_division=0)

    results: list[dict] = []
    for suspect in suspects[:max_ablations]:
        feature = suspect["feature"]
        if feature not in test_view.columns:
            results.append({
                "feature": feature,
                "delta_macro_f1": None,
                "delta_per_class_recall": None,
                "verdict": "not_in_view",
                "baseline_macro_f1": round(baseline_macro, 4),
                "ablated_macro_f1": None,
            })
            continue

        ablated_view = _zero_feature_in_view(test_view, feature)
        try:
            ablated_pred = _predict(model, ablated_view)
        except Exception as exc:  # pragma: no cover - predict should not fail here
            results.append({
                "feature": feature,
                "delta_macro_f1": None,
                "delta_per_class_recall": None,
                "verdict": "ablation_error",
                "error": str(exc)[:200],
                "baseline_macro_f1": round(baseline_macro, 4),
                "ablated_macro_f1": None,
            })
            continue

        ablated_macro = float(f1_score(y_test, ablated_pred, average="macro"))
        ablated_recall = recall_score(y_test, ablated_pred, average=None, zero_division=0)
        delta_macro = ablated_macro - baseline_macro

        if class_names and len(class_names) == len(ablated_recall):
            delta_recall = {
                class_names[i]: round(float(ablated_recall[i] - baseline_recall[i]), 4)
                for i in range(len(class_names))
            }
        else:
            delta_recall = [
                round(float(ablated_recall[i] - baseline_recall[i]), 4)
                for i in range(len(ablated_recall))
            ]

        if abs(delta_macro) < 0.005:
            verdict = "weak_signal"
        elif delta_macro < -0.02:
            verdict = "real_signal"
        else:
            verdict = "inconclusive"

        results.append({
            "feature": feature,
            "signals": suspect.get("signals", []),
            "baseline_macro_f1": round(baseline_macro, 4),
            "ablated_macro_f1": round(ablated_macro, 4),
            "delta_macro_f1": round(delta_macro, 4),
            "delta_per_class_recall": delta_recall,
            "verdict": verdict,
        })

    return results


def run_shortcut_audit(
    model: Any,
    test_view: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    shap_importance: list[dict],
    pfi_grouped: list[dict],
    class_names: list[str] | None = None,
    max_ablations: int = 2,
) -> dict:
    """Top-level entrypoint: detect + ablate. Returns a dict ready to persist."""
    suspects = detect_shortcut_suspects(shap_importance, pfi_grouped)
    ablations = ablate_suspects(
        model, test_view, y_test, suspects,
        class_names=class_names, max_ablations=max_ablations,
    )
    return {
        "suspects": suspects,
        "ablations": ablations,
        "max_ablations": max_ablations,
    }
