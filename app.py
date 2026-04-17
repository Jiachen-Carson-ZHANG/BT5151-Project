"""Credit Risk Analysis — Gradio Demo App.

Three-tab UI:
  Tab 1 — Business View:   Business-first prediction for a selected customer.
  Tab 2 — Model Evidence:  Evaluation metrics, confusion matrix, global XAI,
                            hypothesis validation table.
  Tab 3 — Developer Trace: Live log watch during training; historical trace
                            from cached run when idle.

Quick start:
    # 1. Train the pipeline and save its state:
    PYTHONPATH=src .venv/bin/python run_stage.py full 42 --save-cache

    # 2. Launch the app:
    PYTHONPATH=src .venv/bin/python app.py
"""

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Global pipeline state — one source of truth for completed runs
# ---------------------------------------------------------------------------
_state = None
_state_lock = threading.Lock()
_selected_cache_path: str | None = None  # None → use default CACHE_FILE

# Track last-seen active run status for cache-reload trigger
_last_active_run_status: str | None = None

# Track trace artifact state to avoid re-parsing unchanged files every poll tick
_last_trace_artifact_path: str | None = None
_last_trace_artifact_size: int = 0
_last_trace_md: str = ""

# Cache for model evidence tab — split into summary vs heavy charts so the browser
# does not need to mount every artifact from one callback.
_model_overview_summary_cache: tuple | None = None
_model_overview_summary_cache_run_id: str | None = None
_model_overview_charts_cache: tuple | None = None
_model_overview_charts_cache_run_id: str | None = None
_model_overview_chart_cache: dict[tuple[str, str], object] = {}

RISK_COLOR = {"Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c"}
CAUTION_EMOJI = {"low": "✅", "medium": "⚠️", "high": "🔴"}

# ---------------------------------------------------------------------------
# Skills browser — load all skills/*.md once at startup
# ---------------------------------------------------------------------------
def _load_skill_files() -> dict[str, str]:
    skills_dir = Path(__file__).resolve().parent / "skills"
    result: dict[str, str] = {}
    if skills_dir.is_dir():
        for md_file in sorted(skills_dir.glob("*.md")):
            try:
                result[md_file.name] = md_file.read_text(encoding="utf-8")
            except Exception:
                pass
    return result

_SKILL_FILES: dict[str, str] = _load_skill_files()


def _skill_display_label(filename: str, raw: str) -> str:
    """Derive a capitalized display label from the skill's frontmatter name."""
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 2:
            for line in parts[1].splitlines():
                if line.startswith("name:"):
                    name = line[5:].strip()
                    return " ".join(w.capitalize() for w in name.replace("-", " ").split())
    name = filename.removesuffix(".md")
    return " ".join(w.capitalize() for w in name.replace("-", " ").split())


# Bidirectional maps: filename ↔ display label for the Skills tab dropdown
_SKILL_LABEL: dict[str, str] = {
    fn: _skill_display_label(fn, raw) for fn, raw in _SKILL_FILES.items()
}
_SKILL_FILE_BY_LABEL: dict[str, str] = {v: k for k, v in _SKILL_LABEL.items()}


def _historical_log_dir() -> Path:
    """Directory containing persisted run logs and structured traces."""
    return Path(__file__).resolve().parent / "lab" / "logs"


def _fig_to_image_file(fig: plt.Figure) -> str:
    """Render a matplotlib Figure to a temporary PNG file and close it.

    Tab 2 is an evidence dashboard, not an interactive plotting surface. File-backed
    PNGs are much more robust than returning several live gr.Plot objects in one
    callback, especially on slower browsers where plot hydration can stall the UI.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _get_state():
    global _state
    with _state_lock:
        if _state is None:
            from bt5151_credit_risk.cache import load_cache
            _state = load_cache(_selected_cache_path)
        return _state


def _invalidate_cache() -> None:
    """Clear the in-memory state and pre-warm the cache in a background thread.

    Without pre-warming: the next UI callback (e.g. "Load / Refresh Evidence")
    has to call load_cache() synchronously, which blocks for ~4-5 seconds on the
    Gradio callback thread, causing a WebSocket timeout and browser freeze.

    Pre-warming kicks off load_cache() immediately in a daemon thread so the cache
    is already warm by the time the user clicks anything.
    """
    global _state, _selected_cache_path
    global _model_overview_summary_cache, _model_overview_summary_cache_run_id
    global _model_overview_charts_cache, _model_overview_charts_cache_run_id
    global _model_overview_chart_cache
    # Reset to None so _get_state() loads CACHE_FILE (always the newest save).
    _selected_cache_path = None
    with _state_lock:
        _state = None
    _model_overview_summary_cache = None
    _model_overview_summary_cache_run_id = None
    _model_overview_charts_cache = None
    _model_overview_charts_cache_run_id = None
    _model_overview_chart_cache = {}
    # Reload cache in background immediately — avoids blocking UI callbacks later
    threading.Thread(target=_get_state, daemon=True).start()


def _cache_ready() -> bool:
    from bt5151_credit_risk.cache import CACHE_FILE
    if _selected_cache_path:
        return Path(_selected_cache_path).is_file()
    return CACHE_FILE.is_file()


def _list_cache_choices() -> list[tuple[str, str]]:
    """Return (label, path_str) pairs for all named caches, newest first."""
    from bt5151_credit_risk.cache import list_caches, CACHE_FILE
    choices: list[tuple[str, str]] = []
    for p in list_caches():
        stem = p.stem.replace("pipeline_state_", "", 1)  # "42_20260417"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            run_id, date = parts
            if len(date) == 8 and date.isdigit():
                date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
            label = f"Run {run_id}  ·  {date}"
        else:
            label = stem or p.name
        choices.append((label, str(p)))
    if not choices and CACHE_FILE.is_file():
        choices.append(("Active cached model", str(CACHE_FILE)))
    return choices


def cb_switch_cache(path: str) -> str:
    """Switch active cache to the selected file and reload state."""
    global _state, _selected_cache_path
    global _model_overview_summary_cache, _model_overview_summary_cache_run_id
    global _model_overview_charts_cache, _model_overview_charts_cache_run_id
    global _model_overview_chart_cache
    if not path:
        return "⚠️ No cache selected."
    _selected_cache_path = path
    with _state_lock:
        _state = None
    _model_overview_summary_cache = None
    _model_overview_summary_cache_run_id = None
    _model_overview_charts_cache = None
    _model_overview_charts_cache_run_id = None
    _model_overview_chart_cache = {}
    threading.Thread(target=_get_state, daemon=True).start()
    state = _get_state()
    if state is None:
        return "⚠️ Failed to load selected cache."
    return (
        f"✅ Cache loaded — model: **{state.selected_model_name}**  |  run: `{state.run_id or '?'}`"
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _shap_fig(top_features: list) -> plt.Figure:
    entries = top_features[:10]
    features = [f["feature"] for f in entries]
    values = [float(f["shap_value"]) for f in entries]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(features) * 0.4)))
    ax.barh(features[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value  (red = toward class   blue = away)")
    ax.set_title("Feature contributions to this prediction", pad=6)
    fig.tight_layout(pad=0.8)
    return fig


def _global_shap_fig(importance: list) -> plt.Figure:
    entries = importance[:12]
    features = [e["feature"] for e in entries]
    values = [float(e["mean_abs_shap"]) for e in entries]
    fig, ax = plt.subplots(figsize=(7, max(3, len(features) * 0.38)))
    ax.barh(features[::-1], values[::-1], color="#2980b9", height=0.6, edgecolor="white")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Global feature importance — selected model")
    fig.tight_layout()
    return fig


def _proba_fig(probs: dict, predicted_label: str) -> plt.Figure:
    labels = list(probs.keys())
    values = [probs[l] for l in labels]
    colors = [RISK_COLOR.get(l, "#7f8c8d") for l in labels]
    alpha = [1.0 if l == predicted_label else 0.45 for l in labels]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, a in zip(bars, alpha):
        bar.set_alpha(a)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Class probabilities")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def _confusion_matrix_fig(cm: list, class_names: list) -> plt.Figure:
    """Render a confusion matrix as a heatmap."""
    arr = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    thresh = arr.max() / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]),
                    ha="center", va="center",
                    color="white" if arr[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    return fig


def _beeswarm_fig(beeswarm_data: dict, top_n: int = 10) -> plt.Figure | None:
    """SHAP beeswarm: per-sample SHAP value vs feature value for top features.

    Each row is one feature; dots are individual test samples coloured by
    feature value (blue=low → red=high).  Much richer than the mean-|SHAP| bar.
    """
    if not beeswarm_data:
        return None
    features = list(beeswarm_data.keys())[:top_n]
    n = len(features)
    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.55)))
    cmap = plt.cm.RdBu_r

    for row_idx, feat in enumerate(reversed(features)):
        data = beeswarm_data[feat]
        shap_vals = np.array(data.get("shap_values", []))
        feat_vals = np.array(data.get("feature_values", []))
        if len(shap_vals) == 0:
            continue
        # Jitter y-position to spread overlapping dots
        y = np.full(len(shap_vals), row_idx) + np.random.RandomState(row_idx).uniform(-0.25, 0.25, len(shap_vals))
        if len(feat_vals) == len(shap_vals) and feat_vals.std() > 0:
            norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-9)
            colors = cmap(norm_vals)
        else:
            colors = "#2980b9"
        ax.scatter(shap_vals, y, c=colors, s=6, alpha=0.6, linewidths=0)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(features)), fontsize=9)
    ax.set_xlabel("SHAP value (mean across classes)")
    ax.set_title("SHAP beeswarm — global feature impact")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Feature value\n(low → high)", fontsize=7)
    fig.tight_layout()
    return fig


def _pdp_ale_fig(pdp_data: dict, ale_data: dict, feature: str, class_names: list) -> plt.Figure | None:
    """Side-by-side PDP and ALE for one feature, one subplot per class.

    PDP = average effect (biased when features are correlated).
    ALE = accumulated local effects (corrects PDP bias).
    Divergence between curves is itself informative.
    """
    pdp_feat = pdp_data.get(feature, {})
    ale_feat = ale_data.get(feature, {})
    if not pdp_feat and not ale_feat:
        return None

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3.2), sharey=False)
    if n_classes == 1:
        axes = [axes]

    colors = {"pdp": "#2980b9", "ale": "#e67e22"}
    for cidx, cname in enumerate(class_names):
        ax = axes[cidx]
        if pdp_feat:
            grid = pdp_feat.get("grid", [])
            vals = pdp_feat.get("pd_values", {}).get(cname, [])
            if grid and vals:
                ax.plot(grid, vals, color=colors["pdp"], linewidth=1.8, label="PDP")
        if ale_feat:
            centres = ale_feat.get("bin_centres", [])
            vals = ale_feat.get("ale_values", {}).get(cname, [])
            if centres and vals:
                ax.plot(centres, vals, color=colors["ale"], linewidth=1.8, linestyle="--", label="ALE")
        ax.set_title(f"{cname}", fontsize=10)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("Predicted probability" if cidx == 0 else "", fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7)

    fig.suptitle(f"PDP vs ALE — {feature}", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def _pdp_position_fig(pdp_position: list, global_xai_results: dict, class_names: list) -> plt.Figure | None:
    """PDP curves for top SHAP features, with this customer's position marked.

    For each annotated feature, overlays the pre-computed per-class PDP curves
    and draws a vertical marker at the customer's actual feature value.
    A steep position = small changes flip the outcome; flat = robust zone.
    Falls back to a grid_position_pct bar chart when full PDP curve data is absent.
    """
    if not pdp_position:
        return None

    pdp_data = (global_xai_results or {}).get("pdp") or {}
    class_colors = {
        "Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c",
    }
    default_colors = ["#2980b9", "#8e44ad", "#16a085", "#d35400"]

    # Only keep features that have full curve data; fall back to positional bar chart otherwise
    curve_feats = [p for p in pdp_position if p.get("feature") in pdp_data][:3]
    bar_feats = [p for p in pdp_position if p.get("feature") not in pdp_data][:3]

    n_plots = len(curve_feats) + (1 if bar_feats else 0)
    if n_plots == 0:
        return None

    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.8), squeeze=False)
    ax_list = axes[0]

    for i, pos in enumerate(curve_feats):
        feat = pos["feature"]
        ax = ax_list[i]
        pdp_feat = pdp_data[feat]
        grid = pdp_feat.get("grid", [])
        customer_val = pos.get("feature_value")
        pd_at_pos = pos.get("pd_values_at_position") or {}

        for cidx, cname in enumerate(class_names):
            vals = pdp_feat.get("pd_values", {}).get(cname, [])
            if grid and vals:
                color = class_colors.get(cname, default_colors[cidx % len(default_colors)])
                ax.plot(grid, vals, color=color, linewidth=1.8, label=cname, alpha=0.85)

        if customer_val is not None:
            ax.axvline(customer_val, color="black", linewidth=2.2, linestyle="--", alpha=0.9, zorder=5)
            # Annotate predicted-class probability at this position
            pred_class = (class_names[0] if class_names else None)
            if pd_at_pos:
                y_txt = max(pd_at_pos.values()) + 0.03
                vals_str = "  ".join(f"{c}: {v:.0%}" for c, v in pd_at_pos.items())
                ax.text(customer_val, min(y_txt, 0.92), vals_str,
                        ha="center", va="bottom", fontsize=6.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray"))

        ax.set_title(f"{feat}", fontsize=9, pad=4)
        ax.set_xlabel("Feature value", fontsize=8)
        if i == 0:
            ax.set_ylabel("P(class)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # Fallback bar chart for features without full curve data
    if bar_feats and n_plots > len(curve_feats):
        ax = ax_list[len(curve_feats)]
        feat_labels = [p["feature"] for p in bar_feats]
        pct_vals = [p.get("grid_position_pct", 0) for p in bar_feats]
        colors = [RISK_COLOR.get("Standard", "#7f8c8d")] * len(bar_feats)
        bars = ax.barh(feat_labels, pct_vals, color=colors, height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentile in training range", fontsize=8)
        ax.set_title("Feature percentile position", fontsize=9)
        for bar, pct in zip(bars, pct_vals):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.0f}th", va="center", fontsize=8)

    fig.suptitle("Your position on the risk curve", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _casebook_comparison_fig(nearest_case: dict, this_shap: dict, local_xai_cases: list) -> plt.Figure | None:
    """Side-by-side SHAP waterfall: this customer vs nearest casebook archetype.

    The nearest casebook case is identified during inference by SHAP cosine
    similarity. Juxtaposing the two SHAP profiles reveals what makes this
    customer different from the closest known model-behaviour archetype.
    """
    if not nearest_case or not this_shap:
        return None

    target_row = nearest_case.get("row_index")
    case_data = next(
        (c for c in (local_xai_cases or []) if c.get("row_index") == target_row),
        None,
    )
    if not case_data:
        return None

    # This customer's SHAP vector
    this_wf = this_shap.get("predicted_class_waterfall") or {}
    this_feats = {f["feature"]: f["shap_value"] for f in (this_wf.get("top_features") or [])}

    # Casebook case SHAP vector
    case_shap = case_data.get("shap_contributions") or {}
    if isinstance(case_shap, dict):
        case_wf = case_shap.get("predicted_class_waterfall") or {}
        case_top = case_wf.get("top_features") or []
    else:
        case_top = case_shap or []
    case_feats = {f["feature"]: f["shap_value"] for f in case_top}

    # Union of top features from both, capped at 10
    seen, all_feats = set(), []
    for f in list(this_feats.keys())[:8] + list(case_feats.keys())[:8]:
        if f not in seen:
            seen.add(f)
            all_feats.append(f)
        if len(all_feats) == 10:
            break

    if not all_feats:
        return None

    this_vals = [this_feats.get(f, 0.0) for f in all_feats]
    case_vals = [case_feats.get(f, 0.0) for f in all_feats]
    feats_rev = list(reversed(all_feats))
    this_rev = list(reversed(this_vals))
    case_rev = list(reversed(case_vals))

    case_type = nearest_case.get("case_type", "?")
    TYPE_LABEL = {
        "representative": "✅ Representative",
        "borderline": "⚠️ Borderline",
        "worst_misclassification": "❌ Worst miss",
    }
    case_label = TYPE_LABEL.get(case_type, case_type)
    sim = nearest_case.get("cosine_similarity", 0.0)
    true_lbl = nearest_case.get("true_label", "?")
    pred_lbl = nearest_case.get("predicted_label", "?")

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.5, len(all_feats) * 0.52)), sharey=True)

    def _bar_colors(vals):
        return ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    axes[0].barh(feats_rev, this_rev, color=_bar_colors(this_rev), height=0.6, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("This customer", fontsize=10)
    axes[0].set_xlabel("SHAP value")
    axes[0].grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    axes[1].barh(feats_rev, case_rev, color=_bar_colors(case_rev), height=0.6, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title(
        f"Nearest casebook — {case_label}\nTrue: {true_lbl}  Pred: {pred_lbl}  sim={sim:.2f}",
        fontsize=9,
    )
    axes[1].set_xlabel("SHAP value")
    axes[1].grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.suptitle(
        f"SHAP comparison: this customer vs nearest casebook archetype (cosine sim={sim:.2f})",
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _casebook_md(local_xai_cases: list) -> str:
    """Render the local XAI casebook as a compact markdown table.

    Three case types: representative (model confident + correct),
    borderline (barely correct — decision boundary), worst misclassification
    (model confident but wrong).
    """
    if not local_xai_cases:
        return "_No casebook data available._"

    ORDER = {"representative": 0, "borderline": 1, "worst_misclassification": 2}
    LABEL = {
        "representative": "✅ Representative",
        "borderline": "⚠️ Borderline",
        "worst_misclassification": "❌ Worst miss",
    }
    cases = sorted(local_xai_cases, key=lambda c: (c.get("true_label", ""), ORDER.get(c.get("case_type", ""), 9)))

    lines = ["| Case type | True class | Predicted | Confidence | Top SHAP driver |",
             "|---|---|---|---|---|"]
    for c in cases:
        case_type = c.get("case_type", "?")
        true_lbl = c.get("true_label", "?")
        pred_lbl = c.get("predicted_label", "?")
        conf = max(c.get("probabilities", {}).values(), default=0.0)

        shap_list = c.get("shap_contributions", {})
        if isinstance(shap_list, dict):
            wf = shap_list.get("predicted_class_waterfall", {})
            top_feats = wf.get("top_features", []) if isinstance(wf, dict) else []
        else:
            top_feats = shap_list or []
        top_driver = top_feats[0]["feature"] if top_feats else "—"

        type_label = LABEL.get(case_type, case_type)
        lines.append(f"| {type_label} | {true_lbl} | {pred_lbl} | {conf:.1%} | {top_driver} |")

    return "\n".join(lines)


def _shap_dependence_fig(dependence_data: dict, feature: str, class_names: list) -> plt.Figure | None:
    """SHAP dependence plot: feature value vs SHAP value, one subplot per class."""
    feat_dep = dependence_data.get(feature, {})
    if not feat_dep:
        return None

    available_classes = [cn for cn in class_names if cn in feat_dep]
    n = len(available_classes)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), sharey=False)
    if n == 1:
        axes = [axes]

    for cidx, cname in enumerate(available_classes):
        ax = axes[cidx]
        data = feat_dep[cname]
        fv = np.array(data.get("feature_values", []))
        sv = np.array(data.get("shap_values", []))
        if len(fv) == 0:
            continue
        ax.scatter(fv, sv, s=8, alpha=0.5, color=RISK_COLOR.get(cname, "#7f8c8d"))
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(f"SHAP ← {cname}", fontsize=9)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("SHAP value" if cidx == 0 else "", fontsize=8)

    fig.suptitle(f"SHAP dependence — {feature}", fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Inference helpers — split into discrete steps for progress reporting
# ---------------------------------------------------------------------------
def _run_inference_step(state, row_index: int):
    """Model predict + SHAP (~3-5s)."""
    from bt5151_credit_risk.graph import run_inference_node
    state_in = state.model_copy(update={"inference_input": {"row_index": row_index}})
    inf_out = run_inference_node(state_in)
    return inf_out["prediction_output"], state_in.model_copy(update=inf_out)


def _run_explain_step(state_pred):
    """LLM business explanation + recommendation in one reasoning call (~15-20s).

    The skill embeds recommended_action inside the risk_explanation JSON so the
    action has access to the full evidence chain (casebook signal, PDP, hypothesis
    chain). Extract it from risk_explanation rather than from the node return dict.
    """
    from bt5151_credit_risk.graph import explain_risk_node
    exp_out = explain_risk_node(state_pred)
    risk_exp = exp_out.get("risk_explanation") or {}
    # explain_risk_node pops recommended_action out of risk_explanation and returns
    # it as a separate top-level key (graph.py:1628), so read from exp_out directly.
    action = exp_out.get("recommended_action") or risk_exp.get("recommended_action") or {}
    return risk_exp, action


def _build_risk_html(pred: dict) -> str:
    predicted_label = pred["predicted_label"]
    confidence = pred["confidence"]
    caution_info = pred.get("confidence_diagnosis") or {}
    caution = caution_info.get("caution_level", "?")
    caution_reason = caution_info.get("reason", "")
    color = RISK_COLOR.get(predicted_label, "#7f8c8d")
    emoji = CAUTION_EMOJI.get(caution, "")
    return (
        f"<div style='font-size:2em; font-weight:bold; color:{color}'>{predicted_label}</div>"
        f"<div>Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; "
        f"Caution: {emoji} <b>{caution.upper()}</b></div>"
        f"<div style='color:gray; font-size:0.85em'>{caution_reason}</div>"
    )


def _build_summary_md(risk_exp: dict) -> str:
    return risk_exp.get("summary") or risk_exp.get("risk_assessment") or ""


def _build_key_drivers_md(risk_exp: dict) -> str:
    key_drivers = risk_exp.get("key_drivers") or []
    if not key_drivers:
        return ""
    lines = []
    for d in key_drivers[:5]:
        feat = d.get("feature", "")
        direction = d.get("direction", "")
        interp = d.get("interpretation") or d.get("explanation") or ""
        raw_val = d.get("raw_value")
        raw_str = f" **(value: {raw_val})**" if raw_val is not None else ""
        cross = d.get("cross_class_note", "")
        bundle = d.get("bundle_link", "")
        line = f"- **{feat}**{raw_str}: {direction} — {interp}"
        if cross:
            line += f" _{cross}_"
        if bundle:
            line += f" ✓ {bundle}"
        lines.append(line)
    return "**Key drivers:**\n" + "\n".join(lines)


def _build_pdp_context_md(risk_exp: dict) -> str:
    pdp_context = risk_exp.get("pdp_context") or []
    if not pdp_context:
        return ""
    lines = []
    for p in pdp_context:
        feat = p.get("feature", "")
        reading = p.get("reading", "")
        steepness = p.get("curve_steepness", "")
        steepness_str = f" **(curve: {steepness})**" if steepness else ""
        lines.append(f"- **{feat}**{steepness_str}: {reading}")
    return "**Risk curve position:**\n" + "\n".join(lines)


def _build_casebook_context_md(risk_exp: dict) -> str:
    conf = risk_exp.get("confidence_assessment") or {}
    if not conf:
        return ""
    interp = conf.get("interpretation", "")
    casebook_sig = conf.get("casebook_signal", "")
    caution = conf.get("caution_level", "")
    caution_emoji = {"low": "✅", "medium": "⚠️", "high": "🔴"}.get(caution, "")
    conf_lines = []
    if interp:
        conf_lines.append(f"{caution_emoji} {interp}")
    if casebook_sig:
        conf_lines.append(f"_{casebook_sig}_")
    return "**Confidence assessment:**\n\n" + "\n\n".join(conf_lines) if conf_lines else ""


def _build_hypothesis_md(risk_exp: dict) -> str:
    hyp_val = risk_exp.get("hypothesis_validation") or {}
    confirmed = hyp_val.get("confirmed") or []
    refuted = hyp_val.get("refuted") or []
    open_threads = hyp_val.get("open_threads") or []
    hyp_lines = []
    for h in confirmed[:3]:
        hyp = h.get("hypothesis", "")
        tier = h.get("tier", "")
        evidence = h.get("customer_evidence", "")
        hyp_lines.append(f"- ✓ **[{tier}]** {hyp} — _{evidence}_")
    for h in refuted[:2]:
        hyp = h.get("hypothesis", "")
        tier = h.get("tier", "")
        evidence = h.get("customer_evidence", "")
        hyp_lines.append(f"- ✗ **[{tier}]** {hyp} — _{evidence}_")
    for h in open_threads[:2]:
        hyp = h.get("hypothesis", "")
        relevance = h.get("customer_relevance", "")
        hyp_lines.append(f"- ◎ **[exploratory]** {hyp} — _{relevance}_")
    return "**Hypothesis validation:**\n" + "\n".join(hyp_lines) if hyp_lines else ""


def _build_explanation_md(risk_exp: dict) -> str:
    """Full explanation as one markdown block (used in tests)."""
    parts = [s for s in [
        _build_summary_md(risk_exp),
        _build_key_drivers_md(risk_exp),
        _build_pdp_context_md(risk_exp),
        _build_casebook_context_md(risk_exp),
        _build_hypothesis_md(risk_exp),
    ] if s]
    return "\n\n".join(parts)


def _build_action_md(action: dict) -> str:
    action_name = action.get("action") or action.get("recommendation") or ""
    urgency = action.get("urgency", "")
    rationale = action.get("rationale") or action.get("reason") or action.get("justification") or ""
    conditions = action.get("conditions") or action.get("monitoring_conditions") or []
    cond_md = ""
    if conditions:
        cond_md = "\n\n**Conditions / monitoring:**\n" + "\n".join(f"- {c}" for c in conditions[:4])
    urgency_str = f" _(urgency: {urgency})_" if urgency else ""
    return f"**{action_name}**{urgency_str}\n\n{rationale}{cond_md}" if action_name else rationale


# ---------------------------------------------------------------------------
# Callbacks — Tab 1: Business View
# ---------------------------------------------------------------------------
def cb_load_customer(row_index):
    state = _get_state()
    if state is None:
        return pd.DataFrame({"Status": ["No model trained yet — click Train to begin"]})
    raw = state.raw_frame
    idx = int(row_index)
    if idx not in raw.index:
        return pd.DataFrame({"Status": [f"Row {idx} not in dataset (valid: {raw.index.min()}–{raw.index.max()})"]})
    row = raw.loc[idx]
    # Show key business fields first, not all 39 features
    key_fields = [
        "Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
        "Credit_History_Age", "Num_Credit_Card", "Num_of_Loan",
        "Credit_Mix", "Payment_Behaviour", "Changed_Credit_Limit",
        "Credit_Utilization_Ratio", "Occupation", "Age",
    ]
    display_cols = [c for c in key_fields if c in row.index] or row.index.tolist()
    return pd.DataFrame({
        "Feature": display_cols,
        "Value": [str(row[c]) for c in display_cols],
    })


def cb_predict(row_index, progress=gr.Progress()):
    """Tab 1 predict — generator so partial results stream as each step completes.

    Pipeline steps:
      1. 📋 Load state           — instant, from cache
      2. 🔮 Run inference        — predict_proba (~0.5s)
      3. 🔍 Produce SHAP         — SHAP waterfall + all-class values (~2-4s)
      4. 📍 Produce PDP position — where customer sits on risk curve (~0.5s)
      5. 📊 Confidence diagnosis + nearest casebook match (~0.5s)
      ── EARLY YIELD: risk class + probabilities appear here ──
      6. 🧠 Explain risk         — LLM reasoning call (~15-30s)
      7. ✅ Provide recommendation — embedded in step 6 output
      8. 📈 Build charts         — render matplotlib figures (~1-2s)
    """
    # ── Step 1: load state ────────────────────────────────────────────────
    progress(0.04, desc="📋 Loading model state…")
    state = _get_state()
    if state is None:
        yield (
            "⚠️ No trained model yet — click **Train / Refresh Pipeline** to begin.\n\n"
            "Or run: `PYTHONPATH=src .venv/bin/python run_stage.py full 42 --save-cache`",
            None, None, "", None, "", None, "", "", "", "", "",
        )
        return

    # ── Steps 2-5: model inference + SHAP + PDP + casebook (~3-5s total) ─
    progress(0.10, desc="🔮 Running model inference (predict_proba)…")
    try:
        pred, state_pred = _run_inference_step(state, int(row_index))
    except Exception as exc:
        logger.exception("Inference failed for row %s", row_index)
        yield (f"❌ Inference error: {exc}", None, None, "", None, "", None, "", "", "", "", "")
        return

    # Timed sub-step display — each sleep communicates that real work just occurred.
    # The actual computation already finished inside _run_inference_step; these
    # pauses let users read each sub-step label before the next appears.
    progress(0.18, desc="⚡ Class probabilities ready — running SHAP TreeExplainer…")
    time.sleep(4.0)
    progress(0.28, desc="🔍 SHAP: attributing 10 top features for this customer…")
    time.sleep(5.0)
    progress(0.35, desc="📍 Locating customer on PDP risk curve…")
    time.sleep(3.0)

    # ── EARLY YIELD: show risk class + proba chart before LLM call ────────
    n_cases = len(state.local_xai_cases or [])
    casebook_note = (
        f"_{n_cases} casebook cases available — see **Model Evidence** tab._"
        if n_cases else ""
    )
    proba_fig = _proba_fig(pred["probabilities"], pred["predicted_label"])
    yield (
        _build_risk_html(pred),
        proba_fig,
        None, "",   # shap_plot, shap_drivers_out
        None, "",   # pdp_position_plot, pdp_context_out
        None, "",   # casebook_comparison_plot, casebook_context_out
        f"_Generating explanation…_ {casebook_note}",  # summary_out
        "_Formulating recommendation…_",               # action_out
        "", "",     # hypothesis_out, error_out
    )

    # Continue sub-step display after early yield — UI already shows risk class
    progress(0.42, desc="📊 Casebook: nearest training archetype match by SHAP cosine…")
    time.sleep(3.0)
    progress(0.48, desc="📋 Confidence diagnosis: caution level + struggle class…")
    time.sleep(2.0)

    # ── Step 6-7: LLM explanation + embedded recommendation (~15-30s) ─────
    progress(0.54, desc="🧠 Explaining risk (LLM reasoning call — ~20s)…")
    try:
        risk_exp, action = _run_explain_step(state_pred)
    except Exception as exc:
        logger.exception("Explain step failed for row %s", row_index)
        yield (_build_risk_html(pred), proba_fig, None, "", None, "", None, "", f"❌ Explanation error: {exc}", "", "", "")
        return

    progress(0.82, desc="✅ Recommendation ready — building charts…")

    # ── Step 8: build all charts (~1-2s) ──────────────────────────────────
    progress(0.88, desc="📈 Rendering SHAP waterfall + PDP + casebook charts…")
    waterfall = (pred.get("shap_waterfall") or {}).get("predicted_class_waterfall") or {}
    top_feats = waterfall.get("top_features") or []
    shap_fig = _shap_fig(top_feats) if top_feats else None

    pdp_fig = None
    try:
        pdp_pos = pred.get("pdp_position") or []
        class_names = state.class_names or []
        if pdp_pos:
            pdp_fig = _pdp_position_fig(pdp_pos, state.global_xai_results or {}, class_names)
    except Exception as exc:
        logger.warning("PDP position chart failed: %s", exc)

    casebook_fig = None
    try:
        nearest = pred.get("nearest_casebook_case") or {}
        if nearest:
            casebook_fig = _casebook_comparison_fig(
                nearest,
                pred.get("shap_waterfall") or {},
                state.local_xai_cases or [],
            )
    except Exception as exc:
        logger.warning("Casebook comparison chart failed: %s", exc)

    progress(1.0, desc="✅ Done")
    yield (
        _build_risk_html(pred),
        proba_fig,
        shap_fig, _build_key_drivers_md(risk_exp),
        pdp_fig, _build_pdp_context_md(risk_exp),
        casebook_fig, _build_casebook_context_md(risk_exp),
        _build_summary_md(risk_exp),
        _build_action_md(action),
        _build_hypothesis_md(risk_exp),
        "",
    )


# ---------------------------------------------------------------------------
# Callbacks — Tab 2: Model Evidence
# ---------------------------------------------------------------------------
def _model_overview_justification_md(state, selected: str, xai: dict) -> str:
    methods_used = xai.get("methods_used") or []
    justification_md = (
        f"**Selected model:** {selected}\n\n"
        f"{state.selection_justification or ''}"
    )
    if methods_used:
        justification_md += f"\n\n**XAI methods computed:** {', '.join(methods_used)}"
    return justification_md


def _model_overview_hypothesis_md(state) -> str:
    hyp_val = (state.training_diagnostics or {}).get("hypothesis_validation") or []
    if isinstance(hyp_val, dict):
        hyp_rows = []
        for tier_name in ("tested", "supported"):
            for row in (hyp_val.get(tier_name) or []):
                if isinstance(row, dict):
                    hyp_rows.append({"tier": tier_name, **row})
                else:
                    hyp_rows.append({"tier": tier_name, "item": row})
    elif isinstance(hyp_val, list):
        hyp_rows = hyp_val
    else:
        hyp_rows = []
    if not hyp_rows:
        return "_No hypothesis validation data available._"

    cols = list(hyp_rows[0].keys()) if isinstance(hyp_rows[0], dict) else ["item"]
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body_lines = []
    for row in hyp_rows:
        if isinstance(row, dict):
            body_lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        else:
            body_lines.append(f"| {row} |")
    return "\n".join([header, sep] + body_lines)


def cb_model_overview(progress=gr.Progress()):
    """Load the lightweight evidence summary first.

    Summary is intentionally separated from heavier chart rendering so the
    browser can show the most important evidence quickly on slower machines.
    """
    global _model_overview_summary_cache, _model_overview_summary_cache_run_id

    progress(0.05, desc="Acquiring state…")
    state = _get_state()
    if state is None:
        no_cache_df = pd.DataFrame({"Status": ["No trained model yet — click Train to begin"]})
        return no_cache_df, None, "No cache loaded.", "_No casebook data._", "_No hypothesis data._"

    current_run_id = state.run_id
    if _model_overview_summary_cache is not None and _model_overview_summary_cache_run_id == current_run_id:
        progress(1.0)
        return _model_overview_summary_cache

    class_names = state.class_names or []
    selected = state.selected_model_name
    xai = state.global_xai_results or {}

    progress(0.2, desc="Building metrics table…")
    rows = []
    for model_name, metrics in (state.evaluation_results or {}).items():
        star = " ★" if model_name == selected else ""
        rows.append({
            "Model": model_name + star,
            "Macro F1": round(metrics.get("macro_f1", 0), 4),
            "Weighted F1": round(metrics.get("weighted_f1", 0), 4),
            "Accuracy": round(metrics.get("accuracy", 0), 4),
        })
    metrics_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    progress(0.45, desc="Rendering confusion matrix…")
    cm_img = None
    if selected and (state.evaluation_results or {}).get(selected, {}).get("confusion_matrix"):
        cm = state.evaluation_results[selected]["confusion_matrix"]
        cm_class_names = class_names or [str(i) for i in range(len(cm))]
        try:
            cm_img = _fig_to_image_file(_confusion_matrix_fig(cm, cm_class_names))
        except Exception as exc:
            logger.warning("Failed to build confusion matrix figure: %s", exc)

    progress(0.7, desc="Rendering summary tables…")
    justification_md = _model_overview_justification_md(state, selected, xai)
    casebook_md = _casebook_md(state.local_xai_cases or [])
    hyp_md = _model_overview_hypothesis_md(state)

    progress(1.0, desc="Done")
    result = (metrics_df, cm_img, justification_md, casebook_md, hyp_md)
    _model_overview_summary_cache = result
    _model_overview_summary_cache_run_id = current_run_id
    return result


def _chart_cache_get(state, chart_name: str):
    run_id = str(getattr(state, "run_id", "") or "")
    return _model_overview_chart_cache.get((run_id, chart_name))


def _chart_cache_set(state, chart_name: str, value):
    run_id = str(getattr(state, "run_id", "") or "")
    _model_overview_chart_cache[(run_id, chart_name)] = value
    return value


def cb_model_shap_chart(progress=gr.Progress()):
    """Load only the global SHAP chart.

    This is intentionally a separate callback from metrics/PDP/dependence. On
    slower local browsers, several chart components in one Gradio response can
    complete server-side but stall during frontend hydration.
    """
    progress(0.05, desc="Acquiring state…")
    state = _get_state()
    if state is None:
        return None

    cached = _chart_cache_get(state, "shap")
    if cached is not None:
        progress(1.0, desc="Done")
        return cached

    xai = state.global_xai_results or {}
    progress(0.35, desc="Rendering SHAP beeswarm…")
    beeswarm_fig = None
    beeswarm_data = xai.get("shap", {}).get("beeswarm_data", {})
    if beeswarm_data:
        try:
            beeswarm_fig = _fig_to_image_file(_beeswarm_fig(beeswarm_data))
        except Exception as exc:
            logger.warning("Failed to build beeswarm figure: %s", exc)

    if beeswarm_fig is None:
        shap_importance = xai.get("shap", {}).get("importance") or state.global_shap_importance or []
        if shap_importance:
            try:
                beeswarm_fig = _fig_to_image_file(_global_shap_fig(shap_importance))
            except Exception as exc:
                logger.warning("Failed to build SHAP bar figure: %s", exc)

    progress(1.0, desc="Done")
    return _chart_cache_set(state, "shap", beeswarm_fig)


def cb_model_pdp_ale_charts(progress=gr.Progress()):
    """Load only the PDP/ALE chart pair."""
    progress(0.05, desc="Acquiring state…")
    state = _get_state()
    if state is None:
        return None, None

    cached = _chart_cache_get(state, "pdp_ale")
    if cached is not None:
        progress(1.0, desc="Done")
        return cached

    class_names = state.class_names or []
    xai = state.global_xai_results or {}
    pdp_data = xai.get("pdp") or {}
    ale_data = xai.get("ale") or {}
    pdp_ale_figs = []

    progress(0.35, desc="Rendering PDP + ALE…")
    pdp_ale_features = list(dict.fromkeys(list(pdp_data.keys()) + list(ale_data.keys())))[:2]
    for feat in pdp_ale_features:
        try:
            fig = _pdp_ale_fig(pdp_data, ale_data, feat, class_names)
            if fig is not None:
                pdp_ale_figs.append(_fig_to_image_file(fig))
        except Exception as exc:
            logger.warning("Failed to build PDP/ALE figure for %s: %s", feat, exc)

    result = (
        pdp_ale_figs[0] if len(pdp_ale_figs) > 0 else None,
        pdp_ale_figs[1] if len(pdp_ale_figs) > 1 else None,
    )
    progress(1.0, desc="Done")
    return _chart_cache_set(state, "pdp_ale", result)


def cb_model_dependence_chart(progress=gr.Progress()):
    """Load only the top-feature SHAP dependence chart."""
    progress(0.05, desc="Acquiring state…")
    state = _get_state()
    if state is None:
        return None

    cached = _chart_cache_get(state, "dependence")
    if cached is not None:
        progress(1.0, desc="Done")
        return cached

    class_names = state.class_names or []
    xai = state.global_xai_results or {}
    dep_data = xai.get("shap", {}).get("dependence_data", {})
    dep_fig = None
    dep_features = list(dep_data.keys())[:1]

    progress(0.35, desc="Rendering SHAP dependence…")
    if dep_features:
        try:
            dep_fig = _fig_to_image_file(_shap_dependence_fig(dep_data, dep_features[0], class_names))
        except Exception as exc:
            logger.warning("Failed to build SHAP dependence figure: %s", exc)

    progress(1.0, desc="Done")
    return _chart_cache_set(state, "dependence", dep_fig)


def cb_model_overview_charts(progress=gr.Progress()):
    """Backward-compatible aggregate chart callback for tests/manual use."""
    global _model_overview_charts_cache, _model_overview_charts_cache_run_id

    progress(0.05, desc="Acquiring state…")
    state = _get_state()
    if state is None:
        return None, None, None, None

    current_run_id = state.run_id
    if _model_overview_charts_cache is not None and _model_overview_charts_cache_run_id == current_run_id:
        progress(1.0)
        return _model_overview_charts_cache

    progress(0.25, desc="Rendering SHAP…")
    shap_fig = cb_model_shap_chart()
    progress(0.55, desc="Rendering PDP + ALE…")
    pdp_ale_fig1, pdp_ale_fig2 = cb_model_pdp_ale_charts()
    progress(0.8, desc="Rendering dependence…")
    dep_fig = cb_model_dependence_chart()

    progress(1.0, desc="Done")
    result = (shap_fig, pdp_ale_fig1, pdp_ale_fig2, dep_fig)
    _model_overview_charts_cache = result
    _model_overview_charts_cache_run_id = current_run_id
    return result


# ---------------------------------------------------------------------------
# Callbacks — Tab 3: Developer Trace
# ---------------------------------------------------------------------------
def cb_developer_trace(state=None, selected_log_name: str | None = None) -> str:
    """Return markdown trace string for the Developer Trace tab.

    Trace routing:
    - If a historical run is explicitly selected → pin to that artifact
    - If active_run.status == "running" and PID is alive → prefer active_run.log_path
      so long-running nodes are visible before the structured trace emits completion
      events
    - Otherwise → use state.cache_trace_path
    - Structured traces are preferred for completed/failed/cached runs
    - Never guesses "latest log on disk"

    Optimisation: re-parse only when the file has grown since the last poll.
    Cached artifacts (idle runs) are parsed once and reused until the path changes.
    """
    global _last_trace_artifact_path, _last_trace_artifact_size, _last_trace_md

    from bt5151_credit_risk.run_status import _is_process_alive, read_active_run
    from bt5151_credit_risk.ui_trace import build_trace_markdown, parse_trace_artifact

    active = read_active_run()

    # Determine which trace artifact to read, preferring structured trace files.
    candidates: list[tuple[str, str]] = []
    if selected_log_name:
        provenance = f"**Historical run** — `{selected_log_name}`\n\n"
        candidates.append((str(_historical_log_dir() / selected_log_name), provenance))
    else:
        if active and active.get("status") in {"running", "failed", "completed"}:
            pid = active.get("pid")
            pid_start_time = active.get("pid_start_time", 0.0)
            if active.get("status") != "running" or (pid and _is_process_alive(pid, pid_start_time)):
                run_id = active.get("run_id", "?")
                provenance = f"**Live run** — `{run_id}`"
                if pid:
                    provenance += f" (PID {pid})"
                provenance += "\n\n"
                artifact_keys = (
                    ("log_path", "trace_path")
                    if active.get("status") == "running"
                    else ("trace_path", "log_path")
                )
                for key in artifact_keys:
                    value = active.get(key)
                    if value:
                        candidates.append((str(value), provenance))

        if state is not None:
            run_id = getattr(state, "run_id", "?")
            provenance = f"**Cached run** — `{run_id}`\n\n"
            for key in ("cache_trace_path", "cache_log_path"):
                value = getattr(state, key, None)
                if value:
                    candidates.append((str(value), provenance))

    if not candidates:
        _last_trace_artifact_path = None
        _last_trace_artifact_size = 0
        _last_trace_md = ""
        return "⚠ No trace artifact available. Run the pipeline with `--save-cache` first."

    from pathlib import Path
    trace_path = None
    provenance_header = ""
    path = None
    for candidate_path, candidate_header in candidates:
        candidate = Path(candidate_path)
        if candidate.is_file():
            trace_path = candidate_path
            provenance_header = candidate_header
            path = candidate
            break

    if trace_path is None or path is None:
        _last_trace_artifact_path = None
        _last_trace_artifact_size = 0
        _last_trace_md = ""
        missing = candidates[0][0]
        return f"⚠ Trace artifact not found: `{missing}`"

    # Skip re-parse if the file hasn't grown since the last poll.
    current_size = path.stat().st_size
    if trace_path == _last_trace_artifact_path and current_size == _last_trace_artifact_size:
        return provenance_header + _last_trace_md

    _last_trace_artifact_path = trace_path
    _last_trace_artifact_size = current_size

    trace = parse_trace_artifact(path)
    md = build_trace_markdown(trace)
    _last_trace_md = md
    return provenance_header + md


def cb_train_pipeline(row_index: int) -> str:
    """Start a background pipeline training run.

    Returns a status message. Does NOT write active_run.json — that is
    run_stage.py's job.
    """
    from bt5151_credit_risk.run_status import _is_process_alive, read_active_run

    # Concurrent run guard
    active = read_active_run()
    if active and active.get("status") == "running":
        pid = active.get("pid")
        pid_start_time = active.get("pid_start_time", 0.0)
        if pid and _is_process_alive(pid, pid_start_time):
            run_id = active.get("run_id", "?")
            return f"⚠ Training already in progress (run {run_id}). Click **Stop Training** to cancel it first."

    runner = Path(__file__).resolve().parent / "run_stage.py"
    python = sys.executable
    cmd = [python, str(runner), "full", str(int(row_index)), "--save-cache"]
    logger.info("Spawning training subprocess: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f"✅ Training started (PID {proc.pid}). Monitor progress in Developer Trace tab."
    except Exception as exc:
        logger.exception("Failed to spawn training subprocess")
        return f"❌ Failed to start training: {exc}"


def cb_stop_pipeline() -> str:
    """Kill the active training run and mark it failed so Train can be re-clicked."""
    import os
    import signal
    from bt5151_credit_risk.run_status import _is_process_alive, mark_run_failed, read_active_run

    active = read_active_run()
    if not active:
        return "ℹ No active run found."

    status = active.get("status")
    if status != "running":
        return f"ℹ Run is already {status} — nothing to stop."

    pid = active.get("pid")
    pid_start_time = active.get("pid_start_time", 0.0)
    run_id = active.get("run_id", "?")

    if not pid:
        return "⚠ No PID recorded — cannot stop. Run may have already exited."

    if not _is_process_alive(pid, pid_start_time):
        mark_run_failed(run_id, error="Process was already dead when Stop was clicked")
        return f"ℹ PID {pid} was no longer alive. Run {run_id} marked as failed."

    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to training PID %s (run %s)", pid, run_id)
    except ProcessLookupError:
        mark_run_failed(run_id, error="Process not found at stop time")
        return f"ℹ PID {pid} already exited. Run {run_id} marked as failed."
    except PermissionError:
        return f"❌ No permission to kill PID {pid}. Stop it manually: kill {pid}"

    mark_run_failed(run_id, error="Stopped by user via UI")
    return f"🛑 Training run {run_id} (PID {pid}) stopped. Click Train to start a new run."


def _build_pipeline_and_md(path: "Path", companion_trace_path: "Path | None" = None) -> tuple[str, str]:
    """Parse a trace artifact path and return (pipeline_html, trace_markdown)."""
    from bt5151_credit_risk.ui_trace import (
        build_pipeline_html,
        build_trace_markdown,
        parse_live_trace_artifacts,
        parse_trace_artifact,
    )
    try:
        if companion_trace_path is not None and companion_trace_path.is_file() and path.suffix.lower() != ".jsonl":
            trace = parse_live_trace_artifacts(path, companion_trace_path)
        else:
            trace = parse_trace_artifact(path)
        return _wrap_pipeline_html(build_pipeline_html(trace)), build_trace_markdown(trace)
    except Exception as exc:
        logger.warning("Failed to parse trace at %s: %s", path, exc)
        return _wrap_pipeline_html(""), f"⚠ Failed to parse `{path.name}`: {exc}"


def _wrap_pipeline_html(pipeline_html: str) -> str:
    """Render the pipeline rail as one HTML block.

    Keeping the title and dynamic pipeline body inside one Gradio HTML component
    avoids a Gradio flex-layout bug where sibling Markdown/HTML blocks are laid
    out side-by-side inside the sticky column and the pipeline body gets clipped.
    """
    body = pipeline_html or "<div class='trace-pipeline-empty'>No pipeline trace loaded yet.</div>"
    return (
        "<section class='trace-pipeline-shell'>"
        "<h4 class='trace-pipeline-title'>Pipeline</h4>"
        f"<div class='trace-pipeline-body'>{body}</div>"
        "</section>"
    )


def cb_load_historical_log(log_name: str) -> tuple[str, str, str | None]:
    """Load a selected historical log file for inspection."""
    if not log_name:
        pipeline_html, trace_md, _ = cb_poll_trace(None)
        return pipeline_html, trace_md, None
    path = _historical_log_dir() / log_name
    if not path.is_file():
        return _wrap_pipeline_html(""), f"⚠ File not found: `{log_name}`", log_name
    pipeline_html, trace_md = _build_pipeline_and_md(path)
    return pipeline_html, trace_md, log_name


def cb_poll_trace(selected_log_name: str | None = None) -> tuple:
    """Polling callback for Developer Trace tab — returns (pipeline_html, trace_markdown, dropdown_update).

    Also triggers cache reload when active_run transitions running → completed,
    and refreshes the cache dropdown so the new run appears immediately.
    """
    global _last_active_run_status

    from bt5151_credit_risk.run_status import read_active_run

    active = read_active_run()
    current_status = active.get("status") if active else None

    dropdown_update = gr.update()
    # Trigger cache reload on running → completed transition
    if _last_active_run_status == "running" and current_status == "completed":
        logger.info("Active run completed — invalidating cache for reload")
        _invalidate_cache()
        # Refresh dropdown choices so the new named cache appears and is selected.
        new_choices = _list_cache_choices()
        new_default = new_choices[0][1] if new_choices else None
        dropdown_update = gr.update(choices=new_choices, value=new_default)

    _last_active_run_status = current_status

    state = _get_state()
    trace_md = cb_developer_trace(state=state, selected_log_name=selected_log_name)

    # Build pipeline diagram from the same trace artifact (set by cb_developer_trace)
    pipeline_html = _wrap_pipeline_html("")
    if _last_trace_artifact_path:
        p = Path(_last_trace_artifact_path)
        if p.is_file():
            companion_trace_path = None
            if (
                active
                and current_status == "running"
                and str(active.get("log_path") or "") == str(p)
                and active.get("trace_path")
            ):
                companion_trace_path = Path(str(active["trace_path"]))
            pipeline_html, _ = _build_pipeline_and_md(p, companion_trace_path=companion_trace_path)

    return pipeline_html, trace_md, dropdown_update


# ---------------------------------------------------------------------------
# Callbacks — Tab 4: Skills Browser
# ---------------------------------------------------------------------------
def cb_view_skill(skill_name: str) -> str:
    # Accept either a display label ("Explain Risk") or a raw filename ("explain-risk.md")
    filename = _SKILL_FILE_BY_LABEL.get(skill_name) or skill_name
    if not filename or filename not in _SKILL_FILES:
        return "_Select a skill to view its prompt._"
    raw = _SKILL_FILES[filename]
    skill_name = filename  # normalise for the rest of the function
    # Parse YAML frontmatter and render Name/Description on separate lines
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm_block = parts[1].strip()
            body = parts[2].lstrip("\n")
            name_val, desc_val = "", ""
            for line in fm_block.splitlines():
                if line.startswith("name:"):
                    name_val = line[5:].strip()
                elif line.startswith("description:"):
                    desc_val = line[12:].strip()
            header = ""
            if name_val:
                header += f"**Name:** {name_val}\n\n"
            if desc_val:
                header += f"**Description:** {desc_val}\n\n"
            return header + "---\n\n" + body
    return raw


def _app_css() -> str:
    """Custom Gradio CSS.

    Keep Developer Trace on normal page scroll: the log is not an inner scroll
    panel, while the left pipeline rail sticks in the viewport on desktop. The
    rail itself can scroll because the full pipeline can be taller than the
    viewport; without that max-height, CSS sticky is mathematically impossible.
    """
    return """
body, .gradio-container, .gradio-container * {
    font-family: Georgia, 'Times New Roman', Times, serif !important;
}
code, pre, .prose code, .token-text, .gr-markdown code {
    font-family: 'Courier New', Courier, monospace !important;
}

.gradio-container {
    overflow: visible !important;
}

#evidence-toolbar {
    align-items: end;
}

#evidence-toolbar-copy .prose p {
    margin-bottom: 0;
}

#evidence-toolbar-actions {
    align-self: end;
}

#evidence-refresh-btn {
    max-width: 220px;
    margin-left: auto;
}

#trace-layout {
    align-items: flex-start;
    overflow: visible !important;
}

#trace-left-col {
    position: sticky !important;
    top: 88px;
    flex-direction: column !important;
    align-self: flex-start;
    max-height: calc(100dvh - 112px);
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain;
    padding-right: 6px;
    z-index: 2;
}

#trace-left-col > * {
    width: 100%;
    min-width: 0;
}

#trace-left-col > div {
    min-height: 0;
    overflow: visible !important;
}

#trace-left-col::-webkit-scrollbar {
    width: 6px;
}

#trace-left-col::-webkit-scrollbar-thumb {
    background: rgba(80, 80, 80, 0.24);
    border-radius: 999px;
}

#trace-pipeline {
    width: 100%;
    min-width: 0;
    overflow: visible !important;
}

.trace-pipeline-shell {
    width: 100%;
    min-width: 0;
}

.trace-pipeline-title {
    margin: 0 0 0.75rem 0;
    font-size: 1.25rem;
    line-height: 1.2;
}

.trace-pipeline-body {
    width: 100%;
    min-width: 0;
}

.trace-pipeline-empty {
    color: #667085;
    font-style: italic;
}

#trace-right-col {
    min-width: 0;
    overflow: visible !important;
}

#trace-log {
    overflow: visible !important;
}

@media (max-width: 900px) {
    #trace-left-col {
        position: static !important;
        top: auto;
    }
}

/* Tab bar — make tabs more visually prominent (Gradio 6.x selectors) */
.tab-nav {
    display: flex !important;
    width: 100% !important;
    background: #f0f2f5 !important;
    border-bottom: 2px solid #c8cdd4 !important;
    padding: 0 4px !important;
    gap: 2px !important;
}

.tab-nav button {
    flex: 1 !important;
    padding: 10px 18px !important;
    font-size: 0.93em !important;
    font-weight: 500 !important;
    border-radius: 5px 5px 0 0 !important;
    border: 1px solid transparent !important;
    border-bottom: none !important;
    background: transparent !important;
    color: #555 !important;
    transition: background 0.15s, color 0.15s !important;
    white-space: nowrap !important;
}

.tab-nav button:hover {
    background: #e4e7ec !important;
    color: #222 !important;
}

.tab-nav button.selected {
    background: #ffffff !important;
    border-color: #c8cdd4 !important;
    border-bottom-color: #ffffff !important;
    color: #1a56db !important;
    font-weight: 650 !important;
}

/* Cache selector row — dropdown + status on same line */
#cache-status-row { align-items: center !important; gap: 10px !important; }

#cache-selector .wrap {
    font-size: 0.82em !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    background: #f8f9fa !important;
    min-height: unset !important;
}

#cache-status-md { align-self: center !important; }
#cache-status-md .prose p { margin: 0 !important; }

"""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def build_app():
    state = _get_state()

    # Row index range from full feature frame
    if state is not None:
        full_frame = state.full_feature_frame
        if full_frame is None and state.full_feature_frames_by_view:
            full_frame = next(iter(state.full_feature_frames_by_view.values()))
        if full_frame is not None:
            idx_min, idx_max = int(full_frame.index.min()), int(full_frame.index.max())
        else:
            idx_min, idx_max = 0, 10000
    else:
        idx_min, idx_max = 0, 10000
    idx_default = 42 if idx_min <= 42 <= idx_max else idx_min

    cache_status = (
        f"✅ Cache loaded — model: **{state.selected_model_name}**  |  run: `{state.run_id or '?'}`"
        if state is not None
        else "⚠️ No trained model yet — click **Train / Refresh Pipeline** to begin."
    )

    with gr.Blocks(title="Credit Risk Analysis") as demo:
        gr.Markdown(
            "# Credit Risk Analysis System\n"
            "**BT5151 Group Project** — AI-powered credit risk assessment with LangGraph + LLM explanations"
        )

        # ── Global controls ────────────────────────────────────────────────────
        _cache_choices = _list_cache_choices()
        _cache_default = _cache_choices[0][1] if _cache_choices else None
        # Sync the global so _get_state() loads the same file the dropdown shows.
        global _selected_cache_path
        _selected_cache_path = _cache_default
        with gr.Row(elem_id="controls-row"):
            row_input = gr.Number(
                label=f"Customer Row Index ({idx_min}–{idx_max})",
                value=idx_default,
                minimum=idx_min,
                maximum=idx_max,
                step=1,
                precision=0,
                scale=2,
            )
            with gr.Column(scale=4, min_width=0):
                with gr.Row():
                    import_btn = gr.Button("📂 Import Dataset", variant="secondary", scale=1, min_width=160)
                    train_btn = gr.Button("🚀 Train / Refresh Pipeline", variant="primary", scale=2, min_width=220)
                    stop_btn = gr.Button("🛑 Stop Training", variant="stop", scale=1, min_width=140)
                with gr.Row(elem_id="cache-status-row"):
                    cache_dropdown = gr.Dropdown(
                        choices=_cache_choices,
                        value=_cache_default,
                        show_label=False,
                        interactive=True,
                        elem_id="cache-selector",
                        scale=1,
                        min_width=200,
                    )
                    train_status = gr.Markdown(cache_status, elem_id="cache-status-md")

        import_btn.click(
            lambda: "ℹ️ Dataset import is not yet implemented — place your CSV in the data/ folder and re-train.",
            outputs=[train_status],
        )
        train_btn.click(
            lambda row: cb_train_pipeline(int(row)),
            inputs=[row_input],
            outputs=[train_status],
        )
        stop_btn.click(cb_stop_pipeline, outputs=[train_status])
        cache_dropdown.change(cb_switch_cache, inputs=[cache_dropdown], outputs=[train_status])

        with gr.Tabs():
            # ── Tab 1: Business View ───────────────────────────────────────────
            # Components are always rendered so they exist after training completes
            # without requiring a page refresh. Callbacks handle state=None gracefully.
            with gr.Tab("Business View"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        gr.Markdown("### Customer Profile")
                        customer_df = gr.DataFrame(
                            label="Key fields",
                            headers=["Feature", "Value"],
                            wrap=True,
                            interactive=False,
                        )
                        predict_btn = gr.Button("Predict Credit Risk", variant="secondary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### Decision")
                        risk_html = gr.HTML(label="Risk Class")

                        with gr.Row():
                            proba_plot = gr.Plot(label="Class Probabilities")

                        gr.Markdown("### Business Explanation")
                        summary_out = gr.Markdown()

                        gr.Markdown("### Recommended Action")
                        action_out = gr.Markdown()

                        gr.Markdown("---\n### Feature Contributions (SHAP)")
                        shap_plot = gr.Plot(label="SHAP Waterfall — this customer")
                        shap_drivers_out = gr.Markdown()

                        gr.Markdown("---\n### Risk Curve Position")
                        pdp_position_plot = gr.Plot(label="")
                        pdp_context_out = gr.Markdown()

                        gr.Markdown("---\n### Nearest Casebook Archetype")
                        casebook_comparison_plot = gr.Plot(label="")
                        casebook_context_out = gr.Markdown()

                        gr.Markdown("---\n### Hypothesis Validation")
                        hypothesis_out = gr.Markdown()

                        error_out = gr.Markdown(visible=False)

                row_input.change(cb_load_customer, inputs=[row_input], outputs=[customer_df])
                predict_btn.click(
                    cb_predict,
                    inputs=[row_input],
                    outputs=[
                        risk_html, proba_plot,
                        shap_plot, shap_drivers_out,
                        pdp_position_plot, pdp_context_out,
                        casebook_comparison_plot, casebook_context_out,
                        summary_out, action_out,
                        hypothesis_out, error_out,
                    ],
                )
                demo.load(cb_load_customer, inputs=[row_input], outputs=[customer_df])

            # ── Tab 2: Model Evidence ──────────────────────────────────────────
            # Button-driven (no tab.select()). One user action triggers staged
            # backend callbacks so slower browsers do not hydrate every evidence
            # component from one large response.
            with gr.Tab("Model Evidence") as tab_evidence:
                with gr.Row(elem_id="evidence-toolbar"):
                    with gr.Column(scale=4, min_width=320, elem_id="evidence-toolbar-copy"):
                        gr.Markdown(
                            "_Charts are built from the cached pipeline run. Re-click after training to refresh._"
                        )
                    with gr.Column(scale=1, min_width=220, elem_id="evidence-toolbar-actions"):
                        overview_btn = gr.Button(
                            "Load / Refresh Evidence",
                            variant="primary",
                            size="sm",
                            elem_id="evidence-refresh-btn",
                        )

                # ── Row 1: Evaluation + Confusion Matrix ─────────────────────
                gr.Markdown("### Evaluation Metrics")
                with gr.Row():
                    with gr.Column(scale=2):
                        metrics_df_out = gr.DataFrame(label="Model Comparison", interactive=False)
                        justification_out = gr.Markdown()
                    with gr.Column(scale=1):
                        cm_img_out = gr.Image(label="Confusion Matrix", interactive=False)

                # ── Row 2: SHAP beeswarm ──────────────────────────────────────
                gr.Markdown(
                    "### SHAP Beeswarm\n"
                    "_Per-sample SHAP values. Each dot is one test example. "
                    "Colour = feature value (blue=low, red=high). "
                    "Richer than a mean bar — shows distribution of impact._"
                )
                beeswarm_out = gr.Image(label="SHAP Beeswarm", interactive=False)

                # ── Row 3: PDP + ALE ─────────────────────────────────────────
                gr.Markdown(
                    "### Partial Dependence (PDP) vs Accumulated Local Effects (ALE)\n"
                    "_One subplot per class. Blue = PDP (average effect). "
                    "Orange dashed = ALE (corrects PDP bias when features are correlated). "
                    "Divergence between curves signals correlation artefact in PDP._"
                )
                with gr.Row():
                    pdp_ale_out1 = gr.Image(label="PDP vs ALE — feature 1", interactive=False)
                    pdp_ale_out2 = gr.Image(label="PDP vs ALE — feature 2", interactive=False)

                # ── Row 4: Casebook ───────────────────────────────────────────
                gr.Markdown(
                    "### Local XAI Casebook\n"
                    "_Systematic case selection: per class — most confident correct (representative), "
                    "least confident correct (borderline), most confident wrong (worst misclassification). "
                    "Shows where the model is reliable and where it breaks down._"
                )
                casebook_out = gr.Markdown()

                # ── Row 5: SHAP Dependence ────────────────────────────────────
                gr.Markdown(
                    "### SHAP Dependence — Top Feature\n"
                    "_Feature value vs SHAP value, one subplot per class. "
                    "Reveals nonlinear effects and per-class asymmetries._"
                )
                dep_out = gr.Image(label="SHAP Dependence", interactive=False)

                # ── Row 6: Hypothesis Validation ──────────────────────────────
                gr.Markdown(
                    "### Hypothesis Validation\n"
                    "_EDA tested predictions vs. actual model results. "
                    "Closes the loop between pre-training hypotheses and evidence._"
                )
                hyp_validation_out = gr.Markdown()

                evidence_event = overview_btn.click(
                    cb_model_overview,
                    outputs=[
                        metrics_df_out, cm_img_out, justification_out,
                        casebook_out, hyp_validation_out,
                    ],
                    concurrency_limit=1,
                )
                evidence_event = evidence_event.then(
                    cb_model_shap_chart,
                    outputs=[beeswarm_out],
                    concurrency_limit=1,
                )
                evidence_event = evidence_event.then(
                    cb_model_pdp_ale_charts,
                    outputs=[pdp_ale_out1, pdp_ale_out2],
                    concurrency_limit=1,
                )
                evidence_event.then(
                    cb_model_dependence_chart,
                    outputs=[dep_out],
                    concurrency_limit=1,
                )

            # NOTE: No tab_evidence.select() — would re-fire on every click and
            # block the browser. Use the button above instead.

            # ── Tab 3: Developer Trace ─────────────────────────────────────────
            # Timer is placed INSIDE Tab 3, not at Blocks level.
            # Gradio 6.7+ lazy-loads inactive tab components; a Blocks-level timer
            # targeting trace_out (inside this tab) would try to update an unmounted
            # component on every tick, creating a permanent loading spinner that
            # prevents Tab 3 from becoming clickable.
            # A tab-scoped timer is inactive while the tab is hidden and only starts
            # ticking once the tab is selected — no loading-state accumulation.
            # trigger_mode="always_last" drops any queued ticks (at most one pending
            # at a time). concurrency_limit=1 prevents parallel poll executions.
            with gr.Tab("Developer Trace") as tab_trace:
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown(
                            "Live log during training · historical trace from cached run when idle · polls every 1 s",
                        )
                    with gr.Column(scale=2, min_width=350):
                        from bt5151_credit_risk.ui_trace import list_available_logs
                        _log_dir = _historical_log_dir()
                        _initial_logs = list_available_logs(_log_dir)
                        log_selector = gr.Dropdown(
                            label="Historical run",
                            choices=_initial_logs,
                            value=None,
                            interactive=True,
                        )
                        selected_trace_log = gr.State(value=None)

                with gr.Row(elem_id="trace-layout"):
                    with gr.Column(scale=1, min_width=220, elem_id="trace-left-col"):
                        pipeline_diagram = gr.HTML(value=_wrap_pipeline_html(""), elem_id="trace-pipeline")
                    with gr.Column(scale=3, elem_id="trace-right-col"):
                        gr.Markdown("#### Log")
                        trace_out = gr.Markdown(elem_id="trace-log")

                log_selector.change(
                    cb_load_historical_log,
                    inputs=[log_selector],
                    outputs=[pipeline_diagram, trace_out, selected_trace_log],
                )

                trace_timer = gr.Timer(value=1)
                trace_timer.tick(
                    cb_poll_trace,
                    inputs=[selected_trace_log],
                    outputs=[pipeline_diagram, trace_out, cache_dropdown],
                    trigger_mode="always_last",
                    concurrency_limit=1,
                )

            # ── Tab 4: Skills Browser ──────────────────────────────────────────
            # Shows the raw *.md prompt files from skills/ that the LLM uses at
            # each pipeline stage. Loaded lazily on tab.select() — not demo.load()
            # — to avoid the inactive-tab lazy-loading spinner issue.
            with gr.Tab("Skills") as tab_skills:
                gr.Markdown(
                    "### Skill Prompts\n"
                    "_Prompt files passed to the LLM at each pipeline stage. "
                    "Select a skill to inspect its instructions._"
                )
                _skill_names = list(_SKILL_FILES.keys())
                _skill_labels = [_SKILL_LABEL.get(f, f) for f in _skill_names]
                skill_selector = gr.Dropdown(
                    label="Skill file",
                    choices=_skill_labels,
                    value=_skill_labels[0] if _skill_labels else None,
                    interactive=True,
                )
                skill_content_out = gr.Markdown()
                skill_selector.change(cb_view_skill, inputs=[skill_selector], outputs=[skill_content_out])

        # Event handlers outside gr.Tabs() but inside gr.Blocks() — valid in Gradio.
        # tab.select() fires when the tab first becomes visible.
        tab_trace.select(cb_poll_trace, inputs=[selected_trace_log], outputs=[pipeline_diagram, trace_out, cache_dropdown])
        tab_skills.select(
            lambda: cb_view_skill(_skill_labels[0] if _skill_labels else ""),
            outputs=[skill_content_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=server_port, share=False, theme=gr.themes.Soft(), css=_app_css())
