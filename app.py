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

import io
import logging
import subprocess
import sys
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Global pipeline state — one source of truth for completed runs
# ---------------------------------------------------------------------------
_state = None
_state_lock = threading.Lock()

# Track last-seen active run status for cache-reload trigger
_last_active_run_status: str | None = None

# Track trace artifact state to avoid re-parsing unchanged files every poll tick
_last_trace_artifact_path: str | None = None
_last_trace_artifact_size: int = 0
_last_trace_md: str = ""

# Cache for model overview tab — keyed by run_id so it resets after a new training run
_model_overview_cache: tuple | None = None
_model_overview_cache_run_id: str | None = None

RISK_COLOR = {"Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c"}
CAUTION_EMOJI = {"low": "✅", "medium": "⚠️", "high": "🔴"}


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image and close the figure immediately.

    Used for Tab 2's static charts. gr.Image with PIL is a simple PNG <img> tag —
    much lighter than gr.Plot's plotly/savefig serialization path, and avoids
    browser-side canvas rendering that can freeze the UI.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()  # copy so the BytesIO can be GC'd


def _get_state():
    global _state
    with _state_lock:
        if _state is None:
            from bt5151_credit_risk.cache import load_cache
            _state = load_cache()
        return _state


def _invalidate_cache() -> None:
    """Clear the in-memory state and pre-warm the cache in a background thread.

    Without pre-warming: the next UI callback (e.g. "Load / Refresh Evidence")
    has to call load_cache() synchronously, which blocks for ~4-5 seconds on the
    Gradio callback thread, causing a WebSocket timeout and browser freeze.

    Pre-warming kicks off load_cache() immediately in a daemon thread so the cache
    is already warm by the time the user clicks anything.
    """
    global _state, _model_overview_cache, _model_overview_cache_run_id
    with _state_lock:
        _state = None
    _model_overview_cache = None
    _model_overview_cache_run_id = None
    # Reload cache in background immediately — avoids blocking UI callbacks later
    threading.Thread(target=_get_state, daemon=True).start()


def _cache_ready() -> bool:
    from bt5151_credit_risk.cache import CACHE_FILE
    return CACHE_FILE.is_file()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _shap_fig(top_features: list) -> plt.Figure:
    entries = top_features[:10]
    features = [f["feature"] for f in entries]
    values = [float(f["shap_value"]) for f in entries]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(features) * 0.4)))
    ax.barh(features[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (red = pushes toward predicted class, blue = away)")
    ax.set_title("Feature contributions to this prediction")
    fig.tight_layout()
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
    # Annotate cells
    thresh = arr.max() / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]),
                    ha="center", va="center",
                    color="white" if arr[i, j] > thresh else "black", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def _predict(row_index: int):
    from bt5151_credit_risk.graph import explain_risk_node, run_inference_node
    state = _get_state()
    state_in = state.model_copy(update={"inference_input": {"row_index": row_index}})
    inf_out = run_inference_node(state_in)
    pred = inf_out["prediction_output"]
    state_pred = state_in.model_copy(update=inf_out)
    exp_out = explain_risk_node(state_pred)
    risk_exp = exp_out.get("risk_explanation") or {}
    action = exp_out.get("recommended_action") or {}
    return pred, risk_exp, action


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
    state = _get_state()
    if state is None:
        no_cache_msg = (
            "⚠️ No trained model yet — click **Train / Refresh Pipeline** to begin.\n\n"
            "Or run: `PYTHONPATH=src .venv/bin/python run_stage.py full 42 --save-cache`"
        )
        return no_cache_msg, None, None, "", "", ""

    progress(0.1, desc="Running inference …")
    try:
        pred, risk_exp, action = _predict(int(row_index))
    except Exception as exc:
        logger.exception("Inference failed for row %s", row_index)
        return f"❌ Inference error: {exc}", None, None, "", "", ""

    predicted_label = pred["predicted_label"]
    probs = pred["probabilities"]
    confidence = pred["confidence"]
    caution_info = pred.get("confidence_diagnosis") or {}
    caution = caution_info.get("caution_level", "?")
    caution_reason = caution_info.get("reason", "")

    color = RISK_COLOR.get(predicted_label, "#7f8c8d")
    emoji = CAUTION_EMOJI.get(caution, "")
    risk_md = (
        f"<div style='font-size:2em; font-weight:bold; color:{color}'>{predicted_label}</div>"
        f"<div>Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; "
        f"Caution: {emoji} <b>{caution.upper()}</b></div>"
        f"<div style='color:gray; font-size:0.85em'>{caution_reason}</div>"
    )

    progress(0.5, desc="Building charts …")
    proba_fig = _proba_fig(probs, predicted_label)

    waterfall = (pred.get("shap_waterfall") or {}).get("predicted_class_waterfall") or {}
    top_feats = waterfall.get("top_features") or []
    shap_fig = _shap_fig(top_feats) if top_feats else None

    progress(0.8, desc="Building explanation …")
    summary = risk_exp.get("summary") or risk_exp.get("risk_assessment") or ""
    key_drivers = risk_exp.get("key_drivers") or []
    drivers_lines = []
    for d in key_drivers[:5]:
        feat = d.get("feature", "")
        direction = d.get("direction", "")
        interp = d.get("interpretation") or d.get("explanation") or ""
        raw_val = d.get("raw_value")
        raw_str = f" (value: {raw_val})" if raw_val is not None else ""
        drivers_lines.append(f"- **{feat}**{raw_str}: {direction} — {interp}")

    explanation_md = summary
    if drivers_lines:
        explanation_md += "\n\n**Key drivers:**\n" + "\n".join(drivers_lines)

    action_name = action.get("action") or action.get("recommendation") or ""
    rationale = action.get("rationale") or action.get("reason") or action.get("justification") or ""
    conditions = action.get("conditions") or action.get("monitoring_conditions") or []
    cond_md = ""
    if conditions:
        cond_md = "\n\n**Conditions / monitoring:**\n" + "\n".join(f"- {c}" for c in conditions[:4])
    action_md = f"**{action_name}**\n\n{rationale}{cond_md}" if action_name else rationale

    progress(1.0, desc="Done")
    return risk_md, proba_fig, shap_fig, explanation_md, action_md, ""


# ---------------------------------------------------------------------------
# Callbacks — Tab 2: Model Evidence
# ---------------------------------------------------------------------------
def cb_model_overview():
    global _model_overview_cache, _model_overview_cache_run_id
    state = _get_state()
    if state is None:
        no_cache_df = pd.DataFrame({"Status": ["No trained model yet — click Train to begin"]})
        return no_cache_df, None, "No cache loaded.", None, "No hypothesis data."  # cm_img=None, shap_img=None

    # Return cached result if the run hasn't changed — avoids re-rendering matplotlib figures on every tab visit
    current_run_id = state.run_id
    if _model_overview_cache is not None and _model_overview_cache_run_id == current_run_id:
        return _model_overview_cache

    # Metrics table
    rows = []
    for model_name, metrics in (state.evaluation_results or {}).items():
        star = " ★" if model_name == state.selected_model_name else ""
        rows.append({
            "Model": model_name + star,
            "Macro F1": round(metrics.get("macro_f1", 0), 4),
            "Weighted F1": round(metrics.get("weighted_f1", 0), 4),
            "Accuracy": round(metrics.get("accuracy", 0), 4),
        })
    metrics_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Confusion matrix for selected model — returned as PIL Image (gr.Image), not gr.Plot.
    # gr.Plot with matplotlib goes through plotly serialization which is heavy and can
    # freeze the browser when triggered by tab.select(). PIL → gr.Image is a plain PNG.
    cm_img = None
    selected = state.selected_model_name
    if selected and (state.evaluation_results or {}).get(selected, {}).get("confusion_matrix"):
        cm = state.evaluation_results[selected]["confusion_matrix"]
        class_names = state.class_names or [str(i) for i in range(len(cm))]
        try:
            cm_img = _fig_to_pil(_confusion_matrix_fig(cm, class_names))
        except Exception as exc:
            logger.warning("Failed to build confusion matrix figure: %s", exc)

    # Global SHAP chart — also PIL Image for the same reason
    shap_img = None
    shap_importance = (state.global_xai_results or {}).get("shap", {}).get("importance") \
        or state.global_shap_importance or []
    if shap_importance:
        try:
            shap_img = _fig_to_pil(_global_shap_fig(shap_importance))
        except Exception as exc:
            logger.warning("Failed to build SHAP figure: %s", exc)

    # Justification + method gating
    methods_used = (state.global_xai_results or {}).get("methods_used") or []
    justification_md = (
        f"**Selected model:** {state.selected_model_name}\n\n"
        f"{state.selection_justification or ''}"
    )
    if methods_used:
        justification_md += f"\n\n**XAI methods used:** {', '.join(methods_used)}"

    # Hypothesis validation table
    hyp_val = (state.training_diagnostics or {}).get("hypothesis_validation") or []
    if isinstance(hyp_val, dict):
        hyp_rows = []
        for tier_name in ("tested", "supported"):
            tier_rows = hyp_val.get(tier_name) or []
            for row in tier_rows:
                if isinstance(row, dict):
                    hyp_rows.append({"tier": tier_name, **row})
                else:
                    hyp_rows.append({"tier": tier_name, "item": row})
    elif isinstance(hyp_val, list):
        hyp_rows = hyp_val
    else:
        hyp_rows = []
    if hyp_rows:
        # Build markdown table manually (avoids tabulate dependency)
        cols = list(hyp_rows[0].keys()) if isinstance(hyp_rows[0], dict) else ["item"]
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body_lines = []
        for row in hyp_rows:
            if isinstance(row, dict):
                body_lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
            else:
                body_lines.append(f"| {row} |")
        hyp_md = "\n".join([header, sep] + body_lines)
    else:
        hyp_md = "_No hypothesis validation data available._"

    result = metrics_df, cm_img, justification_md, shap_img, hyp_md
    _model_overview_cache = result
    _model_overview_cache_run_id = current_run_id
    return result


# ---------------------------------------------------------------------------
# Callbacks — Tab 3: Developer Trace
# ---------------------------------------------------------------------------
def cb_developer_trace(state=None) -> str:
    """Return markdown trace string for the Developer Trace tab.

    Trace routing:
    - If active_run.status == "running" and PID is alive → use active_run.trace_path
    - Otherwise → use state.cache_trace_path
    - Raw stage logs are fallback only when no trace artifact exists
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
    if active and active.get("status") in {"running", "failed", "completed"}:
        pid = active.get("pid")
        pid_start_time = active.get("pid_start_time", 0.0)
        if active.get("status") != "running" or (pid and _is_process_alive(pid, pid_start_time)):
            run_id = active.get("run_id", "?")
            provenance = f"**Live run** — `{run_id}`"
            if pid:
                provenance += f" (PID {pid})"
            provenance += "\n\n"
            for key in ("trace_path", "log_path"):
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


def _build_pipeline_and_md(path: "Path") -> tuple[str, str]:
    """Parse a trace artifact path and return (pipeline_html, trace_markdown)."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, build_trace_markdown, parse_trace_artifact
    try:
        trace = parse_trace_artifact(path)
        return build_pipeline_html(trace), build_trace_markdown(trace)
    except Exception as exc:
        logger.warning("Failed to parse trace at %s: %s", path, exc)
        return "", f"⚠ Failed to parse `{path.name}`: {exc}"


def cb_load_historical_log(log_name: str) -> tuple[str, str]:
    """Load a selected historical log file for inspection."""
    if not log_name:
        return "", "_No log selected._"
    log_dir = Path(__file__).resolve().parent / "lab" / "logs"
    path = log_dir / log_name
    if not path.is_file():
        return "", f"⚠ File not found: `{log_name}`"
    return _build_pipeline_and_md(path)


def cb_poll_trace() -> tuple[str, str]:
    """Polling callback for Developer Trace tab — returns (pipeline_html, trace_markdown).

    Also triggers cache reload when active_run transitions running → completed.
    """
    global _last_active_run_status

    from bt5151_credit_risk.run_status import read_active_run

    active = read_active_run()
    current_status = active.get("status") if active else None

    # Trigger cache reload on running → completed transition
    if _last_active_run_status == "running" and current_status == "completed":
        logger.info("Active run completed — invalidating cache for reload")
        _invalidate_cache()

    _last_active_run_status = current_status

    state = _get_state()
    trace_md = cb_developer_trace(state=state)

    # Build pipeline diagram from the same trace artifact (set by cb_developer_trace)
    pipeline_html = ""
    if _last_trace_artifact_path:
        p = Path(_last_trace_artifact_path)
        if p.is_file():
            pipeline_html, _ = _build_pipeline_and_md(p)

    return pipeline_html, trace_md


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
        with gr.Row():
            row_input = gr.Number(
                label=f"Customer Row Index ({idx_min}–{idx_max})",
                value=idx_default,
                minimum=idx_min,
                maximum=idx_max,
                step=1,
                precision=0,
                scale=2,
            )
            train_btn = gr.Button("🚀 Train / Refresh Pipeline", variant="primary", scale=1)
            stop_btn = gr.Button("🛑 Stop Training", variant="stop", scale=1)
            with gr.Column(scale=3):
                train_status = gr.Markdown(cache_status)

        train_btn.click(
            lambda row: cb_train_pipeline(int(row)),
            inputs=[row_input],
            outputs=[train_status],
        )
        stop_btn.click(cb_stop_pipeline, outputs=[train_status])

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
                        explanation_out = gr.Markdown()

                        gr.Markdown("### Recommended Action")
                        action_out = gr.Markdown()

                        with gr.Accordion("Feature Contributions (SHAP)", open=False):
                            shap_plot = gr.Plot(label="SHAP Waterfall")

                        error_out = gr.Markdown(visible=False)

                row_input.change(cb_load_customer, inputs=[row_input], outputs=[customer_df])
                predict_btn.click(
                    cb_predict,
                    inputs=[row_input],
                    outputs=[risk_html, proba_plot, shap_plot, explanation_out, action_out, error_out],
                )
                demo.load(cb_load_customer, inputs=[row_input], outputs=[customer_df])

            # ── Tab 2: Model Evidence ──────────────────────────────────────────
            # demo.load() is NOT used here. In Gradio 6.7+, demo.load() targeting
            # components inside an inactive tab creates a permanent loading spinner
            # because inactive tab components are lazily loaded. Instead, we use
            # tab.select() to fire cb_model_overview when Tab 2 becomes visible.
            with gr.Tab("Model Evidence") as tab_evidence:
                gr.Markdown("_Click **Load / Refresh Evidence** below to load charts from the cached model._")
                gr.Markdown("### Confusion Matrix")
                # gr.Image instead of gr.Plot: PIL PNG is a plain <img> tag, no plotly/canvas
                # serialization. gr.Plot with matplotlib can freeze the browser on tab.select().
                cm_img_out = gr.Image(label="Confusion Matrix", type="pil", interactive=False)

                gr.Markdown("### Evaluation Metrics")
                overview_btn = gr.Button("Load / Refresh Evidence", variant="primary", size="sm")
                metrics_df_out = gr.DataFrame(label="Model Comparison", interactive=False)
                justification_out = gr.Markdown()

                gr.Markdown("### Global Feature Importance")
                global_shap_img_out = gr.Image(label="Mean |SHAP|", type="pil", interactive=False)

                gr.Markdown("### Hypothesis Validation")
                gr.Markdown(
                    "EDA tested predictions vs. actual model results. "
                    "This table closes the loop between pre-training hypotheses and evidence."
                )
                hyp_validation_out = gr.Markdown()

                overview_btn.click(
                    cb_model_overview,
                    outputs=[metrics_df_out, cm_img_out, justification_out, global_shap_img_out, hyp_validation_out],
                )

            # NOTE: No tab_evidence.select() here — tab.select() fires on every click
            # and triggers a heavy matplotlib callback that freezes the browser.
            # Use the "Load / Refresh Evidence" button instead, same as Tab 1's predict button.

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
                    gr.Markdown(
                        "Live log during training · historical trace from cached run when idle · polls every 1 s",
                    )
                    from bt5151_credit_risk.ui_trace import list_available_logs
                    _log_dir = Path(__file__).resolve().parent / "lab" / "logs"
                    _initial_logs = list_available_logs(_log_dir)
                    log_selector = gr.Dropdown(
                        label="Historical run",
                        choices=_initial_logs,
                        value=None,
                        interactive=True,
                        scale=1,
                        min_width=180,
                    )

                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        gr.Markdown("#### Pipeline")
                        pipeline_diagram = gr.HTML()
                    with gr.Column(scale=3):
                        gr.Markdown("#### Log")
                        trace_out = gr.Markdown()

                log_selector.change(
                    cb_load_historical_log,
                    inputs=[log_selector],
                    outputs=[pipeline_diagram, trace_out],
                )

                trace_timer = gr.Timer(value=1)
                trace_timer.tick(
                    cb_poll_trace,
                    outputs=[pipeline_diagram, trace_out],
                    trigger_mode="always_last",
                    concurrency_limit=1,
                )

        # Load immediately on first tab click — no wait for first timer tick.
        tab_trace.select(cb_poll_trace, outputs=[pipeline_diagram, trace_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    _CSS = """
body, .gradio-container, .gradio-container * {
    font-family: Georgia, 'Times New Roman', Times, serif !important;
}
code, pre, .prose code, .token-text, .gr-markdown code {
    font-family: 'Courier New', Courier, monospace !important;
}

"""
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=_CSS)
