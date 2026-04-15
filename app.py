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
import subprocess
import sys
import threading
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

# Track last-seen active run status for cache-reload trigger (Task 9)
_last_active_run_status: str | None = None

RISK_COLOR = {"Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c"}
CAUTION_EMOJI = {"low": "✅", "medium": "⚠️", "high": "🔴"}


def _get_state():
    global _state
    with _state_lock:
        if _state is None:
            from bt5151_credit_risk.cache import load_cache
            _state = load_cache()
        return _state


def _invalidate_cache() -> None:
    """Clear the in-memory state so the next _get_state() call reloads from disk."""
    global _state
    with _state_lock:
        _state = None


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
    fig, ax = plt.subplots(figsize=(9, max(3, len(features) * 0.5)))
    ax.barh(features[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (red = pushes toward predicted class, blue = away)")
    ax.set_title("Feature contributions to this prediction")
    fig.tight_layout()
    return fig


def _global_shap_fig(importance: list) -> plt.Figure:
    entries = importance[:15]
    features = [e["feature"] for e in entries]
    values = [float(e["mean_abs_shap"]) for e in entries]
    fig, ax = plt.subplots(figsize=(9, max(4, len(features) * 0.45)))
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
    fig, ax = plt.subplots(figsize=(5, 4))
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
    state = _get_state()
    if state is None:
        no_cache_df = pd.DataFrame({"Status": ["No trained model yet — click Train to begin"]})
        return no_cache_df, None, "No cache loaded.", None, "No hypothesis data."

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

    # Confusion matrix for selected model
    cm_fig = None
    selected = state.selected_model_name
    if selected and (state.evaluation_results or {}).get(selected, {}).get("confusion_matrix"):
        cm = state.evaluation_results[selected]["confusion_matrix"]
        class_names = state.class_names or [str(i) for i in range(len(cm))]
        try:
            cm_fig = _confusion_matrix_fig(cm, class_names)
        except Exception as exc:
            logger.warning("Failed to build confusion matrix figure: %s", exc)

    # Global SHAP chart
    shap_fig = None
    shap_importance = (state.global_xai_results or {}).get("shap", {}).get("importance") \
        or state.global_shap_importance or []
    if shap_importance:
        shap_fig = _global_shap_fig(shap_importance)

    # Justification + method gating
    methods_used = (state.global_xai_results or {}).get("methods_used") or []
    justification_md = (
        f"**Selected model:** {state.selected_model_name}\n\n"
        f"{state.selection_justification or ''}"
    )
    if methods_used:
        justification_md += f"\n\n**XAI methods used:** {', '.join(methods_used)}"

    # Hypothesis validation table
    hyp_rows = (state.training_diagnostics or {}).get("hypothesis_validation") or []
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

    return metrics_df, cm_fig, justification_md, shap_fig, hyp_md


# ---------------------------------------------------------------------------
# Callbacks — Tab 3: Developer Trace
# ---------------------------------------------------------------------------
def cb_developer_trace(state=None) -> str:
    """Return markdown trace string for the Developer Trace tab.

    Log routing:
    - If active_run.status == "running" and PID is alive → use active_run.log_path
    - Otherwise → use state.cache_log_path (the log bound at cache-save time)
    - Never guesses "latest log on disk"
    """
    from bt5151_credit_risk.run_status import _is_process_alive, read_active_run
    from bt5151_credit_risk.ui_trace import build_trace_markdown, parse_stage_log

    active = read_active_run()

    # Determine which log to read
    log_path = None
    provenance_header = ""

    if active and active.get("status") == "running":
        pid = active.get("pid")
        pid_start_time = active.get("pid_start_time", 0.0)
        if pid and _is_process_alive(pid, pid_start_time):
            log_path = active.get("log_path")
            run_id = active.get("run_id", "?")
            provenance_header = f"**Live run** — `{run_id}` (PID {pid})\n\n"

    if log_path is None and state is not None:
        log_path = getattr(state, "cache_log_path", None)
        if log_path:
            run_id = getattr(state, "run_id", "?")
            provenance_header = f"**Cached run** — `{run_id}`\n\n"

    if not log_path:
        return "⚠ No log available. Run the pipeline with `--save-cache` first."

    from pathlib import Path
    path = Path(log_path)
    if not path.is_file():
        return f"⚠ Log not found: `{log_path}`"

    trace = parse_stage_log(path)
    md = build_trace_markdown(trace)
    return provenance_header + md


def cb_train_pipeline(row_index: int) -> str:
    """Start a background pipeline training run.

    Returns a status message. Does NOT write active_run.json — that is
    run_stage.py's job (Task 3).
    """
    from bt5151_credit_risk.run_status import _is_process_alive, read_active_run

    # Concurrent run guard
    active = read_active_run()
    if active and active.get("status") == "running":
        pid = active.get("pid")
        pid_start_time = active.get("pid_start_time", 0.0)
        if pid and _is_process_alive(pid, pid_start_time):
            run_id = active.get("run_id", "?")
            return f"⚠ Training already in progress (run {run_id}). Check Developer Trace tab."

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


def cb_poll_trace() -> str:
    """Polling callback for Developer Trace tab — called every 5 seconds by Gradio.

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
    return cb_developer_trace(state=state)


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

    has_cache = state is not None
    cache_status = (
        f"✅ Cache loaded — model: **{state.selected_model_name}**  |  run: `{state.run_id or '?'}`"
        if has_cache
        else "⚠️ No trained model yet — click **Train / Refresh Pipeline** to begin."
    )

    with gr.Blocks(title="Credit Risk Analysis", theme=gr.themes.Soft()) as demo:
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
            train_status = gr.Markdown(cache_status, scale=3)

        train_btn.click(
            lambda row: cb_train_pipeline(int(row)),
            inputs=[row_input],
            outputs=[train_status],
        )

        with gr.Tabs():
            # ── Tab 1: Business View ───────────────────────────────────────────
            with gr.Tab("Business View"):
                if not has_cache:
                    gr.Markdown(
                        "### No trained model yet\n"
                        "Click **Train / Refresh Pipeline** above to train the model."
                    )
                else:
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
            with gr.Tab("Model Evidence"):
                if not has_cache:
                    gr.Markdown(
                        "### No trained model yet\n"
                        "Click **Train / Refresh Pipeline** above to train the model."
                    )
                else:
                    gr.Markdown("### Confusion Matrix")
                    cm_plot = gr.Plot(label="Confusion Matrix")

                    gr.Markdown("### Evaluation Metrics")
                    overview_btn = gr.Button("Refresh", variant="secondary", size="sm")
                    metrics_df_out = gr.DataFrame(label="Model Comparison", interactive=False)
                    justification_out = gr.Markdown()

                    gr.Markdown("### Global Feature Importance")
                    global_shap_plot = gr.Plot(label="Mean |SHAP|")

                    gr.Markdown("### Hypothesis Validation")
                    gr.Markdown(
                        "EDA tested predictions vs. actual model results. "
                        "This table closes the loop between pre-training hypotheses and evidence."
                    )
                    hyp_validation_out = gr.Markdown()

                    overview_btn.click(
                        cb_model_overview,
                        outputs=[metrics_df_out, cm_plot, justification_out, global_shap_plot, hyp_validation_out],
                    )
                    demo.load(
                        cb_model_overview,
                        outputs=[metrics_df_out, cm_plot, justification_out, global_shap_plot, hyp_validation_out],
                    )

            # ── Tab 3: Developer Trace ─────────────────────────────────────────
            with gr.Tab("Developer Trace"):
                gr.Markdown(
                    "Live log during training. Historical trace from cached run when idle.\n"
                    "Polls every 5 seconds — bound to the explicit `run_id`, not 'latest log'."
                )
                trace_out = gr.Markdown()

                # Initial load
                demo.load(cb_poll_trace, outputs=[trace_out])

                # 5-second polling
                try:
                    timer = gr.Timer(value=5)
                    timer.tick(cb_poll_trace, outputs=[trace_out])
                except AttributeError:
                    # Older Gradio versions without gr.Timer: use demo.load every
                    demo.load(cb_poll_trace, outputs=[trace_out], every=5)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
