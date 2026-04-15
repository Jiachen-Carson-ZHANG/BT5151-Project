"""Credit Risk Analysis — Gradio Demo App.

Loads a pre-trained pipeline state from lab/cache/pipeline_state.pkl and
serves an interactive credit-risk prediction UI.

Quick start:
    # 1. Train the pipeline and save its state:
    PYTHONPATH=src .venv/bin/python run_stage.py full 42 --save-cache

    # 2. Launch the app:
    PYTHONPATH=src .venv/bin/python app.py
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
logger = logging.getLogger("app")

# ── Global pipeline state (loaded once on startup) ────────────────────────────
_state = None

RISK_COLOR = {"Good": "#27ae60", "Standard": "#f39c12", "Poor": "#e74c3c"}
CAUTION_EMOJI = {"low": "✅", "medium": "⚠️", "high": "🔴"}


def _get_state():
    global _state
    if _state is None:
        from bt5151_credit_risk.cache import load_cache
        _state = load_cache()
    return _state


def _cache_ready() -> bool:
    from bt5151_credit_risk.cache import CACHE_FILE
    return CACHE_FILE.is_file()


# ── Chart builders ─────────────────────────────────────────────────────────────
def _shap_fig(top_features: list) -> plt.Figure:
    """Horizontal bar chart of per-prediction SHAP contributions."""
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
    """Global mean |SHAP| importance bar chart."""
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
    """Probability bar chart with predicted class highlighted."""
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


# ── Inference ─────────────────────────────────────────────────────────────────
def _predict(row_index: int):
    """Run run_inference_node + explain_risk_node for a single row."""
    from bt5151_credit_risk.graph import run_inference_node, explain_risk_node

    state = _get_state()
    state_in = state.model_copy(update={"inference_input": {"row_index": row_index}})

    inf_out = run_inference_node(state_in)
    pred = inf_out["prediction_output"]

    state_pred = state_in.model_copy(update=inf_out)
    exp_out = explain_risk_node(state_pred)
    risk_exp = exp_out.get("risk_explanation") or {}
    action = exp_out.get("recommended_action") or {}

    return pred, risk_exp, action


# ── Gradio callbacks ──────────────────────────────────────────────────────────
def cb_load_customer(row_index):
    """Populate customer profile table when row index changes."""
    state = _get_state()
    if state is None:
        return pd.DataFrame({"Status": ["No cache — run the pipeline with --save-cache first"]})

    raw = state.raw_frame
    idx = int(row_index)
    if idx not in raw.index:
        return pd.DataFrame({"Status": [f"Row {idx} not in dataset (valid range: {raw.index.min()}–{raw.index.max()})"]})

    row = raw.loc[idx]
    return pd.DataFrame({"Feature": row.index.tolist(), "Value": [str(v) for v in row.values]})


def cb_predict(row_index, progress=gr.Progress()):
    """Run inference and return all UI outputs."""
    state = _get_state()
    if state is None:
        no_cache_msg = "⚠️ No pipeline cache found. Run:\n```\nPYTHONPATH=src .venv/bin/python run_stage.py full 42 --save-cache\n```"
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

    # ── Risk header ────────────────────────────────────────────────────────────
    color = RISK_COLOR.get(predicted_label, "#7f8c8d")
    emoji = CAUTION_EMOJI.get(caution, "")
    risk_md = (
        f"<div style='font-size:2em; font-weight:bold; color:{color}'>{predicted_label}</div>"
        f"<div>Confidence: <b>{confidence:.1%}</b> &nbsp;|&nbsp; "
        f"Caution: {emoji} <b>{caution.upper()}</b></div>"
        f"<div style='color:gray; font-size:0.85em'>{caution_reason}</div>"
    )

    progress(0.5, desc="Building charts …")

    # ── Probability chart ──────────────────────────────────────────────────────
    proba_fig = _proba_fig(probs, predicted_label)

    # ── SHAP chart ─────────────────────────────────────────────────────────────
    waterfall = (pred.get("shap_waterfall") or {}).get("predicted_class_waterfall") or {}
    top_feats = waterfall.get("top_features") or []
    shap_fig = _shap_fig(top_feats) if top_feats else None

    progress(0.8, desc="Building explanation …")

    # ── Explanation ────────────────────────────────────────────────────────────
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

    hyp_notes = risk_exp.get("hypothesis_notes") or risk_exp.get("hypothesis_validation") or {}
    hyp_md = ""
    if isinstance(hyp_notes, dict):
        confirmed = hyp_notes.get("confirmed") or []
        refuted = hyp_notes.get("refuted") or []
        if confirmed or refuted:
            hyp_lines = [f"✓ {h}" if isinstance(h, str) else f"✓ {h.get('hypothesis','')}" for h in confirmed[:3]]
            hyp_lines += [f"✗ {h}" if isinstance(h, str) else f"✗ {h.get('hypothesis','')}" for h in refuted[:2]]
            hyp_md = "\n\n**Hypothesis validation:**\n" + "\n".join(hyp_lines)

    explanation_md = summary
    if drivers_lines:
        explanation_md += "\n\n**Key drivers:**\n" + "\n".join(drivers_lines)
    explanation_md += hyp_md

    # ── Recommended action ─────────────────────────────────────────────────────
    action_name = action.get("action") or action.get("recommendation") or ""
    rationale = action.get("rationale") or action.get("reason") or action.get("justification") or ""
    conditions = action.get("conditions") or action.get("monitoring_conditions") or []
    cond_md = ""
    if conditions:
        cond_md = "\n\n**Conditions / monitoring:**\n" + "\n".join(f"- {c}" for c in conditions[:4])
    action_md = f"**{action_name}**\n\n{rationale}{cond_md}" if action_name else rationale

    progress(1.0, desc="Done")
    return risk_md, proba_fig, shap_fig, explanation_md, action_md, ""


def cb_model_overview():
    """Build model overview tab content."""
    state = _get_state()
    if state is None:
        return pd.DataFrame({"Error": ["No cache"]}), None, "No cache loaded."

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

    # Global SHAP chart
    shap_fig = None
    if state.global_shap_importance:
        shap_fig = _global_shap_fig(state.global_shap_importance)

    # Justification
    justification_md = (
        f"**Selected model:** {state.selected_model_name}\n\n"
        f"{state.selection_justification or ''}"
    )

    return metrics_df, shap_fig, justification_md


def cb_eda_hypotheses():
    """Show EDA hypotheses from the analysis bundle."""
    state = _get_state()
    if state is None:
        return "No cache loaded."

    hyp = state.eda_hypotheses or {}
    if not hyp:
        return "No EDA hypotheses found in cache."

    lines = []
    for tier, items in [
        ("Tested predictions", hyp.get("tested_predictions") or []),
        ("Supported conjectures", hyp.get("supported_conjectures") or []),
        ("Exploratory leads", hyp.get("exploratory_leads") or []),
    ]:
        if items:
            lines.append(f"### {tier}")
            for item in items:
                text = item if isinstance(item, str) else item.get("prediction") or item.get("hypothesis") or str(item)
                lines.append(f"- {text}")
    return "\n".join(lines) if lines else str(hyp)


# ── Build app ─────────────────────────────────────────────────────────────────
def build_app():
    state = _get_state()

    # Determine valid row range from test set index (or full frame as fallback).
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

    cache_status = "✅ Pipeline cache loaded." if state is not None else (
        "⚠️ No cache found. Run `python run_stage.py full 42 --save-cache` to generate it."
    )

    with gr.Blocks(title="Credit Risk Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Credit Risk Analysis System\n"
            "**BT5151 Group Project** — AI-powered credit risk assessment with LangGraph + LLM explanations"
        )
        gr.Markdown(cache_status)

        with gr.Tabs():
            # ── Tab 1: Customer Prediction ─────────────────────────────────────
            with gr.Tab("Customer Prediction"):
                with gr.Row():
                    # Left: input + customer info
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("### Select Customer")
                        row_input = gr.Number(
                            label=f"Row Index ({idx_min}–{idx_max})",
                            value=idx_default,
                            minimum=idx_min,
                            maximum=idx_max,
                            step=1,
                            precision=0,
                        )
                        predict_btn = gr.Button("Predict Credit Risk", variant="primary", size="lg")
                        customer_df = gr.DataFrame(
                            label="Customer Profile",
                            headers=["Feature", "Value"],
                            wrap=True,
                            interactive=False,
                        )

                    # Right: prediction outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### Prediction")
                        risk_html = gr.HTML(label="Risk Class")
                        proba_plot = gr.Plot(label="Class Probabilities")

                        gr.Markdown("### Feature Contributions (SHAP)")
                        shap_plot = gr.Plot(label="SHAP Waterfall")

                        gr.Markdown("### Explanation")
                        explanation_out = gr.Markdown()

                        gr.Markdown("### Recommended Action")
                        action_out = gr.Markdown()

                        error_out = gr.Markdown(visible=False)

                # Wire events
                row_input.change(cb_load_customer, inputs=[row_input], outputs=[customer_df])

                predict_btn.click(
                    cb_predict,
                    inputs=[row_input],
                    outputs=[risk_html, proba_plot, shap_plot, explanation_out, action_out, error_out],
                )

                # Pre-load customer profile on page load
                demo.load(cb_load_customer, inputs=[row_input], outputs=[customer_df])

            # ── Tab 2: Model Overview ──────────────────────────────────────────
            with gr.Tab("Model Overview"):
                gr.Markdown("### Model Evaluation & Selection")
                overview_btn = gr.Button("Refresh", variant="secondary", size="sm")
                justification_out = gr.Markdown()
                metrics_df_out = gr.DataFrame(label="Evaluation Metrics", interactive=False)
                global_shap_plot = gr.Plot(label="Global Feature Importance")

                overview_btn.click(
                    cb_model_overview,
                    outputs=[metrics_df_out, global_shap_plot, justification_out],
                )
                demo.load(
                    cb_model_overview,
                    outputs=[metrics_df_out, global_shap_plot, justification_out],
                )

            # ── Tab 3: EDA Hypotheses ──────────────────────────────────────────
            with gr.Tab("EDA Hypotheses"):
                gr.Markdown(
                    "Three-tier hypotheses generated from EDA before model training. "
                    "**Tested** = closed-loop verifiable; **Supported** = partially testable; "
                    "**Exploratory** = open threads."
                )
                hyp_out = gr.Markdown()
                demo.load(cb_eda_hypotheses, outputs=[hyp_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
