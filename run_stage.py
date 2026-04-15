"""Run individual pipeline stages for development and debugging.

Each stage runs the graph up to a specific node and logs results for all
nodes that ran. Later stages include earlier ones — no need to run separately.

Usage:
    python run_stage.py specs             # dataset-policy-spec + EDA + column-transform-spec
    python run_stage.py preprocess        # specs + preprocessing loop + FE loop (stops after train-models)
    python run_stage.py train             # same as preprocess (train-models is the first node after FE converges)
    python run_stage.py evaluate          # train + evaluation + model selection + SHAP
    python run_stage.py full              # entire pipeline including inference + explain + recommend
    python run_stage.py full 42           # full pipeline with row_index=42
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bt5151_credit_risk.llm import get_usage_summary, reset_usage_log

LOG_DIR = Path(__file__).resolve().parent / "lab" / "logs"

# Each stage stops the graph after this node completes.
# For loop stages, we stop at the first node AFTER the loop converges,
# because _stream_until halts on the first occurrence of the stop node —
# stopping mid-loop would exit before repair iterations run.
STAGE_STOP_AFTER = {
    "specs": "column-transform-spec",
    "preprocess": "train-models",   # Runs preprocessing loop + FE loop + train-models
    "train": "train-models",        # Same as preprocess (alias for convenience)
    "evaluate": "select-model",
    "full": None,                   # No interrupt — run to completion
}

STAGES = tuple(STAGE_STOP_AFTER.keys())


def setup_logging(stage: str):
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"stage_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s %(levelname)-7s %(name)s  %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    for name in ("httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file


def run_stage(logger, stage: str, row_index: int):
    from bt5151_credit_risk.graph import build_graph

    stop_after = STAGE_STOP_AFTER[stage]
    graph = build_graph()
    compiled = graph.compile()

    dataset_path = str(Path(__file__).resolve().parent / "train.csv")
    if not Path(dataset_path).is_file():
        logger.error("train.csv not found at %s", dataset_path)
        sys.exit(1)

    logger.info("=== Stage '%s' (stop_after=%s, row_index=%d) ===", stage, stop_after or "END", row_index)
    t0 = time.time()

    input_data = {
        "raw_dataset_path": dataset_path,
        "inference_input": {"row_index": row_index},
    }

    try:
        if stop_after is None:
            # Full run — use invoke directly.
            result = compiled.invoke(input_data)
        else:
            # Partial run — stream node outputs and stop after target node.
            result = _stream_until(compiled, input_data, stop_after, logger)
    except Exception:
        logger.exception("Pipeline failed at stage '%s'", stage)
        return

    elapsed = time.time() - t0
    logger.info("=== Stage '%s' completed in %.1fs ===", stage, elapsed)

    # Log outputs for all stages that ran (cumulative).
    _log_outputs(logger, stage, result)


def _stream_until(compiled, input_data, stop_after, logger):
    """Stream node-by-node, accumulating state updates, stop after target node."""
    accumulated = dict(input_data)
    for event in compiled.stream(input_data, stream_mode="updates"):
        # event is {node_name: state_update_dict}
        for node_name, update in event.items():
            if isinstance(update, dict):
                accumulated.update(update)
        if stop_after in event:
            logger.info("    [stream] reached stop node '%s', halting", stop_after)
            break
    return accumulated


def _log_outputs(logger, stage, result):
    """Log outputs cumulatively — later stages include all earlier outputs."""
    # Always log specs if they ran.
    _log_specs(logger, result)
    if stage == "specs":
        return

    _log_preprocess(logger, result)
    _log_fe(logger, result)
    _log_train(logger, result)
    if stage in ("preprocess", "train"):
        return

    _log_evaluate(logger, result)
    if stage == "evaluate":
        return

    _log_full(logger, result)


def _log_specs(logger, result):
    logger.info("--- dataset-policy-spec ---")
    policy = result.get("dataset_policy_spec", {})
    logger.info("  target_column: %s", policy.get("target_column"))
    logger.info("  task_type: %s", policy.get("task_type"))
    logger.info("  split_strategy: %s", policy.get("split_strategy"))

    profile = result.get("dataset_profile") or {}
    logger.info("  rows: %s, target_distribution: %s",
                profile.get("row_count"), profile.get("target_distribution"))

    logger.info("--- exploratory-data-analysis ---")
    eda = result.get("eda_report", {})
    high_pairs = eda.get("correlations", {}).get("high_pairs", [])
    logger.info("  high-correlation pairs (|r|>0.8): %d", len(high_pairs))
    for pair in high_pairs[:5]:
        logger.info("    %s <-> %s: r=%.4f", pair["col_a"], pair["col_b"], pair["correlation"])

    skewed = eda.get("skewness", {}).get("highly_skewed", {})
    logger.info("  highly skewed (|skew|>2): %s", list(skewed.keys())[:10])

    high_card = eda.get("cardinality", {}).get("high_cardinality", [])
    logger.info("  high-cardinality (>20): %s", [c["column"] for c in high_card])

    mnar = eda.get("missing_patterns", {}).get("mnar_suspects", [])
    if mnar:
        logger.info("  MNAR suspects: %s", [c["column"] for c in mnar])

    top_features = eda.get("top_discriminative_features", [])[:10]
    if top_features:
        logger.info("  top discriminative features (mutual information):")
        for f in top_features:
            logger.info("    %s: MI=%.4f", f["column"], f["mutual_information"])

    anova = eda.get("class_separability", {}).get("anova_top_features", [])[:5]
    if anova:
        logger.info("  top ANOVA features:")
        for f in anova:
            logger.info("    %s: F=%.2f (p=%.6f)", f["column"], f["f_statistic"], f["p_value"])

    logger.info("--- column-transform-spec ---")
    col_spec = result.get("column_transform_spec", {})
    transforms = col_spec.get("transforms", {})
    reasoning = col_spec.get("reasoning", {})
    kept = [k for k, v in transforms.items() if v.get("action") == "keep"]
    dropped = [k for k, v in transforms.items() if v.get("action") == "drop"]
    logger.info("  columns: %d kept, %d dropped", len(kept), len(dropped))
    if reasoning:
        logger.info("  reasoning (%d columns):", len(reasoning))
        for col, reason in list(reasoning.items())[:10]:
            logger.info("    %s: %s", col, reason)


def _log_preprocess(logger, result):
    logger.info("--- preprocessing ---")
    val = result.get("preprocessing_validation_report", {})
    logger.info("  validation passed: %s", val.get("passed"))
    logger.info("  attempt count: %d", result.get("preprocessing_attempt_count", 0))
    if result.get("feature_columns"):
        logger.info("  feature columns (%d): %s", len(result["feature_columns"]), result["feature_columns"][:15])
    if result.get("class_names"):
        logger.info("  class names: %s", result["class_names"])
    if result.get("train_frame") is not None:
        logger.info("  train/test split: %d / %d rows",
                    len(result["train_frame"]), len(result.get("test_frame", [])))


def _log_fe(logger, result):
    logger.info("--- feature-engineering ---")
    fe_val = result.get("feature_engineering_validation_report", {})
    logger.info("  validation passed: %s", fe_val.get("passed"))
    logger.info("  attempt count: %d", result.get("feature_engineering_attempt_count", 0))

    hypothesis = result.get("feature_engineering_hypothesis")
    if hypothesis:
        logger.info("  hypothesis:")
        for key, value in hypothesis.items():
            logger.info("    %s: %s", key, value)

    if result.get("feature_columns"):
        n = len(result["feature_columns"])
        logger.info("  feature columns after FE: %d", n)
        if n <= 30:
            logger.info("    %s", result["feature_columns"])


def _log_train(logger, result):
    logger.info("--- train-models ---")
    specs = result.get("candidate_model_specs", {})
    for name, spec in specs.items():
        tuning = spec.get("tuning") or {}
        if tuning:
            logger.info("  %s: best_cv=%.4f params=%s",
                        name, tuning.get("best_cv_score", 0), tuning.get("best_params", {}))
        else:
            logger.info("  %s: no tuning (defaults)", name)

    curves = result.get("learning_curves")
    if curves:
        for model_name, curve_data in curves.items():
            for key, values in curve_data.items():
                if "validation_1" in key and values:
                    best_val = min(values)
                    best_round = values.index(best_val) + 1
                    logger.info("  %s learning curve: %d rounds, best=%s at round %d, final=%s",
                                model_name, len(values), round(best_val, 6), best_round, round(values[-1], 6))

    history = result.get("tuning_trial_history")
    if history:
        for name, trials in history.items():
            valid = [t["value"] for t in trials if t["value"] is not None]
            if valid:
                logger.info("  %s: %d trials, best=%.4f, worst=%.4f",
                            name, len(trials), max(valid), min(valid))


def _log_evaluate(logger, result):
    logger.info("--- evaluate + select-model ---")
    eval_results = result.get("evaluation_results", {})
    for name, metrics in eval_results.items():
        logger.info("  %s: macro_f1=%.4f weighted_f1=%.4f accuracy=%.4f",
                    name, metrics.get("macro_f1", 0), metrics.get("weighted_f1", 0), metrics.get("accuracy", 0))
        per_class = metrics.get("per_class", {})
        for cls, cls_metrics in per_class.items():
            logger.info("    %s: P=%.3f R=%.3f F1=%.3f support=%d",
                        cls, cls_metrics.get("precision", 0), cls_metrics.get("recall", 0),
                        cls_metrics.get("f1-score", 0), cls_metrics.get("support", 0))

    logger.info("  selected model: %s", result.get("selected_model_name"))
    justification = result.get("selection_justification", "")
    if justification:
        logger.info("  justification: %s", justification)

    shap = result.get("global_shap_importance", [])
    if shap:
        logger.info("  global SHAP top-10:")
        for entry in shap[:10]:
            logger.info("    %s: %.4f", entry["feature"], entry["mean_abs_shap"])


def _log_full(logger, result):
    logger.info("--- inference + explain + recommend ---")
    prediction = result.get("prediction_output", {})
    logger.info("  prediction: %s (confidence %.4f)",
                prediction.get("predicted_label"), prediction.get("confidence", 0))

    shap_contribs = prediction.get("shap_contributions", [])
    if shap_contribs:
        logger.info("  per-prediction SHAP top-5:")
        for c in shap_contribs[:5]:
            logger.info("    %s: %s (%s)", c["feature"], c["shap_value"], c["direction"])

    explanation = result.get("risk_explanation", {})
    logger.info("  risk_level: %s, confidence_band: %s",
                explanation.get("risk_level"), explanation.get("confidence_band"))
    logger.info("  summary: %s", explanation.get("summary"))
    if explanation.get("hypothesis_notes"):
        logger.info("  hypothesis_notes: %s", explanation["hypothesis_notes"])

    action = result.get("recommended_action", {})
    logger.info("  action: %s — %s", action.get("action"), action.get("reason"))


def print_usage(logger):
    summary = get_usage_summary()
    logger.info("--- Token usage summary ---")
    logger.info("Total LLM calls: %d", summary["total_calls"])
    logger.info("Total tokens: %d (input: %d, output: %d)",
                summary["total_tokens"], summary["total_input_tokens"], summary["total_output_tokens"])
    logger.info("Total LLM duration: %.2fs", summary["total_duration_s"])
    for call in summary["calls"]:
        logger.info("  [%s] model=%s in=%d out=%d %.2fs",
                    call["caller"], call["model"],
                    call["input_tokens"], call["output_tokens"], call["duration_s"])


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        sys.exit(1)

    stage = sys.argv[1]
    row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    log_file = setup_logging(stage)
    logger = logging.getLogger("run_stage")
    logger.info("Stage: %s | Log file: %s", stage, log_file)

    reset_usage_log()
    run_stage(logger, stage, row_index)
    print_usage(logger)
    logger.info("Full log saved to: %s", log_file)


if __name__ == "__main__":
    main()
