"""Run the classification pipeline end-to-end and print a development trace."""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the source package is importable when running from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bt5151_credit_risk.graph import compile_graph
from bt5151_credit_risk.llm import get_usage_summary, reset_usage_log


LOG_DIR = Path(__file__).resolve().parent / "logs"


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s %(levelname)-7s %(name)s  %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console — INFO level.
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File — DEBUG level (captures everything).
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers.
    for name in ("httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_file


def main():
    log_file = setup_logging()
    logger = logging.getLogger("run_pipeline")
    logger.info("Log file: %s", log_file)

    dataset_path = str(Path(__file__).resolve().parent / "train.csv")
    if not Path(dataset_path).is_file():
        logger.error("train.csv not found at %s", dataset_path)
        sys.exit(1)

    row_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    reset_usage_log()
    graph = compile_graph()

    logger.info("=== Pipeline starting (row_index=%d) ===", row_index)
    t0 = time.time()

    try:
        result = graph.invoke({
            "raw_dataset_path": dataset_path,
            "inference_input": {"row_index": row_index},
        })
    except Exception:
        logger.exception("Pipeline failed")
        _print_usage_summary(logger)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("=== Pipeline finished in %.1fs ===", elapsed)

    # Print key outputs.
    prediction = result.get("prediction_output", {})
    logger.info("Prediction: %s (confidence %.4f)",
                prediction.get("predicted_label"), prediction.get("confidence", 0))
    logger.info("Risk explanation: %s", json.dumps(result.get("risk_explanation", {}), indent=2))
    logger.info("Recommended action: %s", json.dumps(result.get("recommended_action", {}), indent=2))

    _print_usage_summary(logger)
    logger.info("Full log saved to: %s", log_file)


def _print_usage_summary(logger):
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


if __name__ == "__main__":
    main()
