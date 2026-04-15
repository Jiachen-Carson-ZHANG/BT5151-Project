"""Pipeline state cache — serialize and reload trained pipeline artifacts.

After a full pipeline run, call save_cache(result) to persist the state to
lab/cache/pipeline_state.pkl.  The Gradio app (app.py) calls load_cache() on
startup to restore a CreditRiskState without re-running training.

Keys preserved:
  - trained_models, selected_model_name, selection_justification
  - raw_frame, full_feature_frame, full_feature_frames_by_view
  - test_frame, test_views, train_frame, train_views
  - test_target, train_target
  - class_names, id_to_label, label_to_id
  - feature_columns, feature_columns_by_view, model_view_map
  - global_xai_results, local_xai_cases, global_shap_importance
  - training_diagnostics, evaluation_results
  - analysis_bundle, analysis_bundle_summary
  - eda_hypotheses, feature_engineering_hypothesis
"""

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "lab" / "cache"
CACHE_FILE = CACHE_DIR / "pipeline_state.pkl"

# Fields extracted from the LangGraph result dict and stored in the cache.
CACHE_KEYS = [
    # Models
    "trained_models",
    "selected_model_name",
    "selection_justification",
    "candidate_model_specs",
    # Frames
    "raw_frame",
    "full_feature_frame",
    "full_feature_frames_by_view",
    "test_frame",
    "test_views",
    "train_frame",
    "train_views",
    "test_target",
    "train_target",
    # Label encoding
    "class_names",
    "id_to_label",
    "label_to_id",
    # Feature metadata
    "feature_columns",
    "feature_columns_by_view",
    "model_view_map",
    # XAI
    "global_xai_results",
    "local_xai_cases",
    "global_shap_importance",
    # Diagnostics / metrics
    "training_diagnostics",
    "evaluation_results",
    # Analysis bundle
    "analysis_bundle",
    "analysis_bundle_summary",
    # Hypotheses (for display)
    "eda_hypotheses",
    "feature_engineering_hypothesis",
    # Dataset metadata
    "dataset_profile",
    "eda_report",
    # Run provenance
    "run_id",
    "cache_log_path",
    "cache_bundle_path",
    "cache_saved_at",
]


def save_cache(result: dict, metadata: dict | None = None, compress: int = 3) -> Path:
    """Serialize pipeline state to CACHE_FILE.

    Args:
        result: Dict returned by compiled.invoke() or accumulated from stream.
        metadata: Optional provenance metadata (cache_log_path, cache_bundle_path,
            cache_saved_at) to merge into the stored payload.
        compress: joblib compress level (0-9); 3 is a good speed/size balance.

    Returns:
        Path to the saved cache file.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: result.get(k) for k in CACHE_KEYS}
    if metadata:
        for k, v in metadata.items():
            payload[k] = v
    # Log what's present and what's missing so the user knows what will work.
    present = [k for k in CACHE_KEYS if payload[k] is not None]
    missing = [k for k in CACHE_KEYS if payload[k] is None]
    logger.info("    cache: saving %d fields (%d None)", len(present), len(missing))
    if missing:
        logger.debug("    cache: missing fields: %s", missing)
    joblib.dump(payload, CACHE_FILE, compress=compress)
    size_mb = CACHE_FILE.stat().st_size / 1_048_576
    logger.info("    cache saved → %s (%.1f MB)", CACHE_FILE, size_mb)
    return CACHE_FILE


def load_cache() -> "CreditRiskState | None":
    """Load cached pipeline state from CACHE_FILE.

    Returns a CreditRiskState instance, or None if the cache file does not exist.
    """
    from bt5151_credit_risk.state import CreditRiskState

    if not CACHE_FILE.is_file():
        logger.warning("No pipeline cache found at %s. Run the pipeline with --save-cache first.", CACHE_FILE)
        return None
    logger.info("Loading pipeline cache from %s ...", CACHE_FILE)
    payload = joblib.load(CACHE_FILE)
    # raw_dataset_path is required by CreditRiskState but not meaningful at inference time.
    payload.setdefault("raw_dataset_path", "")
    state = CreditRiskState(**{k: v for k, v in payload.items() if k in CreditRiskState.model_fields})
    logger.info("Pipeline cache loaded — selected model: %s, class_names: %s",
                state.selected_model_name, state.class_names)
    return state
