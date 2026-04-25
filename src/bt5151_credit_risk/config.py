import os

RANDOM_SEED = 42
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Production/demo default: deterministic, validated FE. Set to "llm" only when
# intentionally experimenting with generated feature-engineering code.
FEATURE_ENGINEERING_MODE = os.environ.get("BT5151_FEATURE_ENGINEERING_MODE", "deterministic").strip().lower()

# Tuning hygiene controls. Defaults preserve current behavior unless explicitly
# overridden for stronger or cheaper searches.
TUNING_SUBSAMPLE_MAX = int(os.environ.get("BT5151_TUNING_SUBSAMPLE_MAX", "15000"))
TUNING_CV_FOLDS = int(os.environ.get("BT5151_TUNING_CV_FOLDS", "5"))
OPTUNA_TRIALS = int(os.environ.get("BT5151_OPTUNA_TRIALS", "10"))
TUNING_MAX_DEPTH_CAP = int(os.environ.get("BT5151_TUNING_MAX_DEPTH_CAP", "20"))
RF_TUNING_ESTIMATORS = int(os.environ.get("BT5151_RF_TUNING_ESTIMATORS", "100"))
XGB_TUNING_ESTIMATORS = int(os.environ.get("BT5151_XGB_TUNING_ESTIMATORS", "500"))
XGB_TUNING_EARLY_STOPPING_ROUNDS = int(os.environ.get("BT5151_XGB_TUNING_EARLY_STOPPING_ROUNDS", "30"))
XGB_FINAL_EARLY_STOPPING_ESTIMATORS = int(
    os.environ.get("BT5151_XGB_FINAL_EARLY_STOPPING_ESTIMATORS", "1500")
)
XGB_FINAL_EARLY_STOPPING_ROUNDS = int(
    os.environ.get("BT5151_XGB_FINAL_EARLY_STOPPING_ROUNDS", "75")
)
