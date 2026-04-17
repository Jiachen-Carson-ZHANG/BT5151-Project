import os

RANDOM_SEED = 42
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Production/demo default: deterministic, validated FE. Set to "llm" only when
# intentionally experimenting with generated feature-engineering code.
FEATURE_ENGINEERING_MODE = os.environ.get("BT5151_FEATURE_ENGINEERING_MODE", "deterministic").strip().lower()
