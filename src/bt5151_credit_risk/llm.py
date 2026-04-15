import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from bt5151_credit_risk.config import DEFAULT_OPENAI_MODEL

logger = logging.getLogger(__name__)

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


load_dotenv()


MAX_JSON_RETRIES = 2

_usage_log: list[dict] = []


# Keep a lightweight log so we can inspect token usage and latency later.
def get_usage_log() -> list[dict]:
    return list(_usage_log)


# Summarize usage across all LLM calls in the current process.
def get_usage_summary() -> dict:
    total_input = sum(e["input_tokens"] for e in _usage_log)
    total_output = sum(e["output_tokens"] for e in _usage_log)
    total_duration = sum(e["duration_s"] for e in _usage_log)
    return {
        "total_calls": len(_usage_log),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "total_duration_s": round(total_duration, 2),
        "calls": _usage_log,
    }


# Clear the in-memory usage log between runs when needed.
def reset_usage_log():
    _usage_log.clear()


# Store a small per-call record without tying the rest of the code to OpenAI internals.
def _record_usage(response, model: str, caller: str, duration_s: float):
    usage = getattr(response, "usage", None)
    entry = {
        "model": model,
        "caller": caller,
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "duration_s": round(duration_s, 2),
    }
    _usage_log.append(entry)
    logger.info(
        "LLM call [%s] model=%s input_tokens=%d output_tokens=%d duration=%.2fs",
        entry["caller"], model, entry["input_tokens"], entry["output_tokens"], duration_s,
    )


# Call OpenAI and retry a few times if the model returns broken JSON.
def call_json_response(system_prompt: str, payload: dict, caller: str = "") -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before running the graph.")

    client = OpenAI(api_key=api_key)
    global_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    caller_slug = caller.upper().replace("-", "_") if caller else ""
    caller_env_key = f"OPENAI_MODEL_{caller_slug}" if caller_slug else ""
    model = os.getenv(caller_env_key, global_model) if caller_env_key else global_model
    # Reasoning models (o-series) accept an effort knob. Per-caller override wins;
    # otherwise fall back to a global default if set.
    effort = None
    if caller_slug:
        effort = os.getenv(f"OPENAI_REASONING_EFFORT_{caller_slug}")
    if effort is None:
        effort = os.getenv("OPENAI_REASONING_EFFORT")
    extra_kwargs = {}
    if effort and model.startswith("o"):
        extra_kwargs["reasoning"] = {"effort": effort}
    last_error = None
    # Retry only JSON parsing failures, because that is the most common structured-output miss.
    for attempt in range(MAX_JSON_RETRIES + 1):
        t0 = time.time()
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=json.dumps(payload, ensure_ascii=True, indent=2),
            **extra_kwargs,
        )
        duration = time.time() - t0
        _record_usage(response, model, caller or "unknown", duration)
        if not response.output_text:
            raise RuntimeError("OpenAI response did not include output_text.")
        text = response.output_text.strip()
        # Models often wrap JSON in markdown fences despite being told not to.
        fence_match = _MARKDOWN_FENCE_RE.match(text)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed (attempt %d/%d): %s\nRaw output (first 500 chars): %s",
                           attempt + 1, MAX_JSON_RETRIES + 1, exc, response.output_text[:500])
            last_error = exc
    raise RuntimeError(f"OpenAI response was not valid JSON after {MAX_JSON_RETRIES + 1} attempts: {last_error}")
