import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from bt5151_credit_risk.config import DEFAULT_OPENAI_MODEL


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


# Call OpenAI and retry a few times if the model returns broken JSON.
def call_json_response(system_prompt: str, payload: dict, caller: str = "") -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before running the graph.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    last_error = None
    # Retry only JSON parsing failures, because that is the most common structured-output miss.
    for attempt in range(MAX_JSON_RETRIES + 1):
        t0 = time.time()
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=json.dumps(payload, ensure_ascii=True, indent=2),
        )
        duration = time.time() - t0
        _record_usage(response, model, caller or "unknown", duration)
        if not response.output_text:
            raise RuntimeError("OpenAI response did not include output_text.")
        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"OpenAI response was not valid JSON after {MAX_JSON_RETRIES + 1} attempts: {last_error}")
