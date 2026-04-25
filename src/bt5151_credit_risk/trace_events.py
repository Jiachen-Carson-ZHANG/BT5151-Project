"""Structured trace event helpers for run provenance artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_trace_event_path(log_dir: Path, run_id: str) -> Path:
    """Return the JSONL path for the run's structured trace artifact."""
    return Path(log_dir) / f"trace_events_{run_id}.jsonl"


def append_trace_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one JSONL event, adding a timestamp if missing."""
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")
    return payload


def summarize_node_update(
    node_name: str,
    update: dict[str, Any],
    *,
    run_id: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Summarize a LangGraph node update into a compact structured event."""
    state_keys_written = sorted(update.keys())
    warnings = _extract_warnings(update)
    metrics = _extract_metrics(update)
    artifacts = _extract_artifacts(update)
    status = _infer_status(update, warnings)

    event: dict[str, Any] = {
        "event_type": "node_complete",
        "node": node_name,
        "status": status,
        "state_keys_written": state_keys_written,
        "warnings": warnings,
        "metrics": metrics,
        "artifacts": artifacts,
    }
    if run_id is not None:
        event["run_id"] = run_id
    if stage is not None:
        event["stage"] = stage
    return event


def _extract_warnings(update: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ("warnings", "warning", "errors", "issues", "role_violations", "cross_field_violations"):
        value = update.get(key)
        warnings.extend(_stringify_collection(value))
    for value in update.values():
        if isinstance(value, dict):
            if value.get("passed") is False:
                warnings.append("passed=False")
            if value.get("status") in {"failed", "error"}:
                warnings.append(f"status={value.get('status')}")
    return warnings


def _extract_metrics(update: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key, value in update.items():
        if _is_private_trace_key(key):
            continue
        if key in {"evaluation_results", "training_diagnostics", "global_xai_results"} and isinstance(value, dict):
            metrics.update(_flatten_numeric_dict(value, prefix=key))
        elif _is_metric_key(key) and _is_number(value):
            metrics[key] = value
        elif isinstance(value, dict):
            metrics.update(_flatten_numeric_dict(value, prefix=key))
    return metrics


def _extract_artifacts(update: dict[str, Any]) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for key, value in update.items():
        if key.endswith("_path") or key.endswith("_file") or key.endswith("_artifact"):
            if isinstance(value, (str, Path)):
                artifacts[key] = str(value)
    return artifacts


def _infer_status(update: dict[str, Any], warnings: list[str]) -> str:
    for value in update.values():
        if isinstance(value, dict) and value.get("passed") is False:
            return "fail"
        if isinstance(value, dict) and value.get("status") in {"failed", "error"}:
            return "fail"
    if warnings:
        return "warn"
    return "pass"


def _flatten_numeric_dict(value: dict[str, Any], prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        if _is_private_trace_key(key):
            continue
        name = f"{prefix}.{key}"
        if _is_number(item):
            flattened[name] = item
        elif isinstance(item, dict):
            flattened.update(_flatten_numeric_dict(item, name))
    return flattened


def _stringify_collection(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "message" in value and isinstance(value["message"], str):
            return [value["message"]]
        if "rule" in value and "column" in value:
            return [f"{value.get('rule')}:{value.get('column')}"]
        return [json.dumps(value, sort_keys=True, default=str)]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_stringify_collection(item))
        return items
    return [str(value)]


def _is_metric_key(key: str) -> bool:
    return any(
        key.endswith(suffix)
        for suffix in (
            "_score",
            "_f1",
            "_f1_score",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "best_cv_score",
            "confidence",
            "loss",
            "macro_f1_score",
        )
    )


def _is_private_trace_key(key: Any) -> bool:
    return str(key).startswith("_")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
