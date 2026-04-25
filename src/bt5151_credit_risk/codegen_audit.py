from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_SECRET_KEY_TOKENS = (
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "token",
    "secret",
    "password",
    "passwd",
    "bearer",
)


def record_codegen_attempt(
    *,
    log_root: str | Path,
    run_id: str | None,
    family: str,
    attempt_label: str,
    generated_code: dict[str, Any] | None = None,
    prompt_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    execution_log: dict[str, Any] | None = None,
    validation_report: dict[str, Any] | None = None,
    audit_report: dict[str, Any] | None = None,
) -> Path:
    """Persist one codegen attempt in a stable run-scoped folder.

    The function is intentionally merge-friendly so later pipeline nodes can add
    execution, validation, and audit artifacts to the same attempt directory.
    """

    run_id = run_id or "adhoc"
    attempt_path = Path(log_root) / run_id / family / attempt_label
    attempt_path.mkdir(parents=True, exist_ok=True)

    if generated_code:
        code = generated_code.get("code", "")
        if code:
            (attempt_path / "generated.py").write_text(str(code), encoding="utf-8")

    if prompt_payload is not None:
        _write_json(attempt_path / "prompt_payload.json", _redact_obvious_secrets(prompt_payload))

    if response_payload is not None:
        _write_json(attempt_path / "response.json", _strip_private_keys(response_payload))

    if metadata is not None:
        _merge_json(attempt_path / "metadata.json", _redact_obvious_secrets(metadata))

    if execution_log is not None:
        _write_json(attempt_path / "execution_log.json", _redact_obvious_secrets(execution_log))

    if validation_report is not None:
        _write_json(attempt_path / "validation_report.json", _redact_obvious_secrets(validation_report))

    if audit_report is not None:
        _write_json(attempt_path / "audit_report.json", _redact_obvious_secrets(audit_report))

    return attempt_path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _merge_json(path: Path, payload: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    merged = dict(existing)
    merged.update(payload)
    _write_json(path, merged)


def _strip_private_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_private_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_private_keys(item) for item in value]
    return value


def _redact_obvious_secrets(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _looks_secret_key(key_str):
                redacted[key_str] = "<redacted>"
            else:
                redacted[key_str] = _redact_obvious_secrets(item, parent_key=key_str)
        return redacted
    if isinstance(value, list):
        return [_redact_obvious_secrets(item, parent_key=parent_key) for item in value]
    return value


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SECRET_KEY_TOKENS)
