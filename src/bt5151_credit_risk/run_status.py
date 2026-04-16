"""Active run status contract.

Writes and reads lab/cache/active_run.json, which tracks in-progress and
recently-completed pipeline runs so the Gradio app can poll status without
blocking the UI thread.

run_stage.py is the sole writer of this file. app.py is read-only.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)

ACTIVE_RUN_FILE = Path(__file__).resolve().parent.parent.parent / "lab" / "cache" / "active_run.json"


def _is_process_alive(pid: int, pid_start_time: float) -> bool:
    """Return True only if PID exists and its create_time matches pid_start_time.

    Handles Linux/WSL PID reuse: a new unrelated process may inherit the same
    PID after the original exits. The start_time comparison (1s tolerance)
    distinguishes the original process from a reused-PID impostor.
    """
    try:
        p = psutil.Process(pid)
        return abs(p.create_time() - pid_start_time) < 1.0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def write_active_run(
    run_id: str,
    stage: str,
    row_index: int,
    log_path: str,
    bundle_path: str,
    trace_path: str,
) -> None:
    """Write a running-status record to ACTIVE_RUN_FILE.

    Should be called by run_stage.py immediately after spawning (or at the
    start of) a pipeline run. The caller's own PID and start_time are stored
    so read_active_run() can verify the process is still alive.
    """
    ACTIVE_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    pid_start_time = psutil.Process(pid).create_time()
    record = {
        "run_id": run_id,
        "stage": stage,
        "row_index": row_index,
        "status": "running",
        "pid": pid,
        "pid_start_time": pid_start_time,
        "log_path": log_path,
        "bundle_path": bundle_path,
        "trace_path": trace_path,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    ACTIVE_RUN_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("active_run written: run_id=%s stage=%s", run_id, stage)


def mark_run_completed(run_id: str) -> None:
    """Update the status field to 'completed'."""
    record = _load_raw()
    if record is None:
        logger.warning("mark_run_completed called but no active_run.json found")
        return
    if record.get("run_id") != run_id:
        logger.warning(
            "mark_run_completed called for run_id=%s but active_run.json belongs to %s; no-op",
            run_id,
            record.get("run_id"),
        )
        return
    record["status"] = "completed"
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    ACTIVE_RUN_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("active_run marked completed: run_id=%s", run_id)


def mark_run_failed(run_id: str, error: str) -> None:
    """Update the status field to 'failed' with an error summary."""
    record = _load_raw()
    if record is None:
        logger.warning("mark_run_failed called but no active_run.json found")
        return
    if record.get("run_id") != run_id:
        logger.warning(
            "mark_run_failed called for run_id=%s but active_run.json belongs to %s; no-op",
            run_id,
            record.get("run_id"),
        )
        return
    record["status"] = "failed"
    record["error"] = error
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    ACTIVE_RUN_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("active_run marked failed: run_id=%s error=%s", run_id, error)


def read_active_run() -> dict | None:
    """Read and return the active run record, or None if no file exists.

    If the record says 'running' but the stored PID is no longer alive (or has
    been reused by a different process), this function rewrites the record to
    'failed' with error='process died' before returning it. This ensures the
    caller always sees an accurate status.
    """
    record = _load_raw()
    if record is None:
        return None

    if record.get("status") == "running":
        pid = record.get("pid")
        pid_start_time = record.get("pid_start_time")
        if pid is None or pid_start_time is None or not _is_process_alive(pid, pid_start_time):
            record["status"] = "failed"
            record["error"] = "process died"
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            ACTIVE_RUN_FILE.write_text(json.dumps(record, indent=2), encoding="utf-8")
            logger.warning("active_run PID %s no longer alive — marked failed", pid)

    return record


def _load_raw() -> dict | None:
    """Load raw JSON from ACTIVE_RUN_FILE, returning None if missing or invalid."""
    if not ACTIVE_RUN_FILE.is_file():
        return None
    try:
        return json.loads(ACTIVE_RUN_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read active_run.json: %s", exc)
        return None
