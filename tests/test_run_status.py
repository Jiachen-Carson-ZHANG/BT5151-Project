"""Tests for active run status contract."""

import json
import os
import time

import pytest


@pytest.fixture(autouse=True)
def patch_active_run_file(tmp_path, monkeypatch):
    """Redirect ACTIVE_RUN_FILE to a temp path for all tests."""
    import bt5151_credit_risk.run_status as rs_mod

    monkeypatch.setattr(rs_mod, "ACTIVE_RUN_FILE", tmp_path / "active_run.json")


def test_active_run_round_trip(tmp_path):
    from bt5151_credit_risk.run_status import (
        mark_run_completed,
        read_active_run,
        write_active_run,
    )

    write_active_run(
        run_id="20260416_120000",
        stage="full",
        row_index=42,
        log_path="/tmp/stage_full_20260416_120000.log",
        bundle_path="/tmp/analysis_bundle_20260416_120000.json",
    )
    status = read_active_run()
    assert status["status"] == "running"
    assert status["run_id"] == "20260416_120000"
    assert status["row_index"] == 42
    assert "pid" in status
    assert "pid_start_time" in status

    mark_run_completed("20260416_120000")
    status = read_active_run()
    assert status["status"] == "completed"
    assert status["completed_at"] is not None


def test_mark_run_failed(tmp_path):
    from bt5151_credit_risk.run_status import (
        mark_run_failed,
        read_active_run,
        write_active_run,
    )

    write_active_run(
        run_id="20260416_130000",
        stage="full",
        row_index=5,
        log_path="/tmp/stage.log",
        bundle_path="/tmp/bundle.json",
    )
    mark_run_failed("20260416_130000", error="OutOfMemory")
    status = read_active_run()
    assert status["status"] == "failed"
    assert "OutOfMemory" in status["error"]


def test_read_active_run_returns_none_when_missing(tmp_path):
    from bt5151_credit_risk.run_status import read_active_run

    result = read_active_run()
    assert result is None


def test_stale_pid_rewritten_to_failed(tmp_path, monkeypatch):
    """A dead PID causes read_active_run to rewrite status to failed."""
    import bt5151_credit_risk.run_status as rs_mod
    from bt5151_credit_risk.run_status import read_active_run

    # Write a record with a dead PID (use PID 1 with a bogus start_time so
    # the create_time mismatch triggers the stale-PID path).
    fake_record = {
        "run_id": "20260416_140000",
        "stage": "full",
        "row_index": 0,
        "status": "running",
        "pid": os.getpid(),
        "pid_start_time": 0.0,  # wrong start time → treated as stale
        "log_path": "/tmp/stage.log",
        "bundle_path": "/tmp/bundle.json",
        "started_at": "2026-04-16T14:00:00Z",
        "completed_at": None,
        "error": None,
    }
    rs_mod.ACTIVE_RUN_FILE.write_text(json.dumps(fake_record), encoding="utf-8")

    status = read_active_run()
    assert status["status"] == "failed"
    assert "process died" in status["error"]


def test_is_process_alive_current_process():
    """_is_process_alive returns True for the current process with correct start_time."""
    import psutil

    from bt5151_credit_risk.run_status import _is_process_alive

    pid = os.getpid()
    start_time = psutil.Process(pid).create_time()
    assert _is_process_alive(pid, start_time) is True


def test_is_process_alive_wrong_start_time():
    """_is_process_alive returns False when start_time is wrong (stale PID)."""
    from bt5151_credit_risk.run_status import _is_process_alive

    pid = os.getpid()
    assert _is_process_alive(pid, 0.0) is False
