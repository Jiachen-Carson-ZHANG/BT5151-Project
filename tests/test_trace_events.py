"""Tests for structured trace event artifacts."""

import json
import logging


def test_build_trace_event_path_uses_run_id(tmp_path):
    from bt5151_credit_risk.trace_events import build_trace_event_path

    path = build_trace_event_path(tmp_path, "20260416_123000")
    assert path == tmp_path / "trace_events_20260416_123000.jsonl"


def test_append_trace_event_writes_one_json_line(tmp_path):
    from bt5151_credit_risk.trace_events import append_trace_event

    path = tmp_path / "trace_events.jsonl"
    written = append_trace_event(
        path,
        {
            "run_id": "20260416_123000",
            "node": "train-models",
            "event_type": "node_complete",
            "status": "pass",
            "state_keys_written": ["trained_models"],
        },
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["node"] == "train-models"
    assert payload["event_type"] == "node_complete"
    assert payload["timestamp"]
    assert written["run_id"] == "20260416_123000"


def test_summarize_node_update_extracts_structured_fields():
    from bt5151_credit_risk.trace_events import summarize_node_update

    event = summarize_node_update(
        "validate-preprocessing-output",
        {
            "preprocessing_validation_report": {
                "passed": False,
                "errors": [{"rule": "target_excluded", "message": "target still present"}],
            },
            "evaluation_results": {
                "xgboost": {"macro_f1": 0.6943, "weighted_f1": 0.7012},
            },
            "cache_bundle_path": "/tmp/analysis_bundle.json",
        },
        run_id="20260416_123000",
        stage="preprocess",
    )

    assert event["node"] == "validate-preprocessing-output"
    assert event["status"] == "fail"
    assert event["metrics"]["evaluation_results.xgboost.macro_f1"] == 0.6943
    assert event["artifacts"]["cache_bundle_path"] == "/tmp/analysis_bundle.json"
    assert "preprocessing_validation_report" in event["state_keys_written"]


def test_summarize_node_update_ignores_private_codegen_audit_payload_in_metrics():
    from bt5151_credit_risk.trace_events import summarize_node_update

    event = summarize_node_update(
        "generate-preprocessing-code",
        {
            "preprocessing_code": {
                "entrypoint": "run_preprocessing",
                "_codegen_audit": {
                    "prompt_payload": {
                        "dataset_profile": {"row_count": 100000},
                    }
                },
            },
            "preprocessing_codegen_snapshot_path": "/tmp/codegen/preprocessing",
        },
        run_id="20260425_120000",
        stage="preprocess",
    )

    assert event["artifacts"]["preprocessing_codegen_snapshot_path"] == "/tmp/codegen/preprocessing"
    assert not any("_codegen_audit" in key for key in event["metrics"])
    assert "preprocessing_code" in event["state_keys_written"]


def test_stream_until_appends_trace_events(tmp_path):
    import run_stage as rs

    class FakeCompiled:
        def stream(self, input_data, stream_mode="updates"):
            yield {"node-a": {"trained_models": {"x": 1}}}
            yield {
                "node-b": {
                    "preprocessing_validation_report": {
                        "passed": False,
                        "errors": [{"rule": "x", "message": "bad"}],
                    }
                }
            }

    trace_path = tmp_path / "trace_events_20260416_123000.jsonl"
    result = rs._stream_until(
        FakeCompiled(),
        {"seed": 1},
        stop_after=None,
        logger=logging.getLogger("test_trace_events"),
        trace_path=trace_path,
        run_id="20260416_123000",
        stage="full",
    )

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert result["trained_models"] == {"x": 1}
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["node"] == "node-a"
    assert second["node"] == "node-b"
    assert second["status"] == "fail"


def test_main_marks_completed_even_if_cache_save_fails(tmp_path, monkeypatch):
    import sys

    import bt5151_credit_risk.run_status as rs_status
    import run_stage as rs

    monkeypatch.setattr(rs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(rs, "setup_logging", lambda stage, run_id: tmp_path / f"stage_{stage}_{run_id}.log")
    monkeypatch.setattr(rs, "run_stage", lambda logger, stage, row_index, run_id, trace_path=None: {"selected_model_name": "xgboost"})
    monkeypatch.setattr(rs, "print_usage", lambda logger: None)
    monkeypatch.setattr(
        "bt5151_credit_risk.cache.save_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
    )
    monkeypatch.setattr(rs_status, "ACTIVE_RUN_FILE", tmp_path / "active_run.json")
    monkeypatch.setattr(sys, "argv", ["run_stage.py", "full", "42", "--save-cache"])

    rs.main()

    record = json.loads((tmp_path / "active_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["trace_path"].endswith(".jsonl")


def test_main_marks_completed_even_if_cache_trace_append_fails(tmp_path, monkeypatch):
    import sys

    import bt5151_credit_risk.run_status as rs_status
    import run_stage as rs

    monkeypatch.setattr(rs, "LOG_DIR", tmp_path)
    monkeypatch.setattr(rs, "setup_logging", lambda stage, run_id: tmp_path / f"stage_{stage}_{run_id}.log")
    monkeypatch.setattr(rs, "run_stage", lambda logger, stage, row_index, run_id, trace_path=None: {"selected_model_name": "xgboost"})
    monkeypatch.setattr(rs, "print_usage", lambda logger: None)
    monkeypatch.setattr(
        "bt5151_credit_risk.cache.save_cache",
        lambda result, metadata=None, compress=3: tmp_path / "pipeline_state.pkl",
    )

    def fake_append_trace_event(path, event):
        if event.get("event_type") == "cache_saved":
            raise RuntimeError("trace write failed")
        return event

    monkeypatch.setattr(rs, "append_trace_event", fake_append_trace_event)
    monkeypatch.setattr(rs_status, "ACTIVE_RUN_FILE", tmp_path / "active_run.json")
    monkeypatch.setattr(sys, "argv", ["run_stage.py", "full", "42", "--save-cache"])

    rs.main()

    record = json.loads((tmp_path / "active_run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["trace_path"].endswith(".jsonl")
