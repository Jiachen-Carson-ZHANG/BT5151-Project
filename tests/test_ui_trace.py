"""Tests for ui_trace pure log-parsing helpers."""

import json
from pathlib import Path


def test_parse_stage_log_builds_node_cards(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage_full_20260416_120000.log"
    log_path.write_text(
        "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n"
        "12:00:01 WARNING bt5151_credit_risk.graph      baseline fold skipped\n"
        "12:00:05 INFO    run_stage  --- Token usage summary ---\n"
        "12:00:05 INFO    run_stage  Total LLM calls: 12\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)

    assert trace["run_summary"]["total_llm_calls"] == 12
    assert len(trace["cards"]) >= 1
    assert trace["cards"][0]["node_name"] == "train-models"
    assert trace["cards"][0]["status"] == "warn"


def test_parse_stage_log_detects_error_status(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage.log"
    log_path.write_text(
        "12:00:00 INFO    bt5151_credit_risk.graph  >>> generate-preprocessing-code\n"
        "12:00:01 ERROR   bt5151_credit_risk.graph      code execution failed\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    assert trace["cards"][0]["status"] == "error"


def test_parse_stage_log_detects_repair_status(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage.log"
    log_path.write_text(
        "12:00:00 INFO    bt5151_credit_risk.graph  >>> repair-preprocessing-code\n"
        "12:00:01 INFO    bt5151_credit_risk.graph      attempt 2\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    assert trace["cards"][0]["node_name"] == "repair-preprocessing-code"
    assert trace["cards"][0]["status"] == "repair"


def test_parse_stage_log_pass_status_when_no_issues(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage.log"
    log_path.write_text(
        "12:00:00 INFO    bt5151_credit_risk.graph  >>> evaluate-models\n"
        "12:00:01 INFO    bt5151_credit_risk.graph      macro_f1=0.80\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    assert trace["cards"][0]["status"] == "pass"


def test_parse_stage_log_extracts_token_summary(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage.log"
    log_path.write_text(
        "12:00:00 INFO    run_stage  --- Token usage summary ---\n"
        "12:00:00 INFO    run_stage  Total LLM calls: 7\n"
        "12:00:00 INFO    run_stage  Total tokens: 150000 (input: 120000, output: 30000)\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    assert trace["run_summary"]["total_llm_calls"] == 7
    assert trace["run_summary"]["total_tokens"] == 150000


def test_parse_stage_log_missing_file_returns_empty_trace():
    from bt5151_credit_risk.ui_trace import parse_stage_log

    trace = parse_stage_log(Path("/nonexistent/path/stage.log"))
    assert trace["cards"] == []
    assert trace["run_summary"]["total_llm_calls"] == 0
    assert "error" in trace["run_summary"]


def test_read_log_tail_returns_new_content_and_offset(tmp_path):
    from bt5151_credit_risk.ui_trace import read_log_tail

    log_path = tmp_path / "stage.log"
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")

    content, offset = read_log_tail(log_path, offset=0)
    assert "line 1" in content
    assert "line 2" in content
    assert offset == len("line 1\nline 2\n")

    # Append and tail from offset
    with log_path.open("a", encoding="utf-8") as f:
        f.write("line 3\n")

    new_content, new_offset = read_log_tail(log_path, offset=offset)
    assert "line 3" in new_content
    assert "line 1" not in new_content
    assert new_offset > offset


def test_read_log_tail_missing_file_returns_empty():
    from bt5151_credit_risk.ui_trace import read_log_tail

    content, offset = read_log_tail(Path("/nonexistent/stage.log"), offset=0)
    assert content == ""
    assert offset == 0


def test_build_trace_markdown_renders_node_cards(tmp_path):
    from bt5151_credit_risk.ui_trace import build_trace_markdown, parse_stage_log

    log_path = tmp_path / "stage.log"
    log_path.write_text(
        "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n"
        "12:00:01 WARNING bt5151_credit_risk.graph      slow convergence\n"
        "12:00:05 INFO    run_stage  --- Token usage summary ---\n"
        "12:00:05 INFO    run_stage  Total LLM calls: 3\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    md = build_trace_markdown(trace)

    assert "train-models" in md
    assert "slow convergence" in md
    assert "3" in md  # total_llm_calls


def test_parse_structured_trace_jsonl_builds_node_cards(tmp_path):
    from bt5151_credit_risk.ui_trace import build_trace_markdown, parse_trace_artifact

    trace_path = tmp_path / "trace_events_20260416_123000.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "20260416_123000",
                        "stage": "full",
                        "event_type": "run_start",
                    }
                ),
                json.dumps(
                    {
                        "run_id": "20260416_123000",
                        "stage": "full",
                        "event_type": "node_complete",
                        "node": "train-models",
                        "status": "warn",
                        "state_keys_written": ["trained_models", "selected_model_name"],
                        "warnings": ["fold skipped"],
                        "metrics": {"evaluation_results.xgboost.macro_f1": 0.6943},
                        "artifacts": {"trace_path": "/tmp/trace_events_20260416_123000.jsonl"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trace = parse_trace_artifact(trace_path)
    md = build_trace_markdown(trace)

    assert trace["run_summary"]["artifact_type"] == "structured_trace"
    assert trace["run_summary"]["run_id"] == "20260416_123000"
    assert trace["run_summary"]["stage"] == "full"
    assert trace["run_summary"]["total_events"] == 2
    assert any(card["node_name"] == "train-models" and card["status"] == "warn" for card in trace["cards"])
    assert "State keys written" in md
    assert "structured trace" in md.lower()


def test_parse_structured_trace_jsonl_includes_lifecycle_events(tmp_path):
    from bt5151_credit_risk.ui_trace import build_trace_markdown, parse_trace_artifact

    trace_path = tmp_path / "trace_events_20260416_123500.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "20260416_123500",
                        "stage": "full",
                        "event_type": "run_complete",
                        "status": "pass",
                        "metrics": {"macro_f1": 0.6943},
                    }
                ),
                json.dumps(
                    {
                        "run_id": "20260416_123500",
                        "stage": "full",
                        "event_type": "cache_saved",
                        "status": "pass",
                        "artifacts": {"cache_trace_path": "/tmp/trace_events_20260416_123500.jsonl"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    trace = parse_trace_artifact(trace_path)
    md = build_trace_markdown(trace)

    assert any(card["node_name"] == "run_complete" for card in trace["cards"])
    assert any(card["node_name"] == "cache_saved" for card in trace["cards"])
    assert "Lifecycle event: run_complete" in md
    assert "Lifecycle event: cache_saved" in md
