"""Tests for ui_trace pure log-parsing helpers."""

import json
import os
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


def test_parse_stage_log_synthesizes_run_start_and_run_complete(tmp_path):
    from bt5151_credit_risk.ui_trace import parse_stage_log

    log_path = tmp_path / "stage_success.log"
    log_path.write_text(
        "12:00:00 INFO    run_stage  === Stage 'full' (stop_after=END, row_index=42) ===\n"
        "12:00:01 INFO    bt5151_credit_risk.graph  >>> recommend-action\n"
        "12:00:02 INFO    bt5151_credit_risk.graph      action: standard_handling\n"
        "12:00:03 INFO    run_stage  === Stage 'full' completed in 3.0s ===\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    node_names = [card["node_name"] for card in trace["cards"]]

    assert node_names[0] == "run_start"
    assert node_names[-1] == "run_complete"


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
    assert "Lifecycle event" in md and "run_complete" in md
    assert "Lifecycle event" in md and "cache_saved" in md


def test_build_pipeline_html_renders_retry_nodes_in_execution_order(tmp_path):
    """Conditional repair-loop nodes should appear where they ran, not at the bottom."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_trace_artifact

    trace_path = tmp_path / "trace_events_retry.jsonl"
    events = [
        {"event_type": "run_start", "status": "pass", "node": "__run__"},
        {"event_type": "node_complete", "node": "generate-preprocessing-code", "status": "pass"},
        {"event_type": "node_complete", "node": "inspect-preprocessing-code", "status": "pass"},
        {"event_type": "node_complete", "node": "execute-generated-preprocessing", "status": "pass"},
        {"event_type": "node_complete", "node": "validate-preprocessing-output", "status": "fail"},
        {"event_type": "node_complete", "node": "review-preprocessing-quality", "status": "pass"},
        {"event_type": "node_complete", "node": "repair-preprocessing-code", "status": "pass"},
        {"event_type": "node_complete", "node": "inspect-preprocessing-code", "status": "pass"},
        {"event_type": "node_complete", "node": "execute-generated-preprocessing", "status": "pass"},
        {"event_type": "node_complete", "node": "validate-preprocessing-output", "status": "pass"},
        {"event_type": "node_complete", "node": "review-preprocessing-quality", "status": "pass"},
        {"event_type": "node_complete", "node": "generate-feature-engineering-code", "status": "pass"},
    ]
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    trace = parse_trace_artifact(trace_path)
    html = build_pipeline_html(trace)

    first_validate = html.index("validate‑preprocessing‑output")
    repair = html.index("repair‑preprocessing‑code")
    second_validate = html.rindex("validate‑preprocessing‑output")
    generate_fe = html.index("generate‑feature‑engineering‑code")

    assert first_validate < repair < second_validate < generate_fe
    assert "preprocess</div>" not in html


def test_build_pipeline_html_shows_terminal_run_failed_event(tmp_path):
    """A structured run_failed lifecycle event must be visible in the pipeline rail."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_trace_artifact

    trace_path = tmp_path / "trace_events_failed.jsonl"
    events = [
        {"event_type": "run_start", "status": "pass", "node": "__run__"},
        {"event_type": "node_complete", "node": "generate-preprocessing-code", "status": "pass"},
        {"event_type": "node_complete", "node": "validate-preprocessing-output", "status": "fail"},
        {"event_type": "run_failed", "status": "fail", "node": "__run__", "error": "validation never passed"},
    ]
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    trace = parse_trace_artifact(trace_path)
    html = build_pipeline_html(trace)

    assert "run‑failed" in html
    assert "generate‑feature‑engineering‑code" not in html


def test_build_pipeline_html_shows_completed_run_as_finished_not_running(tmp_path):
    """Completed raw stage logs should show lifecycle start/end nodes and no blue spinner."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_stage_log

    log_path = tmp_path / "stage_success.log"
    log_path.write_text(
        "12:00:00 INFO    run_stage  === Stage 'full' (stop_after=END, row_index=42) ===\n"
        "12:00:01 INFO    bt5151_credit_risk.graph  >>> recommend-action\n"
        "12:00:02 INFO    bt5151_credit_risk.graph      action: standard_handling\n"
        "12:00:03 INFO    run_stage  === Stage 'full' completed in 3.0s ===\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    html = build_pipeline_html(trace)

    assert "run‑start" in html
    assert "run‑complete" in html
    assert 'class="pulsing"' not in html


def test_build_pipeline_html_shows_run_start_for_in_progress_raw_log(tmp_path):
    """In-progress raw logs should include a green run-start node plus a blue current node."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_stage_log

    log_path = tmp_path / "stage_running.log"
    log_path.write_text(
        "12:00:00 INFO    run_stage  === Stage 'full' (stop_after=END, row_index=42) ===\n"
        "12:00:01 INFO    bt5151_credit_risk.graph  >>> train-models\n"
        "12:00:02 INFO    bt5151_credit_risk.graph      fitting xgboost\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)
    html = build_pipeline_html(trace)

    assert "run‑start" in html
    assert "train‑models" in html
    assert 'class="pulsing"' in html


def test_live_trace_uses_structured_completion_status_for_last_raw_node(tmp_path):
    """Live raw logs should not show a node as running after JSONL records completion."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_live_trace_artifacts

    log_path = tmp_path / "stage_full_20260417_190732.log"
    log_path.write_text(
        "19:30:00 INFO    run_stage  === Stage 'full' (stop_after=END, row_index=42) ===\n"
        "19:31:02 INFO    bt5151_credit_risk.graph  >>> global-xai\n"
        "19:31:02 INFO    bt5151_credit_risk.graph      SHAP reused from select-model (skipping recomputation)\n"
        "19:39:35 INFO    bt5151_credit_risk.graph      methods used: ['shap', 'pfi_grouped', 'pdp', 'ale']\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace_events_20260417_190732.jsonl"
    events = [
        {"event_type": "run_start", "status": "pass", "node": "__run__"},
        {"event_type": "node_complete", "node": "global-xai", "status": "pass"},
    ]
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    trace = parse_live_trace_artifacts(log_path, trace_path)
    html = build_pipeline_html(trace)

    assert "global‑xai" in html
    assert "local‑xai" in html
    assert 'class="pulsing"' not in html


def test_live_trace_marks_raw_node_after_last_jsonl_completion_as_running(tmp_path):
    """If raw log has advanced past JSONL completion, only the new raw node is running."""
    from bt5151_credit_risk.ui_trace import build_pipeline_html, parse_live_trace_artifacts

    log_path = tmp_path / "stage_full_20260417_190732.log"
    log_path.write_text(
        "19:30:00 INFO    run_stage  === Stage 'full' (stop_after=END, row_index=42) ===\n"
        "19:31:02 INFO    bt5151_credit_risk.graph  >>> global-xai\n"
        "19:39:35 INFO    bt5151_credit_risk.graph      methods used: ['shap', 'pfi_grouped']\n"
        "19:39:35 INFO    bt5151_credit_risk.graph  >>> local-xai (casebook)\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "trace_events_20260417_190732.jsonl"
    events = [
        {"event_type": "run_start", "status": "pass", "node": "__run__"},
        {"event_type": "node_complete", "node": "global-xai", "status": "pass"},
    ]
    trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    trace = parse_live_trace_artifacts(log_path, trace_path)
    html = build_pipeline_html(trace)

    assert "global‑xai" in html
    assert "local‑xai" in html
    assert 'class="pulsing"' in html


def test_list_available_logs_orders_history_by_recency_not_file_type(tmp_path):
    """Recent raw stage logs should not be buried below all JSONL traces."""
    from bt5151_credit_risk.ui_trace import list_available_logs

    older_trace = tmp_path / "trace_events_20260416_010000.jsonl"
    older_trace.write_text("{}\n", encoding="utf-8")
    newer_log = tmp_path / "stage_full_20260416_020000.log"
    newer_log.write_text("12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n", encoding="utf-8")

    os.utime(older_trace, (100, 100))
    os.utime(newer_log, (200, 200))

    names = list_available_logs(tmp_path)

    assert names[0] == newer_log.name
    assert older_trace.name in names
