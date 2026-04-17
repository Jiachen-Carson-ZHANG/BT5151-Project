"""Tests for Gradio app callback behavior.

Covers: cold-start, train button, concurrent run guard, cache reload.
Does NOT test Gradio rendering — only callback logic.
"""

import os
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Helper to build a minimal mock state for tests that need one
# ---------------------------------------------------------------------------
def _mock_state():
    """Return a MagicMock with enough state attributes to avoid AttributeErrors."""
    state = MagicMock()
    state.raw_frame = _make_raw_frame()
    state.evaluation_results = {
        "xgboost": {
            "macro_f1": 0.80,
            "weighted_f1": 0.81,
            "accuracy": 0.82,
            "confusion_matrix": [[100, 10, 5], [8, 90, 12], [6, 9, 95]],
            "per_class": {
                "Good": {"precision": 0.83, "recall": 0.82, "f1-score": 0.83, "support": 115},
                "Standard": {"precision": 0.82, "recall": 0.83, "f1-score": 0.82, "support": 110},
                "Poor": {"precision": 0.88, "recall": 0.85, "f1-score": 0.86, "support": 110},
            },
        }
    }
    state.selected_model_name = "xgboost"
    state.selection_justification = "Best macro_f1."
    state.global_shap_importance = [
        {"feature": "Annual_Income", "mean_abs_shap": 0.25},
        {"feature": "Outstanding_Debt", "mean_abs_shap": 0.18},
    ]
    state.global_xai_results = {"methods_used": ["shap", "pfi"]}
    state.training_diagnostics = {
        "hypothesis_validation": [
            {"prediction": "LR macro_f1 >= 0.55", "actual": "0.58", "verdict": "confirmed"},
        ]
    }
    state.class_names = ["Good", "Poor", "Standard"]
    state.eda_hypotheses = None
    state.full_feature_frame = _make_raw_frame()
    state.full_feature_frames_by_view = None
    state.cache_trace_path = "/tmp/trace_events_20260416_120000.jsonl"
    state.cache_log_path = "/tmp/stage_full_20260416_120000.log"
    state.cache_bundle_path = "/tmp/analysis_bundle_20260416_120000.json"
    state.run_id = "20260416_120000"
    return state


def _make_raw_frame():
    import pandas as pd

    return pd.DataFrame(
        {"Age": [35, 42], "Annual_Income": [50000.0, 80000.0], "Credit_Score": ["Good", "Poor"]},
        index=[0, 1],
    )


# ---------------------------------------------------------------------------
# Task 5: Cold-start path
# ---------------------------------------------------------------------------
class TestColdStart:
    def test_build_app_constructs_under_current_gradio(self):
        """build_app() should construct without Gradio constructor errors."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=_mock_state()):
            demo = app_mod.build_app()

        assert demo is not None

    def test_build_app_exposes_sticky_trace_and_toolbar_ids(self):
        """Layout hooks for the evidence toolbar and sticky trace sidebar must exist."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=_mock_state()):
            demo = app_mod.build_app()
            cfg = demo.get_config_file()

        elem_ids = {
            c.get("props", {}).get("elem_id")
            for c in cfg["components"]
            if c.get("props", {}).get("elem_id")
        }
        assert {
            "evidence-toolbar",
            "evidence-toolbar-copy",
            "evidence-toolbar-actions",
            "evidence-refresh-btn",
            "trace-layout",
            "trace-left-col",
            "trace-right-col",
            "trace-pipeline",
            "trace-log",
        }.issubset(elem_ids)
        assert "evidence-advanced-btn" not in elem_ids

    def test_trace_sidebar_css_keeps_pipeline_sticky_without_inner_log_scroll(self):
        """Developer Trace should use page scroll, with the left pipeline column sticky."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        css = app_mod._app_css()

        gradio_container_block = css.split(".gradio-container {", 1)[1].split("}", 1)[0]
        assert "overflow: visible" in gradio_container_block

        assert "#trace-left-col" in css
        trace_left_block = css.split("#trace-left-col", 1)[1].split("}", 1)[0]
        assert "position: sticky" in trace_left_block
        assert "top: 88px" in trace_left_block
        assert "flex-direction: column" in trace_left_block
        assert "max-height: calc(100dvh - 112px)" in trace_left_block
        assert "overflow-y: auto" in trace_left_block

        trace_children_block = css.split("#trace-left-col > *", 1)[1].split("}", 1)[0]
        assert "width: 100%" in trace_children_block
        assert "min-width: 0" in trace_children_block

        trace_log_block = css.split("#trace-log", 1)[1].split("}", 1)[0]
        assert "overflow: visible" in trace_log_block
        assert "max-height" not in trace_log_block
        assert "overflow-y: auto" not in trace_log_block

    def test_pipeline_html_wraps_title_and_body_in_one_component(self):
        """Pipeline title and body must stay in one HTML component to avoid clipping."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        html = app_mod._wrap_pipeline_html("<div class='step'>train-models</div>")

        assert "trace-pipeline-shell" in html
        assert "trace-pipeline-title" in html
        assert "Pipeline" in html
        assert "train-models" in html

    def test_cb_predict_returns_no_cache_message_when_state_is_none(self):
        """cb_predict must not crash when no cache is loaded."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = list(app_mod.cb_predict(0))[-1]

        # First element is the risk_html / status message
        first = result[0] if isinstance(result, (list, tuple)) else result
        text = str(first).lower()
        assert any(kw in text for kw in ("no", "cache", "train")), f"Expected no-cache message, got: {first!r}"

    def test_cb_load_customer_returns_status_when_state_is_none(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = app_mod.cb_load_customer(0)

        # Should return a DataFrame with a Status column, not raise
        import pandas as pd

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Task 5: Train button — background subprocess, not blocking
# ---------------------------------------------------------------------------
class TestTrainButton:
    def test_cb_train_pipeline_spawns_subprocess_when_no_active_run(self, tmp_path, monkeypatch):
        """Train button should launch run_stage.py in background when no active run."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        spawned = {}

        def fake_popen(args, **kwargs):
            spawned["args"] = args
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "bt5151_credit_risk.run_status.read_active_run",
            lambda: None,
        )

        result = app_mod.cb_train_pipeline(row_index=42)

        assert "spawned" in spawned or spawned.get("args") is not None, "Popen was not called"
        assert any("run_stage" in str(a) for a in spawned["args"]), "run_stage.py not in subprocess args"
        assert "42" in str(spawned["args"]) or 42 in spawned["args"], "row_index not passed to subprocess"

    def test_cb_train_pipeline_rejects_concurrent_run(self, monkeypatch):
        """Train button must not spawn a second process when a run is already in progress."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        active_run = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
        }

        spawned = {}

        def fake_popen(args, **kwargs):
            spawned["called"] = True
            return MagicMock()

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr(
            "bt5151_credit_risk.run_status.read_active_run",
            lambda: active_run,
        )

        result = app_mod.cb_train_pipeline(row_index=42)

        assert "called" not in spawned, "Popen must NOT be called when a run is already live"
        text = str(result).lower()
        assert any(kw in text for kw in ("in progress", "already", "running")), (
            f"Expected 'already in progress' message, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# Task 6: Business View — cb_predict structure
# ---------------------------------------------------------------------------
class TestBusinessView:
    def test_cb_predict_returns_structured_output(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        mock_pred = {
            "predicted_label": "Good",
            "probabilities": {"Good": 0.75, "Standard": 0.20, "Poor": 0.05},
            "confidence": 0.75,
            "confidence_diagnosis": {"caution_level": "low", "reason": "High confidence."},
            "shap_waterfall": {
                "predicted_class_waterfall": {
                    "class": "Good",
                    "top_features": [{"feature": "Income", "shap_value": 0.1, "direction": "positive"}],
                }
            },
        }
        mock_risk = {"summary": "Low risk customer.", "key_drivers": [], "risk_level": "Good"}
        mock_action = {"action": "Approve", "rationale": "Strong financials."}

        with patch.object(app_mod, "_get_state", return_value=_mock_state()):
            with patch.object(app_mod, "_run_inference_step", return_value=(mock_pred, _mock_state())):
                with patch.object(app_mod, "_run_explain_step", return_value=(mock_risk, mock_action)):
                    result = list(app_mod.cb_predict(0))[-1]

        assert isinstance(result, (list, tuple))
        risk_html = result[0]
        assert "Good" in str(risk_html)


# ---------------------------------------------------------------------------
# Task 7: Model Evidence — cb_model_overview structure
# ---------------------------------------------------------------------------
class TestModelEvidence:
    def test_cb_model_overview_returns_metrics_df_and_confusion_matrix(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=_mock_state()):
            result = app_mod.cb_model_overview()

        assert isinstance(result, (list, tuple))
        assert len(result) == 5
        # metrics DataFrame should be first element
        import pandas as pd

        metrics_df = result[0]
        assert isinstance(metrics_df, pd.DataFrame)
        assert len(metrics_df) >= 1

    def test_cb_model_overview_includes_confusion_matrix(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=_mock_state()):
            result = app_mod.cb_model_overview()

        # Confusion matrix figure (plt.Figure or None) should be in the result
        # It can be None if matplotlib not available, but not an exception
        assert result is not None

    def test_cb_model_overview_charts_returns_file_backed_chart_images(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
        state.global_xai_results = {
            "methods_used": ["shap", "pdp", "ale"],
            "shap": {
                "beeswarm_data": {
                    "Age": {
                        "shap_values": [0.1, -0.2, 0.05],
                        "feature_values": [25, 40, 55],
                    }
                },
                "dependence_data": {
                    "Age": {
                        "Good": {"feature_values": [25, 40], "shap_values": [0.1, 0.2]},
                        "Poor": {"feature_values": [30, 50], "shap_values": [-0.2, -0.1]},
                        "Standard": {"feature_values": [35, 45], "shap_values": [0.0, 0.05]},
                    }
                },
            },
            "pdp": {
                "Age": {
                    "grid": [20, 40, 60],
                    "pd_values": {
                        "Good": [0.6, 0.5, 0.4],
                        "Poor": [0.2, 0.3, 0.4],
                        "Standard": [0.2, 0.2, 0.2],
                    },
                },
                "Annual_Income": {
                    "grid": [30000, 60000, 90000],
                    "pd_values": {
                        "Good": [0.3, 0.5, 0.7],
                        "Poor": [0.5, 0.3, 0.2],
                        "Standard": [0.2, 0.2, 0.1],
                    },
                },
            },
            "ale": {
                "Age": {
                    "bin_centres": [25, 45, 55],
                    "ale_values": {
                        "Good": [0.05, 0.02, -0.01],
                        "Poor": [-0.03, 0.0, 0.04],
                        "Standard": [0.01, 0.0, -0.01],
                    },
                },
                "Annual_Income": {
                    "bin_centres": [35000, 65000, 85000],
                    "ale_values": {
                        "Good": [0.01, 0.03, 0.04],
                        "Poor": [-0.02, -0.01, 0.0],
                        "Standard": [0.0, 0.0, -0.01],
                    },
                },
            },
        }

        with patch.object(app_mod, "_get_state", return_value=state):
            result = app_mod.cb_model_overview_charts()

        assert isinstance(result, (list, tuple))
        assert len(result) == 4
        for item in result:
            assert isinstance(item, str)
            assert item

    def test_model_evidence_heavy_charts_can_load_individually(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
        state.global_xai_results = {
            "methods_used": ["shap", "pdp", "ale"],
            "shap": {
                "beeswarm_data": {
                    "Age": {
                        "shap_values": [0.1, -0.2, 0.05],
                        "feature_values": [25, 40, 55],
                    }
                },
                "dependence_data": {
                    "Age": {
                        "Good": {"feature_values": [25, 40], "shap_values": [0.1, 0.2]},
                        "Poor": {"feature_values": [30, 50], "shap_values": [-0.2, -0.1]},
                        "Standard": {"feature_values": [35, 45], "shap_values": [0.0, 0.05]},
                    }
                },
            },
            "pdp": {
                "Age": {
                    "grid": [20, 40, 60],
                    "pd_values": {
                        "Good": [0.6, 0.5, 0.4],
                        "Poor": [0.2, 0.3, 0.4],
                        "Standard": [0.2, 0.2, 0.2],
                    },
                },
                "Annual_Income": {
                    "grid": [30000, 60000, 90000],
                    "pd_values": {
                        "Good": [0.3, 0.5, 0.7],
                        "Poor": [0.5, 0.3, 0.2],
                        "Standard": [0.2, 0.2, 0.1],
                    },
                },
            },
            "ale": {
                "Age": {
                    "bin_centres": [25, 45, 55],
                    "ale_values": {
                        "Good": [0.05, 0.02, -0.01],
                        "Poor": [-0.03, 0.0, 0.04],
                        "Standard": [0.01, 0.0, -0.01],
                    },
                },
                "Annual_Income": {
                    "bin_centres": [35000, 65000, 85000],
                    "ale_values": {
                        "Good": [0.01, 0.03, 0.04],
                        "Poor": [-0.02, -0.01, 0.0],
                        "Standard": [0.0, 0.0, -0.01],
                    },
                },
            },
        }

        with patch.object(app_mod, "_get_state", return_value=state):
            shap = app_mod.cb_model_shap_chart()
            pdp_ale = app_mod.cb_model_pdp_ale_charts()
            dependence = app_mod.cb_model_dependence_chart()

        assert isinstance(shap, str)
        assert isinstance(pdp_ale, tuple)
        assert len(pdp_ale) == 2
        assert all(isinstance(item, str) for item in pdp_ale)
        assert isinstance(dependence, str)

    def test_cb_model_overview_handles_nested_hypothesis_validation_dict(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
        state.training_diagnostics = {
            "hypothesis_validation": {
                "tested": [
                    {
                        "original": "XGBoost will win",
                        "outcome": "confirmed",
                        "actual": "macro_f1=0.69",
                    }
                ],
                "supported": [
                    {
                        "original": "Standard recall will lag",
                        "status": "weak_evidence",
                        "evidence": "recall=0.61",
                    }
                ],
            }
        }

        with patch.object(app_mod, "_get_state", return_value=state):
            result = app_mod.cb_model_overview()

        hyp_md = result[4]
        assert "tier" in hyp_md
        assert "tested" in hyp_md
        assert "supported" in hyp_md
        assert "XGBoost will win" in hyp_md
        assert "Standard recall will lag" in hyp_md

    def test_cb_model_overview_returns_no_cache_df_when_state_none(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = app_mod.cb_model_overview()

        assert result is not None

    def test_cb_model_overview_charts_returns_empty_values_when_state_none(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = app_mod.cb_model_overview_charts()

        assert result == (None, None, None, None)


# ---------------------------------------------------------------------------
# Task 8: Developer Trace — cb_developer_trace callback behavior
# ---------------------------------------------------------------------------
class TestDeveloperTrace:
    def test_cb_poll_trace_honors_pinned_historical_log_selection(self, tmp_path, monkeypatch):
        """Timer polling must not overwrite a manually selected historical run."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        historical_log = tmp_path / "stage_full_20260415_131136.log"
        historical_log.write_text(
            "13:38:03 INFO    bt5151_credit_risk.graph  >>> evaluate-models\n",
            encoding="utf-8",
        )
        live_trace = tmp_path / "trace_events_20260417_124505.jsonl"
        live_trace.write_text(
            json.dumps(
                {
                    "run_id": "20260417_124505",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "validate-feature-engineering",
                    "status": "fail",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260417_124505",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "trace_path": str(live_trace),
        }

        monkeypatch.setattr(app_mod, "_historical_log_dir", lambda: tmp_path)
        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

        with patch.object(app_mod, "_get_state", return_value=None):
            pipeline_html, trace_md, _ = app_mod.cb_poll_trace(historical_log.name)

        assert "evaluate-models" in trace_md
        assert "validate-feature-engineering" not in trace_md
        assert "evaluate" in pipeline_html.lower()

    def test_cb_load_historical_log_empty_selection_returns_auto_trace(self, tmp_path, monkeypatch):
        """Clearing the historical selection should fall back to live/cached auto mode."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        cached_trace = tmp_path / "trace_events_cached.jsonl"
        cached_trace.write_text(
            json.dumps(
                {
                    "run_id": "20260416_012744",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "train-models",
                    "status": "pass",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        state = _mock_state()
        state.cache_trace_path = str(cached_trace)
        state.cache_log_path = None

        monkeypatch.setattr(app_mod, "_historical_log_dir", lambda: tmp_path)
        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: None)

        with patch.object(app_mod, "_get_state", return_value=state):
            pipeline_html, trace_md, selected = app_mod.cb_load_historical_log("")

        assert selected is None
        assert "train-models" in trace_md
        assert "train" in pipeline_html.lower()

    def test_cb_developer_trace_prefers_active_run_log_when_alive(self, tmp_path, monkeypatch):
        """Live runs should use the raw log so long-running nodes appear before trace completion."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        active_trace = tmp_path / "trace_events_20260416_120000.jsonl"
        active_trace.write_text(
            "\n".join(
                [
                    json.dumps({"run_id": "20260416_120000", "stage": "full", "event_type": "run_start"}),
                    json.dumps(
                        {
                            "run_id": "20260416_120000",
                            "stage": "full",
                            "event_type": "node_complete",
                            "node": "validate-feature-engineering",
                            "status": "pass",
                            "state_keys_written": ["feature_engineering_validation_report"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raw_log = tmp_path / "stage_full_20260416_120000.log"
        raw_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "log_path": str(raw_log),
            "trace_path": str(active_trace),
            "bundle_path": "/tmp/bundle.json",
        }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

        result = app_mod.cb_developer_trace(state=None)

        assert "20260416_120000" in result
        assert "train-models" in result
        assert "validate-feature-engineering" not in result

    def test_cb_developer_trace_uses_active_run_trace_when_failed(self, tmp_path, monkeypatch):
        """Failed active runs should still render their own trace artifact."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        failed_trace = tmp_path / "trace_events_failed.jsonl"
        failed_trace.write_text(
            "\n".join(
                [
                    json.dumps({"run_id": "20260416_120000", "stage": "full", "event_type": "run_failed", "status": "fail"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        cached_trace = tmp_path / "trace_events_cached.jsonl"
        cached_trace.write_text(
            json.dumps(
                {
                    "run_id": "20260416_999999",
                    "stage": "full",
                    "event_type": "cache_saved",
                    "status": "pass",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "failed",
            "pid": os.getpid(),
            "pid_start_time": 0.0,
            "log_path": str(tmp_path / "stage_full_failed.log"),
            "trace_path": str(failed_trace),
            "bundle_path": "/tmp/bundle.json",
        }

        state = _mock_state()
        state.cache_trace_path = str(cached_trace)
        state.cache_log_path = str(tmp_path / "stage_cached.log")

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

        result = app_mod.cb_developer_trace(state=state)

        assert "run_failed" in result
        assert "20260416_999999" not in result

    def test_cb_developer_trace_falls_back_to_active_run_log_when_trace_missing(self, tmp_path, monkeypatch):
        """If the trace artifact is missing, cb_developer_trace should fall back to the raw log."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        raw_log = tmp_path / "stage_full_20260416_120000.log"
        raw_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "trace_path": str(tmp_path / "missing_trace.jsonl"),
            "log_path": str(raw_log),
            "bundle_path": "/tmp/bundle.json",
        }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

        result = app_mod.cb_developer_trace(state=None)

        assert "train-models" in result
        assert "missing_trace" not in result

    def test_cb_developer_trace_falls_back_to_cache_trace_when_no_active_run(self, tmp_path, monkeypatch):
        """cb_developer_trace must use state.cache_trace_path when no active run."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        cache_trace = tmp_path / "trace_events_cached.jsonl"
        cache_trace.write_text(
            "\n".join(
                [
                    json.dumps({"run_id": "20260416_120000", "stage": "full", "event_type": "run_start"}),
                    json.dumps(
                        {
                            "run_id": "20260416_120000",
                            "stage": "full",
                            "event_type": "node_complete",
                            "node": "evaluate-models",
                            "status": "warn",
                            "state_keys_written": ["evaluation_results"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raw_log = tmp_path / "stage_full_cached.log"
        raw_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n",
            encoding="utf-8",
        )

        state = _mock_state()
        state.cache_trace_path = str(cache_trace)
        state.cache_log_path = str(raw_log)

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: None)

        result = app_mod.cb_developer_trace(state=state)

        assert "evaluate-models" in result
        assert "train-models" not in result

    def test_cb_developer_trace_falls_back_to_raw_log_when_no_trace_artifact(self, tmp_path, monkeypatch):
        """cb_developer_trace must use raw stage logs only as the final fallback."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
        state.cache_trace_path = None
        state.cache_log_path = None
        raw_log = tmp_path / "stage.log"
        raw_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> evaluate-models\n",
            encoding="utf-8",
        )
        state.cache_log_path = str(raw_log)

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: None)

        result = app_mod.cb_developer_trace(state=state)

        assert "evaluate-models" in result

    def test_cb_developer_trace_friendly_message_when_log_missing(self, monkeypatch):
        """cb_developer_trace must not crash when no trace artifact exists."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
        state.cache_trace_path = None
        state.cache_log_path = "/nonexistent/path/stage.log"

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: None)

        result = app_mod.cb_developer_trace(state=state)

        assert result is not None
        assert isinstance(result, str)
        # Should be a friendly message, not a crash
        text = result.lower()
        assert any(kw in text for kw in ("not found", "unavailable", "no log", "missing", "⚠")), (
            f"Expected friendly not-found message, got: {result!r}"
        )

    def test_cb_developer_trace_reuses_cached_parse_when_file_unchanged(self, tmp_path, monkeypatch):
        """Repeated polls on an unchanged artifact should reuse the cached markdown."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        trace_path = tmp_path / "trace_events_20260416_120000.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "train-models",
                    "status": "pass",
                    "state_keys_written": ["trained_models"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "trace_path": str(trace_path),
        }

        call_count = {"parse": 0}

        def fake_parse(path):
            call_count["parse"] += 1
            return {
                "run_summary": {
                    "log_path": str(path),
                    "artifact_type": "structured_trace",
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "total_events": 1,
                },
                "cards": [
                    {
                        "node_name": "train-models",
                        "status": "pass",
                        "summary_lines": ["cached trace"],
                        "warning_lines": [],
                        "raw_lines": [],
                    }
                ],
            }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)
        monkeypatch.setattr("bt5151_credit_risk.ui_trace.parse_trace_artifact", fake_parse)

        first = app_mod.cb_developer_trace(state=None)
        second = app_mod.cb_developer_trace(state=None)

        assert call_count["parse"] == 1
        assert first == second

    def test_cb_developer_trace_reparses_when_trace_grows(self, tmp_path, monkeypatch):
        """Appending to the trace artifact should invalidate the cached markdown."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        trace_path = tmp_path / "trace_events_20260416_120000.jsonl"
        trace_path.write_text(
            json.dumps(
                {
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "train-models",
                    "status": "pass",
                    "state_keys_written": ["trained_models"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "trace_path": str(trace_path),
        }

        call_count = {"parse": 0}

        def fake_parse(path):
            call_count["parse"] += 1
            return {
                "run_summary": {
                    "log_path": str(path),
                    "artifact_type": "structured_trace",
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "total_events": call_count["parse"],
                },
                "cards": [
                    {
                        "node_name": "train-models",
                        "status": "pass",
                        "summary_lines": [f"parse #{call_count['parse']}"],
                        "warning_lines": [],
                        "raw_lines": [],
                    }
                ],
            }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)
        monkeypatch.setattr("bt5151_credit_risk.ui_trace.parse_trace_artifact", fake_parse)

        first = app_mod.cb_developer_trace(state=None)
        trace_path.write_text(
            trace_path.read_text(encoding="utf-8")
            + json.dumps(
                {
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "event_type": "cache_saved",
                    "status": "pass",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        second = app_mod.cb_developer_trace(state=None)

        assert call_count["parse"] == 2
        assert first != second

    def test_cb_developer_trace_reparses_when_trace_path_changes(self, tmp_path, monkeypatch):
        """Switching trace artifact paths should bypass the memoized markdown."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        trace_path_1 = tmp_path / "trace_events_1.jsonl"
        trace_path_1.write_text(
            json.dumps(
                {
                    "run_id": "20260416_120000",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "train-models",
                    "status": "pass",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        trace_path_2 = tmp_path / "trace_events_2.jsonl"
        trace_path_2.write_text(
            json.dumps(
                {
                    "run_id": "20260416_120001",
                    "stage": "full",
                    "event_type": "node_complete",
                    "node": "evaluate-models",
                    "status": "warn",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "trace_path": str(trace_path_1),
        }

        call_count = {"parse": 0}

        def fake_parse(path):
            call_count["parse"] += 1
            return {
                "run_summary": {
                    "log_path": str(path),
                    "artifact_type": "structured_trace",
                    "run_id": "20260416_120000" if path == trace_path_1 else "20260416_120001",
                    "stage": "full",
                    "total_events": 1,
                },
                "cards": [
                    {
                        "node_name": "train-models" if path == trace_path_1 else "evaluate-models",
                        "status": "pass" if path == trace_path_1 else "warn",
                        "summary_lines": ["cached trace"],
                        "warning_lines": [],
                        "raw_lines": [],
                    }
                ],
            }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)
        monkeypatch.setattr("bt5151_credit_risk.ui_trace.parse_trace_artifact", fake_parse)

        first = app_mod.cb_developer_trace(state=None)
        active["trace_path"] = str(trace_path_2)
        second = app_mod.cb_developer_trace(state=None)

        assert call_count["parse"] == 2
        assert "train-models" in first
        assert "evaluate-models" in second


# ---------------------------------------------------------------------------
# Task 9: Cache reload after live training completes
# ---------------------------------------------------------------------------
class TestCacheReload:
    def test_invalidate_cache_clears_state(self):
        """_invalidate_cache() must set _state to None."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        app_mod._state = _mock_state()
        assert app_mod._state is not None

        app_mod._invalidate_cache()
        assert app_mod._state is None

    def test_get_state_reloads_after_invalidation(self, monkeypatch):
        """After _invalidate_cache(), _get_state() must call load_cache again."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        app_mod._state = None
        new_state = _mock_state()
        monkeypatch.setattr("bt5151_credit_risk.cache.load_cache", lambda path=None: new_state)

        result = app_mod._get_state()
        assert result is new_state
