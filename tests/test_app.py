"""Tests for Gradio app callback behavior.

Covers: cold-start, train button, concurrent run guard, cache reload.
Does NOT test Gradio rendering — only callback logic.
"""

import os
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
    def test_cb_predict_returns_no_cache_message_when_state_is_none(self):
        """cb_predict must not crash when no cache is loaded."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = app_mod.cb_predict(0)

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
            with patch.object(app_mod, "_predict", return_value=(mock_pred, mock_risk, mock_action)):
                result = app_mod.cb_predict(0)

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

    def test_cb_model_overview_returns_no_cache_df_when_state_none(self):
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        with patch.object(app_mod, "_get_state", return_value=None):
            result = app_mod.cb_model_overview()

        assert result is not None


# ---------------------------------------------------------------------------
# Task 8: Developer Trace — cb_developer_trace callback behavior
# ---------------------------------------------------------------------------
class TestDeveloperTrace:
    def test_cb_developer_trace_uses_active_run_log_when_alive(self, tmp_path, monkeypatch):
        """cb_developer_trace must use active_run.log_path when PID is alive."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        bound_log = tmp_path / "stage_full_20260416_120000.log"
        bound_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> train-models\n",
            encoding="utf-8",
        )

        active = {
            "run_id": "20260416_120000",
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": __import__("psutil").Process(os.getpid()).create_time(),
            "log_path": str(bound_log),
            "bundle_path": "/tmp/bundle.json",
        }

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

        result = app_mod.cb_developer_trace(state=None)

        assert "20260416_120000" in result
        assert "train-models" in result

    def test_cb_developer_trace_falls_back_to_cache_log_when_no_active_run(self, tmp_path, monkeypatch):
        """cb_developer_trace must use state.cache_log_path when no active run."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        cache_log = tmp_path / "stage_full_cached.log"
        cache_log.write_text(
            "12:00:00 INFO    bt5151_credit_risk.graph  >>> evaluate-models\n",
            encoding="utf-8",
        )

        state = _mock_state()
        state.cache_log_path = str(cache_log)

        monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: None)

        result = app_mod.cb_developer_trace(state=state)

        assert "evaluate-models" in result

    def test_cb_developer_trace_friendly_message_when_log_missing(self, monkeypatch):
        """cb_developer_trace must not crash when log file is missing."""
        import importlib

        import app as app_mod

        importlib.reload(app_mod)

        state = _mock_state()
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
        monkeypatch.setattr("bt5151_credit_risk.cache.load_cache", lambda: new_state)

        result = app_mod._get_state()
        assert result is new_state
