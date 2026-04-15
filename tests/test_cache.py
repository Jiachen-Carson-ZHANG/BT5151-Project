"""Tests for cache provenance contract."""

import pytest


def test_save_cache_persists_provenance_metadata(tmp_path, monkeypatch):
    import bt5151_credit_risk.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "pipeline_state.pkl")

    from bt5151_credit_risk.cache import load_cache, save_cache

    result = {
        "selected_model_name": "xgboost",
        "class_names": ["Good", "Poor", "Standard"],
        "run_id": "20260416_120000",
    }
    metadata = {
        "cache_log_path": "/tmp/stage_full_20260416_120000.log",
        "cache_bundle_path": "/tmp/analysis_bundle_20260416_120000.json",
        "cache_saved_at": "2026-04-16T12:00:00Z",
    }

    save_cache(result, metadata=metadata)
    state = load_cache()

    assert state.run_id == "20260416_120000"
    assert state.cache_log_path == "/tmp/stage_full_20260416_120000.log"
    assert state.cache_bundle_path == "/tmp/analysis_bundle_20260416_120000.json"
    assert state.cache_saved_at == "2026-04-16T12:00:00Z"


def test_save_cache_without_metadata_does_not_crash(tmp_path, monkeypatch):
    import bt5151_credit_risk.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "pipeline_state.pkl")

    from bt5151_credit_risk.cache import load_cache, save_cache

    result = {
        "selected_model_name": "xgboost",
        "class_names": ["Good", "Poor", "Standard"],
        "run_id": "20260416_130000",
    }

    save_cache(result)
    state = load_cache()

    assert state.run_id == "20260416_130000"
    assert state.cache_log_path is None
    assert state.cache_bundle_path is None
    assert state.cache_saved_at is None


def test_load_cache_returns_none_when_file_missing(tmp_path, monkeypatch):
    import bt5151_credit_risk.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "nonexistent.pkl")

    from bt5151_credit_risk.cache import load_cache

    state = load_cache()
    assert state is None


def test_cache_log_path_accessible_from_state(tmp_path, monkeypatch):
    """Loaded CreditRiskState exposes provenance fields."""
    import bt5151_credit_risk.cache as cache_mod

    monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "pipeline_state.pkl")

    from bt5151_credit_risk.cache import load_cache, save_cache

    save_cache(
        {"run_id": "20260416_140000"},
        metadata={
            "cache_log_path": "/lab/logs/stage_full_20260416_140000.log",
            "cache_bundle_path": "/lab/logs/analysis_bundle_20260416_140000.json",
            "cache_saved_at": "2026-04-16T14:00:00Z",
        },
    )
    state = load_cache()

    assert state.cache_log_path.endswith(".log")
    assert state.cache_bundle_path.endswith(".json")
