import json
from types import SimpleNamespace

import pytest

from bt5151_credit_risk import llm


class _FakeResponses:
    def __init__(self, recorder: dict):
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"ok": True}),
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class _FakeClient:
    def __init__(self, recorder: dict):
        self.responses = _FakeResponses(recorder)


@pytest.fixture
def capture_openai_call(monkeypatch):
    recorder: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "OpenAI", lambda api_key: _FakeClient(recorder))
    llm.reset_usage_log()
    return recorder


def test_reasoning_effort_applied_for_caller_with_o_model(monkeypatch, capture_openai_call):
    monkeypatch.setenv("OPENAI_MODEL_COLUMN_TRANSFORM_SPEC", "o3")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_COLUMN_TRANSFORM_SPEC", "high")

    llm.call_json_response("sys", {"x": 1}, caller="column-transform-spec")

    assert capture_openai_call["model"] == "o3"
    assert capture_openai_call["reasoning"] == {"effort": "high"}


def test_reasoning_effort_falls_back_to_global(monkeypatch, capture_openai_call):
    monkeypatch.setenv("OPENAI_MODEL_COLUMN_TRANSFORM_SPEC", "o4-mini")
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_COLUMN_TRANSFORM_SPEC", raising=False)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")

    llm.call_json_response("sys", {"x": 1}, caller="column-transform-spec")

    assert capture_openai_call["reasoning"] == {"effort": "medium"}


def test_reasoning_effort_skipped_for_non_o_model(monkeypatch, capture_openai_call):
    monkeypatch.setenv("OPENAI_MODEL_EXPLAIN_RISK", "gpt-4o")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_EXPLAIN_RISK", "high")

    llm.call_json_response("sys", {"x": 1}, caller="explain-risk")

    assert "reasoning" not in capture_openai_call


def test_reasoning_effort_absent_when_not_configured(monkeypatch, capture_openai_call):
    monkeypatch.setenv("OPENAI_MODEL_COLUMN_TRANSFORM_SPEC", "o4-mini")
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_COLUMN_TRANSFORM_SPEC", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)

    llm.call_json_response("sys", {"x": 1}, caller="column-transform-spec")

    assert "reasoning" not in capture_openai_call
