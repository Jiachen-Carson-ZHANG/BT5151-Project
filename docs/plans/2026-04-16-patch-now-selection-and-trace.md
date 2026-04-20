# Patch-Now Selection And Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Lock the remaining trust boundary around final model selection and add structured trace artifacts for the Gradio Developer Trace tab without importing Plexe-style search/platform complexity.

**Architecture:** Keep the current LangGraph pipeline, cache, and Gradio app. Make `evaluate.py` the sole authority for winner selection, keep the LLM only for narrative justification, and add a machine-readable `trace_events_<run_id>.jsonl` artifact that the UI can consume directly while preserving raw stage logs as fallback evidence.

**Tech Stack:** Python, LangGraph, Gradio, JSONL, pytest, pathlib, stdlib logging/json.

---

## Scope

Patch now:
- Deterministic final model selection
- Structured trace events for live and historical UI trace
- Cache/run provenance fields needed to bind the exact trace artifact to the cached model

Do not patch now:
- Search journal / multi-hypothesis tree
- Insight store
- Retrain mode
- OpenTelemetry
- `run_registry.jsonl`
- New metric or split-policy redesign

### Task 1: Make Final Model Selection Deterministic

**Files:**
- Modify: `src/bt5151_credit_risk/evaluate.py`
- Modify: `src/bt5151_credit_risk/graph.py`
- Test: `tests/test_evaluate.py`
- Test: `tests/test_graph.py`

**Step 1: Write the failing evaluate contract test**

Add to `tests/test_evaluate.py`:

```python
def test_choose_best_model_prefers_macro_then_weighted():
    results = {
        "logistic_regression": {"macro_f1": 0.64, "weighted_f1": 0.65},
        "random_forest": {"macro_f1": 0.68, "weighted_f1": 0.68},
        "xgboost": {"macro_f1": 0.68, "weighted_f1": 0.69},
    }

    best = choose_best_model(results)

    assert best["model_name"] == "xgboost"
    assert "macro_f1" in best["justification"]
```

**Step 2: Write the failing graph contract test**

Add to `tests/test_graph.py`:

```python
def test_select_model_node_never_allows_llm_to_override_metric_best(monkeypatch):
    state = CreditRiskState(
        raw_dataset_path="train.csv",
        evaluation_results={
            "logistic_regression": {"macro_f1": 0.64, "weighted_f1": 0.65},
            "random_forest": {"macro_f1": 0.67, "weighted_f1": 0.68},
            "xgboost": {"macro_f1": 0.69, "weighted_f1": 0.70},
        },
        trained_models={"xgboost": object()},
        class_names=["Good", "Poor", "Standard"],
        test_frame=pd.DataFrame({"a": [1, 2]}),
        feature_columns=["a"],
    )

    monkeypatch.setattr(
        "bt5151_credit_risk.graph.reason_model_selection",
        lambda **kwargs: {"model_name": "random_forest", "justification": "LLM prefers RF"},
    )
    monkeypatch.setattr(
        "bt5151_credit_risk.graph.compute_global_shap",
        lambda *args, **kwargs: {"importance": []},
    )

    result = select_model_node(state)

    assert result["selected_model_name"] == "xgboost"
    assert "LLM prefers RF" in result["selection_justification"]
```

**Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_evaluate.py tests/test_graph.py -q
```

Expected:
- `test_choose_best_model_prefers_macro_then_weighted` may pass already
- `test_select_model_node_never_allows_llm_to_override_metric_best` fails because the current graph still trusts the LLM-returned `model_name`

**Step 4: Write the minimal deterministic implementation**

In `src/bt5151_credit_risk/evaluate.py`:
- Keep `choose_best_model()` as the authority
- Change the LLM helper so it only returns narrative fields, for example:

```python
def justify_model_selection(...):
    return {
        "justification": "...",
        "hypothesis_validation": "...",
    }
```

If you keep the function name `reason_model_selection`, change its contract so any returned `model_name` is ignored by callers.

In `src/bt5151_credit_risk/graph.py`:
- Compute `metric_best = choose_best_model(...)`
- Set `selected_name = metric_best["model_name"]`
- Call the LLM only to explain that winner
- Never recompute SHAP for an LLM-chosen alternate model
- Keep fallback-safe behavior: if the LLM fails, preserve `metric_best["justification"]`

**Step 5: Run targeted tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_evaluate.py tests/test_graph.py -q
```

Expected:
- PASS

**Step 6: Commit**

```bash
git add tests/test_evaluate.py tests/test_graph.py src/bt5151_credit_risk/evaluate.py src/bt5151_credit_risk/graph.py
git commit -m "refactor: make final model selection deterministic"
```

### Task 2: Add Structured Trace Events At Run Time

**Files:**
- Create: `src/bt5151_credit_risk/trace_events.py`
- Modify: `run_stage.py`
- Modify: `src/bt5151_credit_risk/run_status.py`
- Modify: `src/bt5151_credit_risk/state.py`
- Modify: `src/bt5151_credit_risk/cache.py`
- Test: `tests/test_run_status.py`
- Create: `tests/test_trace_events.py`

**Step 1: Write the failing trace event writer test**

Create `tests/test_trace_events.py`:

```python
def test_append_trace_event_writes_one_json_line(tmp_path):
    path = tmp_path / "trace_events.jsonl"

    append_trace_event(
        path=path,
        event={
            "run_id": "20260416_123000",
            "node": "train-models",
            "event_type": "node_complete",
            "status": "pass",
            "state_keys_written": ["trained_models"],
        },
    )

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["node"] == "train-models"
```

**Step 2: Write the failing active-run schema test**

Add to `tests/test_run_status.py`:

```python
def test_write_active_run_persists_trace_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "ACTIVE_RUN_FILE", tmp_path / "active_run.json")

    run_status.write_active_run(
        run_id="20260416_123000",
        stage="full",
        row_index=42,
        log_path="/tmp/stage.log",
        bundle_path="/tmp/bundle.json",
        trace_path="/tmp/trace.jsonl",
    )

    record = run_status.read_active_run()
    assert record["trace_path"] == "/tmp/trace.jsonl"
```

**Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_run_status.py tests/test_trace_events.py -q
```

Expected:
- FAIL because `trace_events.py` does not exist yet
- FAIL because `write_active_run()` has no `trace_path`

**Step 4: Implement the trace event module**

Create `src/bt5151_credit_risk/trace_events.py` with:
- `append_trace_event(path, event)`
- `build_trace_event_path(log_dir, run_id)`
- `summarize_node_update(node_name, update_dict)` returning a compact payload:

```python
{
    "status": "pass" | "warn" | "repair",
    "metrics": {"macro_f1": 0.6943},
    "warnings": ["role violation: ..."],
    "state_keys_written": ["evaluation_results", "selected_model_name"],
}
```

Do not parse prose logs here.

**Step 5: Emit trace events from `run_stage.py`**

Modify `run_stage.py`:
- Add `cache_trace_path` to `_build_provenance_metadata()`
- Build `trace_events_<run_id>.jsonl` in `lab/logs/`
- Pass `trace_path` into `write_active_run()`
- Change full-run execution to stream updates through `_stream_until(...)` with `stop_after=None`, so every node completion produces a structured event
- Append events for:
  - run started
  - node completed
  - run completed
  - run failed

Each node-complete event must include:
- `run_id`
- `node`
- `event_type`
- `timestamp`
- `status`
- `state_keys_written`
- best-effort `metrics`
- best-effort `warnings`

**Step 6: Persist trace provenance**

Modify:
- `src/bt5151_credit_risk/run_status.py` to accept/store `trace_path`
- `src/bt5151_credit_risk/state.py` to add `cache_trace_path: str | None = None`
- `src/bt5151_credit_risk/cache.py` to preserve `cache_trace_path` in `CACHE_KEYS`

**Step 7: Run targeted tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_run_status.py tests/test_trace_events.py -q
```

Expected:
- PASS

**Step 8: Commit**

```bash
git add src/bt5151_credit_risk/trace_events.py src/bt5151_credit_risk/run_status.py src/bt5151_credit_risk/state.py src/bt5151_credit_risk/cache.py run_stage.py tests/test_run_status.py tests/test_trace_events.py
git commit -m "feat: emit structured trace events for pipeline runs"
```

### Task 3: Teach The UI To Prefer Structured Trace Artifacts

**Files:**
- Modify: `src/bt5151_credit_risk/ui_trace.py`
- Modify: `app.py`
- Test: `tests/test_ui_trace.py`
- Test: `tests/test_app.py`

**Step 1: Write the failing trace reader test**

Add to `tests/test_ui_trace.py`:

```python
def test_parse_trace_events_jsonl_builds_cards(tmp_path):
    path = tmp_path / "trace_events.jsonl"
    path.write_text(
        '\n'.join([
            json.dumps({
                "run_id": "20260416_123000",
                "node": "dataset-policy-spec",
                "event_type": "node_complete",
                "status": "pass",
                "state_keys_written": ["dataset_policy_spec"],
                "warnings": [],
                "metrics": {},
            }),
            json.dumps({
                "run_id": "20260416_123000",
                "node": "validate-preprocessing-output",
                "event_type": "node_complete",
                "status": "warn",
                "state_keys_written": ["preprocessing_validation_report"],
                "warnings": ["role violation: Type_of_Loan dropped"],
                "metrics": {},
            }),
        ])
    )

    trace = parse_trace_events(path)

    assert trace["cards"][0]["node_name"] == "dataset-policy-spec"
    assert trace["cards"][1]["status"] == "warn"
```

**Step 2: Write the failing app routing test**

Add to `tests/test_app.py`:

```python
def test_cb_developer_trace_prefers_active_trace_path_over_log(monkeypatch):
    monkeypatch.setattr(
        "bt5151_credit_risk.run_status.read_active_run",
        lambda: {
            "status": "running",
            "pid": os.getpid(),
            "pid_start_time": psutil.Process(os.getpid()).create_time(),
            "trace_path": "/tmp/live_trace.jsonl",
            "log_path": "/tmp/live_stage.log",
            "run_id": "20260416_123000",
        },
    )
    monkeypatch.setattr("bt5151_credit_risk.app._is_process_alive", lambda pid, start: True)
    monkeypatch.setattr("bt5151_credit_risk.app._render_trace_file", lambda path: f"TRACE:{path}")

    assert cb_developer_trace(state=None) == "TRACE:/tmp/live_trace.jsonl"
```

**Step 3: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py tests/test_app.py -q
```

Expected:
- FAIL because `ui_trace.py` only parses raw stage logs today
- FAIL because `cb_developer_trace()` only routes to `log_path`

**Step 4: Implement trace-event parsing and rendering**

In `src/bt5151_credit_risk/ui_trace.py`:
- Add `parse_trace_events(path)`
- Add `build_trace_events_markdown(trace)`
- Keep `parse_stage_log()` and `build_trace_markdown()` as fallback

Prefer a simple card schema:

```python
{
    "node_name": "train-models",
    "status": "pass",
    "summary_lines": [
        "wrote: trained_models, learning_curves",
        "metric: xgboost best_cv_score=0.6943",
    ],
    "warning_lines": [],
    "raw_lines": [...]
}
```

**Step 5: Route the Gradio callback to trace artifacts first**

In `app.py`:
- Add a tiny helper like `_render_trace_file(path)` that chooses:
  - `parse_trace_events()` when suffix is `.jsonl`
  - `parse_stage_log()` otherwise
- Update `cb_developer_trace()` to prefer:
  - `active_run.trace_path` when a live run is active
  - otherwise `state.cache_trace_path`
  - fallback to `log_path` / `cache_log_path` only when trace artifact is missing

Keep the current size-gated memoization, but cache by `(path, size)`.

**Step 6: Run targeted tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py tests/test_app.py -q
```

Expected:
- PASS

**Step 7: Commit**

```bash
git add src/bt5151_credit_risk/ui_trace.py app.py tests/test_ui_trace.py tests/test_app.py
git commit -m "feat: render developer trace from structured events"
```

### Task 4: Update Architecture Docs And Run Full Verification

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update the architecture document**

In `docs/architecture/current-state.md`:
- Change model selection wording from “LLM-driven” to “metric-deterministic with LLM justification”
- Add `trace_events_<run_id>.jsonl` as a first-class run artifact
- Note that Tab 3 prefers structured trace artifacts and falls back to raw logs

**Step 2: Append implementation log entry**

In `docs/changes/implementation-log.md`:
- Add one concise entry covering:
  - deterministic model selection
  - trace event artifact
  - cache/run provenance updates
  - UI trace routing change

**Step 3: Run the full relevant suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_evaluate.py tests/test_graph.py tests/test_run_status.py tests/test_trace_events.py tests/test_ui_trace.py tests/test_app.py tests/test_state.py tests/test_cache.py -q
```

Expected:
- PASS

If green, run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests -q
```

Expected:
- full suite PASS

**Step 4: Commit**

```bash
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: record deterministic selection and structured trace artifacts"
```

## Acceptance Criteria

- Final model selection is always the metric-best model from `choose_best_model()`
- The LLM can explain the winner, but cannot change the winner
- A full run writes `lab/logs/trace_events_<run_id>.jsonl`
- `active_run.json` includes `trace_path`
- Saved cache includes `cache_trace_path`
- Developer Trace prefers structured trace artifacts for both live and historical runs
- Raw stage logs remain available as fallback evidence
- No Plexe-style journal, search tree, or retrain mode is introduced

## Why This Is The Right Patch Now

- It closes the only remaining high-value LLM trust boundary in the current evaluation path
- It materially improves the UI without forcing a large platform refactor
- It borrows the best part of Plexe now, which is structured observability, not search complexity
- It preserves the current architecture instead of widening scope before the next evaluation run
