# Gradio Three-Tab UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current demo UI with three clear tabs: `Business View`, `Model Evidence`, and `Developer Trace`, where Tab 3 can both show the raw-log provenance of the exact cached model and watch a new training run in real time while it is still running.

**Architecture:** Keep one source of truth for completed model/evidence data: the saved pipeline cache. Tabs 1 and 2 always read from the last completed cache. Add a small `active_run.json` status artifact for in-progress training jobs; Tab 3 polls that artifact and tails the bound raw log in real time. Do not parse the “latest log”; both historical and live trace views must be bound to an explicit `run_id` and `log_path`.

**Tech Stack:** Gradio, joblib cache, matplotlib, pandas, stdlib `pathlib`/`re`/`json`/`subprocess`, existing `run_stage.py` logging.

---

## Target Architecture

```text
App header
  |
  +--> global row selector
  +--> "Train / Refresh Pipeline" button
         |
         +--> background subprocess:
              run_stage.py full <row_index> --save-cache
                     |
                     +--> lab/logs/stage_full_<run_id>.log
                     +--> lab/logs/analysis_bundle_<run_id>.json
                     +--> lab/cache/active_run.json
                     |      {
                     |        run_id,
                     |        status,
                     |        row_index,
                     |        log_path,
                     |        bundle_path,
                     |        started_at,
                     |        completed_at,
                     |        error
                     |      }
                     |
                     +--> on success: lab/cache/pipeline_state.pkl

app.py
  |
  +--> load_cache()                # last completed run only
  +--> read_active_run()           # current in-progress run if any
         |
         +--> Tab 1: Business View
         |      reads last completed cache + runs inference on selected row
         |
         +--> Tab 2: Model Evidence
         |      reads last completed cache metrics / XAI / diagnostics
         |
         +--> Tab 3: Developer Trace
                if active_run.status == "running":
                  tail active_run.log_path in real time
                else:
                  show cached run provenance via cache_log_path
```

## File Plan

- Modify: `app.py`
- Modify: `run_stage.py`
- Modify: `src/bt5151_credit_risk/cache.py`
- Modify: `src/bt5151_credit_risk/state.py`
- Create: `src/bt5151_credit_risk/run_status.py`
- Create: `src/bt5151_credit_risk/ui_trace.py`
- Create: `tests/test_cache.py`
- Create: `tests/test_run_status.py`
- Create: `tests/test_ui_trace.py`
- Modify: `tests/test_state.py`

## Out of Scope

- Cross-run comparison UI
- Full state diff viewer for every node
- WebSocket/SSE streaming; polling is sufficient
- Editable prompt/debug console in Gradio
- Replacing raw logs with a new trace artifact format
- Replacing Tabs 1 and 2 with live partial state from an in-progress run

## Task 1: Add cache provenance contract

**Files:**
- Modify: `src/bt5151_credit_risk/state.py`
- Modify: `src/bt5151_credit_risk/cache.py`
- Modify: `tests/test_state.py`
- Create: `tests/test_cache.py`

**Step 1: Write the failing tests**

Add tests for:
- cache round-trips `run_id`, `cache_log_path`, `cache_bundle_path`, `cache_saved_at`
- loaded `CreditRiskState` exposes those fields

**Note:** `CACHE_FILE = CACHE_DIR / "pipeline_state.pkl"` is bound at module **import time** (cache.py line 28). Monkeypatching `CACHE_DIR` after import has no effect — `save_cache` and `load_cache` reference the already-resolved `CACHE_FILE`. Tests must monkeypatch `CACHE_FILE` directly to redirect I/O to `tmp_path`.

```python
def test_save_cache_persists_provenance_metadata(tmp_path, monkeypatch):
    import bt5151_credit_risk.cache as cache_mod
    monkeypatch.setattr(cache_mod, "CACHE_FILE", tmp_path / "pipeline_state.pkl")

    from bt5151_credit_risk.cache import save_cache, load_cache

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
    assert state.cache_log_path.endswith(".log")
    assert state.cache_bundle_path.endswith(".json")
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_cache.py tests/test_state.py -q
```

**Step 3: Implement the minimal code**

- Add fields to `CreditRiskState`:
  - `cache_log_path: str | None = None`
  - `cache_bundle_path: str | None = None`
  - `cache_saved_at: str | None = None`
- Extend `save_cache(result, metadata=None)` so extra cache metadata is merged into the serialized payload
- Preserve `run_id` in cache keys

**Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_cache.py tests/test_state.py -q
```

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/state.py src/bt5151_credit_risk/cache.py tests/test_state.py tests/test_cache.py
git commit -m "feat: persist cache provenance metadata"
```

## Task 2: Add active run status contract

**Files:**
- Create: `src/bt5151_credit_risk/run_status.py`
- Create: `tests/test_run_status.py`

**Step 1: Write the failing tests**

Cover:
- write active run metadata
- mark run completed
- mark run failed
- missing file returns `None`

```python
def test_active_run_round_trip(tmp_path, monkeypatch):
    from bt5151_credit_risk.run_status import write_active_run, read_active_run, mark_run_completed

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

    mark_run_completed("20260416_120000")
    status = read_active_run()
    assert status["status"] == "completed"
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_run_status.py -q
```

**Step 3: Implement the minimal code**

Create `run_status.py` with:
- `ACTIVE_RUN_FILE = lab/cache/active_run.json`
- `write_active_run(...)`
- `mark_run_completed(run_id)`
- `mark_run_failed(run_id, error)`
- `read_active_run()`

Stored shape:

```json
{
  "run_id": "20260416_120000",
  "stage": "full",
  "row_index": 42,
  "status": "running|completed|failed",
  "pid": 12345,
  "log_path": "...",
  "bundle_path": "...",
  "started_at": "...",
  "completed_at": null,
  "error": null
}
```

**PID field is mandatory, but PID alone is not sufficient.** On Linux/WSL, PIDs are reused after a process exits — a new unrelated process can inherit the same PID, causing `read_active_run()` to falsely report the training run as still alive. Store `pid_start_time` alongside `pid` (use `psutil.Process(pid).create_time()` at spawn time). In `read_active_run()`, verify both: PID exists AND `psutil.Process(pid).create_time() == stored_pid_start_time`. If either check fails, rewrite to `"status": "failed", "error": "process died"` before returning.

```python
import psutil, os

def _is_process_alive(pid: int, pid_start_time: float) -> bool:
    try:
        p = psutil.Process(pid)
        return abs(p.create_time() - pid_start_time) < 1.0  # 1s tolerance
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_run_status.py -q
```

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/run_status.py tests/test_run_status.py
git commit -m "feat: add active run status contract"
```

## Task 3: Bind `run_stage --save-cache` to the exact log and bundle paths, and maintain live run status

**Files:**
- Modify: `run_stage.py`
- Test: `tests/test_cache.py`
- Test: `tests/test_run_status.py`

**Step 1: Write the failing test**

Add tests that verify:
- save-cache receives provenance metadata from `run_stage.py`
- active run file is written at start
- active run file is marked `completed` or `failed` at exit

```python
def test_run_stage_save_cache_includes_log_and_bundle_paths(monkeypatch):
    captured = {}

    def fake_save_cache(result, metadata=None, compress=3):
        captured["result"] = result
        captured["metadata"] = metadata
        return "/tmp/pipeline_state.pkl"

    monkeypatch.setattr("bt5151_credit_risk.cache.save_cache", fake_save_cache)
    # invoke main/run_stage path with --save-cache fixture setup
    assert captured["metadata"]["cache_log_path"].endswith(".log")
    assert captured["metadata"]["cache_bundle_path"].endswith(".json")
```

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_cache.py -q
```

**Step 3: Implement the minimal code**

- In `run_stage.py`, when `--save-cache` is used, pass:
  - `run_id`
  - `cache_log_path=str(log_file)`
  - `cache_bundle_path=str(Path("lab/logs") / f"analysis_bundle_{run_id}.json")`
  - `cache_saved_at`
- Do not guess the latest bundle file
- At run start, write `active_run.json`
- On success, mark it completed
- On exception, mark it failed with error summary

**Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_cache.py -q
```

**Step 5: Commit**

```bash
git add run_stage.py tests/test_cache.py tests/test_run_status.py
git commit -m "feat: bind saved cache and live run status to runner"
```

## Task 4: Build a raw-log parser and log-tail helpers for Developer Trace

**Files:**
- Create: `src/bt5151_credit_risk/ui_trace.py`
- Create: `tests/test_ui_trace.py`

**Step 1: Write the failing tests**

Cover:
- parses node starts from `>>> node-name`
- groups warnings/errors under the active node
- extracts token usage summary
- marks stage sections like `--- preprocessing ---`
- supports incremental tailing from a byte offset
- handles missing log file gracefully

```python
def test_parse_stage_log_builds_node_cards(tmp_path):
    log_path = tmp_path / "stage_full_20260416_120000.log"
    log_path.write_text(
        "12:00:00 INFO bt5151_credit_risk.graph  >>> train-models\\n"
        "12:00:01 WARNING bt5151_credit_risk.graph      baseline fold skipped\\n"
        "12:00:05 INFO run_stage  --- Token usage summary ---\\n"
        "12:00:05 INFO run_stage  Total LLM calls: 12\\n",
        encoding="utf-8",
    )

    trace = parse_stage_log(log_path)

    assert trace["run_summary"]["total_llm_calls"] == 12
    assert trace["cards"][0]["node_name"] == "train-models"
    assert trace["cards"][0]["status"] == "warn"
```

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q
```

**Step 3: Implement the minimal code**

Create `ui_trace.py` with pure helpers:
- `parse_stage_log(path: str | Path) -> dict`
- `read_log_tail(path: str | Path, offset: int = 0) -> tuple[str, int]`
- `summarize_log_card(lines: list[str]) -> dict`
- `build_trace_markdown(trace: dict) -> str`

Return shape:

```python
{
    "run_summary": {
        "stage": "full",
        "run_id": "20260416_120000",
        "total_llm_calls": 19,
        "total_tokens": 333991,
        "log_path": "...",
    },
    "cards": [
        {
            "node_name": "train-models",
            "status": "pass|warn|error|repair",
            "summary_lines": [...],
            "warning_lines": [...],
            "raw_lines": [...],
        }
    ],
}
```

**Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q
```

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/ui_trace.py tests/test_ui_trace.py
git commit -m "feat: add raw log parser for developer trace tab"
```

## Task 5: Reshape `app.py` around one global customer selector, live training control, and three tabs

**Files:**
- Modify: `app.py`
- Test: `tests/test_ui_trace.py`

**Step 1: Write the failing tests**

Create `tests/test_app.py`. Cover:
- global customer selector value is reused by all tabs
- app renders a clear "No model trained yet" state when cache is absent (cold-start path — do not crash)
- train button starts a background subprocess instead of blocking the UI
- train button is disabled (or shows a warning) when `active_run.status == "running"` and PID is alive — no concurrent runs

**Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 3: Implement the minimal code**

Refactor layout to:
- move row selector above tabs
- add `Train / Refresh Pipeline` button near the header
- add app-level training status text
- rename tabs to:
  - `Business View`
  - `Model Evidence`
  - `Developer Trace`
- keep `cb_load_customer` and `cb_predict`, but wire them to the shared row selector
- add callback to start `run_stage.py full <row_index> --save-cache` in a background subprocess — **do not write `active_run.json` from the app**; `run_stage.py` is the sole writer of that file (Task 3)
- **Concurrent run guard**: before spawning, call `read_active_run()`; if `status == "running"` and the process is confirmed alive (PID + start_time check), return "Training already in progress (run {run_id})" and do not spawn
- Tabs 1 and 2 must continue to use the last completed cache while training is running
- Cold-start: if `load_cache()` returns `None`, show "No trained model yet — click Train to begin" in Tab 1 and Tab 2 without crashing

**Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: restructure gradio app into three tabs"
```

## Task 6: Implement Tab 1 Business View

**Files:**
- Modify: `app.py`
- Test: `tests/test_app.py`

**Step 1: Write the failing test**

Test that `cb_predict()` returns:
- decision header (predicted class label, not integer)
- confidence band string
- short business explanation from `risk_explanation`
- recommended action label and rationale

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 3: Implement the minimal code**

Tab 1 should show:
- customer profile table (key fields only — not all 39 features)
- predicted class hero card (Good / Standard / Poor with colour coding)
- confidence badge
- business explanation summary from `risk_explanation["summary"]`
- recommended action badge + rationale from `recommended_action`

Move probability chart and SHAP waterfall out of the first visual row; make them secondary (below the fold or in an accordion) so the tab reads business-first.

**Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: implement business-first prediction tab"
```

## Task 7: Implement Tab 2 Model Evidence

**Files:**
- Modify: `app.py`
- Test: `tests/test_app.py`

**Note on confusion matrix:** `compute_multiclass_metrics` already stores `confusion_matrix` as a list-of-lists in `evaluation_results[model_name]["confusion_matrix"]`. No pipeline changes needed — just render from state.

**Step 1: Write the failing test**

Test that the model evidence callback returns:
- metrics DataFrame with selected model marked
- confusion matrix as a plottable 2-D list for the selected model
- selected-model justification string
- XAI methods used list

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 3: Implement the minimal code**

Tab 2 should show (in order):
- confusion matrix heatmap from `evaluation_results[selected_model]["confusion_matrix"]` using `class_names` as axis labels
- per-class metrics table (precision / recall / F1 / support per class, for selected model)
- selected-model justification from `state.selection_justification`
- global SHAP importance bar chart from `state.global_xai_results["shap"]["importance"]` if present
- method-gating summary: which of SHAP / PFI / ALE / PDP was used and why, from `state.global_xai_results["methods_used"]`
- **hypothesis validation table**: from `state.training_diagnostics["hypothesis_validation"]` — one row per EDA tested prediction, showing predicted value → actual value → confirmed/refuted verdict. This is the most rubric-relevant element in the tab.

**Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add model evidence tab"
```

## Task 8: Implement Tab 3 Developer Trace with live-watch mode

**Files:**
- Modify: `app.py`
- Modify: `src/bt5151_credit_risk/ui_trace.py`
- Modify: `src/bt5151_credit_risk/run_status.py`
- Test (parser logic): `tests/test_ui_trace.py`
- Test (Gradio callback / polling / log routing): `tests/test_app.py`

**Test split rule:** `test_ui_trace.py` covers pure functions (`parse_stage_log`, `read_log_tail`, `build_trace_markdown`). `test_app.py` covers the Gradio-level callback behavior: which log file gets selected, when polling triggers cache reload, graceful degradation.

**Step 1: Write the failing tests**

In `tests/test_ui_trace.py` — pure log routing logic:
- parser picks up node cards and warnings from log content
- `read_log_tail` returns new content and updated byte offset

In `tests/test_app.py` — Gradio callback behavior:
- `cb_developer_trace` uses `active_run.log_path` when PID is alive
- `cb_developer_trace` falls back to `state.cache_log_path` when no active run
- missing log path returns friendly string, not exception

Test that Developer Trace:
- uses `active_run.log_path` when a run is currently in progress (PID + start_time confirmed alive)
- otherwise uses `state.cache_log_path`, never the "latest" log on disk
- shows provenance header (run_id, log path, bundle path)
- renders node cards
- degrades gracefully when the bound log file is missing (shows friendly message, not crash)

```python
def test_developer_trace_prefers_active_run_log_over_cache_log(tmp_path, monkeypatch):
    bound_log = tmp_path / "stage_full_20260416_120000.log"
    bound_log.write_text("12:00:00 INFO bt5151_credit_risk.graph  >>> train-models\n", encoding="utf-8")

    # Simulate active run pointing at bound_log with a live PID (current process)
    import os
    active = {
        "run_id": "20260416_120000", "status": "running",
        "pid": os.getpid(), "log_path": str(bound_log),
    }
    monkeypatch.setattr("bt5151_credit_risk.run_status.read_active_run", lambda: active)

    trace_md = cb_developer_trace(state=None)
    assert "20260416_120000" in trace_md
    assert "train-models" in trace_md
```

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q
```

**Step 3: Implement the minimal code**

Add a callback that:
- reads `active_run.json` (including PID check — stale runs are auto-marked failed)
- if `status == "running"` and PID alive: tails `active_run.log_path` using `read_log_tail`
- otherwise: reads `state.cache_log_path` (the log bound at cache-save time, not "latest")
- calls `parse_stage_log`
- renders:
  - provenance header (run_id, log path, bundle path, status badge)
  - run summary (total LLM calls / tokens / duration when available)
  - node timeline cards (name, status badge, summary lines, warning highlights)
  - expandable raw log text per card

Use Gradio polling (`demo.load(..., every=5)` or a `gr.Timer(value=5)`) rather than websockets. **Do not use `every=1`** — a 30-minute training run at 1-second polling is 1800 filesystem reads for nothing. 5 seconds gives adequate live feedback.

**Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q
```

**Step 5: Commit**

```bash
git add app.py src/bt5151_credit_risk/ui_trace.py tests/test_ui_trace.py
git commit -m "feat: add live developer trace tab from raw logs"
```

## Task 9: Reload completed cache after live training finishes

**Files:**
- Modify: `app.py`
- Test: `tests/test_ui_trace.py`

**Step 1: Write the failing test**

Add to `tests/test_app.py`. Test that when `active_run.status` transitions from `running` to `completed`:
- the app reloads cache (the module-level `_state` global is invalidated)
- Tabs 1 and 2 pick up the new completed model on next interaction
- Tab 3 keeps showing the same completed run's trace (bound by `run_id`, not re-pointed to a newer log)

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 3: Implement the minimal code**

- Add a lightweight cache refresh helper in `app.py`: `_invalidate_cache()` sets module-level `_state = None`
- On the trace polling callback, after calling `read_active_run()`: if status just became `completed` (compare against last-seen status stored in a module-level variable), call `_invalidate_cache()`
- Do not reload continuously while the run is still `running` — only on the `running → completed` transition
- Thread safety: use a simple `threading.Lock` around `_state` reads/writes since Gradio may call callbacks concurrently

**Step 4: Run the test to verify it passes**

Run:

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q
```

**Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: refresh cache after live training completes"
```

## Task 10: Final regression and manual verification

**Files:**
- No new code

**Step 1: Run focused tests**

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_cache.py tests/test_run_status.py tests/test_ui_trace.py tests/test_app.py tests/test_state.py -q
```

**Step 2: Run full suite**

```bash
PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q
```

**Step 3: Generate a fresh cache**

```bash
PYTHONPATH=src .venv/bin/python3 run_stage.py full 42 --save-cache
```

**Step 4: Launch the app**

```bash
PYTHONPATH=src .venv/bin/python3 app.py
```

**Step 5: Manual acceptance checklist**

- **Cold-start**: launching `app.py` with no `pipeline_state.pkl` shows "No trained model yet — click Train to begin" on Tabs 1 and 2 without crashing
- Tab 1 shows predicted class, confidence, business explanation, and recommended action for row 42
- Tab 2 shows confusion matrix heatmap, per-class F1 table, model justification, global SHAP, and hypothesis validation table
- Clicking `Train / Refresh Pipeline` starts a live run without freezing the UI
- Clicking `Train / Refresh Pipeline` a second time while a run is in progress shows "Training already in progress" instead of spawning a second subprocess
- Tab 3 shows live progress from the active bound log while training is still running (polls every 5s)
- Tabs 1 and 2 continue to show the last completed cached model until the live run finishes
- After completion, the app reloads the new cache; Tab 3 still shows the same completed run's trace
- If training crashes (simulate by killing the subprocess), Tab 3 detects the dead PID and shows "failed" status within one poll cycle
- Missing log path on Tab 3 shows a friendly "Log not found" message instead of a crash

**Step 6: Commit**

```bash
git add app.py run_stage.py src/bt5151_credit_risk/cache.py src/bt5151_credit_risk/state.py src/bt5151_credit_risk/run_status.py src/bt5151_credit_risk/ui_trace.py tests/test_cache.py tests/test_run_status.py tests/test_state.py tests/test_ui_trace.py tests/test_app.py
git commit -m "feat: ship real-time three-tab gradio demo with live developer trace"
```
