# Implementation Log

## 2026-04-17 — Live pipeline rail overlays JSONL completion status on raw logs

**Context:** During a live run, the Developer Trace sidebar could show `global-xai` as blue/running even after the right-hand raw log card showed global XAI had produced its completion summaries. The backend was not stuck: the same run continued through `local-xai`, `shortcut-feature-audit`, `interpret-global-xai`, `interpret-local-xai`, packaging, inference, explanation, cache save, and run completion. The root cause was the live-trace compromise: raw logs are freshest but only contain `>>> node-start` markers, while structured JSONL records true node completion but can lag during long nodes.

**Changes (`src/bt5151_credit_risk/ui_trace.py`, `app.py`, `tests/test_ui_trace.py`):**
1. Added `parse_live_trace_artifacts(log_path, trace_path)` to parse the live raw log and overlay structured JSONL completion statuses by node occurrence.
2. The pipeline rail now keeps completed live nodes green when JSONL confirms completion, while still marking a newer raw-log node blue if JSONL has not completed it yet.
3. Tab 3 live polling uses the hybrid parser for the pipeline rail. The right-hand log remains raw-log based for freshness.
4. Added regression tests for both sides of the behavior: completed `global-xai` must not pulse, and a raw-log node after the last JSONL completion must pulse.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py tests/test_app.py::TestDeveloperTrace -q` → 30 passed. Real-log sanity check on `stage_full_20260417_190732.log` + `trace_events_20260417_190732.jsonl` shows `global-xai`, `local-xai`, `shortcut-feature-audit`, `interpret-global-xai`, and `interpret-local-xai` green with zero pulsing nodes after completion.

**Must remain true:**
- Raw logs remain the live content source during active runs.
- JSONL completion events should correct sidebar status when they exist.
- The sidebar should only pulse for the newest raw node that has no matching completion event yet.

## 2026-04-17 — Deterministic feature engineering is now the default production path

**Context:** The FE node repeatedly produced generated artifacts that failed the deterministic contract: protected top-MI parents such as `Monthly_Inhand_Salary` were dropped by correlation heuristics, and declared ratio/product lineage did not replay against the actual engineered outputs. These failed attempts did not poison training once deterministic fallback passed, but they wasted repair cycles, looked like stuck runs in the UI, and made the demo path depend on unreliable generated code.

**Changes (`src/bt5151_credit_risk/config.py`, `src/bt5151_credit_risk/graph.py`, `src/bt5151_credit_risk/feature_engineering.py`, tests):**
1. Added `FEATURE_ENGINEERING_MODE`, defaulting to `deterministic`. Set `BT5151_FEATURE_ENGINEERING_MODE=llm` only when intentionally experimenting with LLM-generated FE code.
2. `generate_feature_engineering_code_node()` now emits deterministic safe FE code by default, preserving validated preprocessing columns and encoding remaining categoricals before validation/training.
3. The deterministic FE generator now records whether it was used as primary deterministic mode or as a repair-exhaustion fallback.
4. Added a regression test proving deterministic mode does not call the LLM FE codegen path.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_graph.py tests/test_feature_engineering.py tests/test_feature_engineering_validation.py -q` → 28 passed.

**Must remain true:**
- Invalid LLM-generated FE artifacts must never enter training.
- The assignment/demo path should run end-to-end without relying on FE repair loops.
- LLM FE remains opt-in experimentation, not the default production contract.

## 2026-04-17 — Live Developer Trace now prefers raw logs while a run is active

**Context:** A frontend run appeared stuck at `validate-feature-engineering` even though the backend had already exhausted FE repair attempts, used the deterministic fallback, passed FE validation, and moved into `train-models`. The live UI was reading `trace_events_*.jsonl` first. That structured trace only emits node completion events, so it stayed stale throughout long-running nodes. The raw stage log already contained `>>> train-models`, but it was only used as fallback.

**Changes (`app.py`, `tests/test_app.py`):**
1. `cb_developer_trace()` now prefers `active_run.log_path` over `active_run.trace_path` while `active_run.status == "running"` and the PID is alive.
2. Completed, failed, cached, and historical traces still prefer structured trace artifacts, preserving the richer post-run cards.
3. Added a regression test proving an alive active run shows the raw-log `train-models` marker instead of stale structured-trace `validate-feature-engineering`.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py::TestDeveloperTrace tests/test_ui_trace.py -q` → 28 passed.

**Must remain true:**
- Live runs should show long-running in-progress nodes from the raw log.
- Failed/completed/cached runs should keep using structured traces when available.
- Historical run selection must remain pinned and must not be overwritten by polling.

## 2026-04-17 — FE execution contract fixed plus deterministic fallback safety net

**Context:** The latest full run failed in feature engineering before writing artifacts. The first generated FE code crashed on a missing interaction parent in its raw-parent snapshot. The repair code then correctly tried to encode deferred categoricals, but referenced `deferred_categorical_columns` as if it were available inside the generated module. The execution wrapper only called `engineer_features(train_df, test_df, workspace_path)`, so that global was missing, the code silently skipped the deferred encoding block, and `Occupation` remained non-numeric until the assertion killed the run.

**Changes (`src/bt5151_credit_risk/feature_engineering.py`, `src/bt5151_credit_risk/graph.py`, tests):**
1. `execute_feature_engineering()` now accepts `deferred_categorical_columns`, writes it to the FE workspace, and injects it into the generated module before calling the entrypoint. The graph passes `state.deferred_categorical_columns` into execution.
2. FE lineage validation now accepts generated categorical lineage aliases: `input_stage="pre_fe_raw_categorical"` and operation alias `frequency_encoding`.
3. Added `deterministic_feature_engineering_fallback_code()`: when all LLM FE repair attempts are exhausted, the repair node returns conservative deterministic FE code instead of raising immediately. The fallback preserves preprocessed columns, encodes object columns, fills numeric gaps using train medians, and writes `feature_engineering_report.json` plus `feature_lineage.json`.
4. Added regression tests for deferred-categorical injection, categorical lineage aliases, and fallback artifact validity.

**Verification:** Replayed the exact latest failed FE repair workspace (`feature_engineering_4343bab9a9dc451c81b5a426a67c0760`) through the patched executor with `{"Occupation": 15}`; execution returned `success=True` and validation returned `passed=True` with no errors or lineage violations. `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_feature_engineering.py tests/test_feature_engineering_validation.py tests/test_graph.py -q` → 27 passed.

**Must remain true:**
- Generated FE code must receive deferred categorical metadata at execution time, not only in the LLM prompt payload.
- Exhausting LLM FE repair attempts should not kill the whole pipeline if a deterministic safe representation can be produced.
- The fallback must be logged and conservative; it must not silently invent high-risk engineered features.

## 2026-04-17 — Feature lineage replay now uses exact report formulas

**Context:** The latest full run reached feature engineering, produced a structurally valid repaired FE artifact, and then failed validation because `feature_lineage.json` only declared coarse operations such as `ratio` or `product`. The validator replayed those as simple `a / b` or `a * b`, falsely rejecting valid formulas like `(Monthly_Inhand_Salary - Total_EMI_per_month) / Monthly_Inhand_Salary`, `Num_Credit_Inquiries / (Num_Credit_Card + 1)`, and `Outstanding_Debt * Interest_Rate / 100`.

**Changes (`src/bt5151_credit_risk/feature_engineering.py`, `src/bt5151_credit_risk/graph.py`, `tests/test_feature_engineering_validation.py`):**
1. Added a safe arithmetic formula replay path for formulas declared in `feature_engineering_report.json`. It supports column names, `df["column"]` references, numeric constants, `+`, `-`, `*`, `/`, unary signs, and `log1p`.
2. Formula replay is now preferred over coarse lineage operations when available. The old operation replay remains as fallback for older artifacts.
3. Added documented fill/clip candidate handling for lineage entries with stability metadata, so formula replay can match generated ratio stabilization without accepting unrelated math.
4. FE validation now logs lineage violations directly in the stage log instead of hiding the true cause inside `feature_contract_report.json`.
5. Added a regression test reproducing the composed-formula false failure from the latest run, plus kept the negative log-transformed-parent test green.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_feature_engineering_validation.py -q` → 12 passed. Replaying the patched validator against the actual repaired workspace from `feature_engineering_26b162e9fc9541bc9e29d8fd7c5b9311` now returns `passed=True` with no lineage violations.

**Must remain true:**
- FE lineage validation must reject wrong math, not reject correct formulas because the lineage operation enum is too coarse.
- Ratios/products must still use pre-FE raw numeric parents unless explicitly declared otherwise.
- Lineage failures must be visible in the stage log so future repair loops are diagnosable from the UI.

## 2026-04-17 — Deterministic preprocessing normalization for unstable generated columns

**Context:** The remaining full-run failures were no longer mainly about prompts or FE. The validator was correctly catching two real preprocessing instability families, but generated code still varied run to run: `Credit_History_Age` could be reparsed and then re-broken by implausible imputation, and `Type_of_Loan` could still carry double-encoded missingness (`Type_of_Loan_missing` plus `Type_of_Loan_Not Specified`) even after repair. The root issue was architectural: we were asking prompt guidance to enforce artifact-level invariants that belong at the preprocessing contract boundary.

**Changes (`src/bt5151_credit_risk/preprocess.py`, `src/bt5151_credit_risk/graph.py`, `tests/test_preprocess.py`):**
1. Added `normalize_preprocessing_artifacts()`, called inside `validate_preprocessing_output()` before the structural/semantic checks run. The normalizer reparses `Credit_History_Age` from `raw_frame.csv`, caps it by plausible adulthood tenure using cleaned `Age`, fills residual gaps from group/global medians, and rewrites the saved feature/cleaned frames.
2. Added deterministic normalization for `multi_value_set` missingness from the raw source: recompute `<col>_missing`, drop duplicate `Not Specified` dummy columns, coerce sibling indicators to binary, and zero sibling indicators when missingness fires.
3. Validation reports now carry a `deterministic_normalization` section, persist it into `preprocessing_contract_report.json`, and log applied normalization actions in the graph node output.
4. Added regression tests proving that broken `Credit_History_Age` and duplicate multi-value missingness are normalized before invariants run, so validation passes on repaired artifacts without relying on another LLM repair turn.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_preprocess.py -q` → 47 passed.

**Must remain true:**
- `Credit_History_Age` plausibility is enforced at the artifact boundary, not left to prompt compliance.
- `_missing` sentinels and `Not Specified` one-hot columns must never coexist for the same multi-value field.
- Deterministic normalization must remain observable in both the runtime log and `preprocessing_contract_report.json`.

**Follow-up hardening:** New runs after the first normalizer patch exposed two boundary bugs: `pd.Series.fillna(grouped_median)` can crash when pandas nullable integer months receive fractional medians, and `Credit_History_Age` cannot be safely capped using a generated `Age` column that may itself be malformed. The normalizer now reparses `Age` from raw values first, fills/clips it to [18, 100], forces numeric helper series to float before group/global fill, and then uses normalized age for the credit-history cap. Regression coverage now includes both failure modes plus a copied real failed workspace sanity check.

## 2026-04-17 — Raw stage logs now emit run-start / run-complete lifecycle nodes

**Context:** In the Developer Trace sidebar, successful historical runs loaded from raw `stage_full_*.log` files still showed the last node as blue/running. The root cause was architectural: only structured JSONL traces carried lifecycle events like `run_start` and `run_complete`. Raw logs had the `=== Stage 'full' (...) ===` and `=== Stage 'full' completed in ... ===` markers, but `parse_stage_log()` ignored them, so the pipeline renderer never saw an explicit terminal event and assumed the last actual node was still running. The sidebar also skipped `run_start` entirely even when structured traces contained it.

**Changes (`src/bt5151_credit_risk/ui_trace.py`, `tests/test_ui_trace.py`):**
1. Added raw-log lifecycle parsing for stage start and stage completion markers. `parse_stage_log()` now synthesizes `run_start` and `run_complete` cards from the existing `run_stage` log lines.
2. Stopped dropping `run_start` from the pipeline rail. The sidebar now shows a green lifecycle entry at the top when a run begins.
3. Updated pending-node logic so `run_start` stays green instead of being incorrectly converted into the single blue “running” node. When only `run_start` exists, the remaining main-path stages render as pending below it.
4. Added regression tests covering:
   - raw log start/completion lifecycle synthesis
   - completed raw runs showing no blue pulsing node
   - in-progress raw logs showing green `run-start` plus a blue current stage

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q` → 17 passed; `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py tests/test_ui_trace.py -q` → 46 passed. Real-log sanity check on `lab/logs/stage_full_20260416_012744.log` now shows cards beginning with `run_start` and ending with `run_complete`, and the generated pipeline HTML contains `run‑start` and `run‑complete` with no pulsing node.

**Must remain true:**
- Successful raw stage logs must terminate with a non-pulsing `run-complete` node.
- `run-start` must be visible in the sidebar for both structured traces and raw stage logs.
- The pipeline rail should only show a blue running node when the run has not reached a terminal lifecycle event.

## 2026-04-17 — Historical trace pinning + FE top-MI leakage filter

**Context:** Two separate bugs were distorting debugging. First, the Developer Trace dropdown looked broken because selecting any historical run was immediately overwritten by the 1-second poller, so every choice snapped back to the same live/cached trace. Second, several recent full runs failed in feature-engineering validation for the wrong reason: the FE top-MI preservation guard was using raw EDA leaders such as `Customer_ID`, `SSN`, and `ID`, even though preprocessing had correctly dropped them as identifiers/leakage.

**Changes (`app.py`, `src/bt5151_credit_risk/ui_trace.py`, `src/bt5151_credit_risk/graph.py`):**
1. Added `_historical_log_dir()` and explicit pinned historical selection state in the Gradio app. `cb_load_historical_log()` now returns the selected artifact key, and `cb_poll_trace()`/`cb_developer_trace()` accept that selection so timer ticks keep showing the chosen historical run instead of auto-reverting to the latest live/cached trace.
2. Cleared the trace memoization path when no artifact is available or when a selected historical file is missing, preventing stale pipeline rails from lingering beside an error message.
3. Changed `list_available_logs()` to order historical artifacts by actual file recency across JSONL traces and raw stage logs, while still preferring one structured trace per run when both exist. This removes the previous “all traces first, logs later” bias that made recent failures dominate the dropdown.
4. Added `_top_mi_features_for_fe_validation()` in `graph.py` to filter out identifier columns, group columns, target columns, and any columns explicitly marked `action=drop` before passing top-MI features into FE validation. The FE preservation guard now applies only to features that are expected to survive preprocessing.
5. Added regression tests for pinned historical selection surviving timer ticks, clearing back to auto mode, history ordering by recency rather than file type, and FE validation filtering out dropped identifier columns.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q` → 29 passed; `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_graph.py -q` → 11 passed; `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py -q` → 14 passed.

**Must remain true:**
- A manually selected historical run must stay pinned until the user clears the dropdown.
- Timer ticks must never overwrite a pinned historical trace with the current live/cached run.
- FE top-MI preservation must ignore columns already dropped by leakage/identifier policy.
- Historical run ordering in the dropdown should reflect real recency, not artifact type priority.

## 2026-04-17 — Developer Trace pipeline ordering and failure visibility

**Context:** The sticky pipeline rail was visible, but the node order was misleading. The sidebar used an old coarse canonical stage list (`preprocess`, `feature-engineering`) while the structured trace emits finer nodes (`inspect-preprocessing-code`, `execute-generated-preprocessing`, `validate-preprocessing-output`, repair loops). Unknown conditional nodes were appended at the bottom, making preprocessing repairs look like they happened after inference. Lifecycle failures were also hidden because `run_failed` events were emitted as `node="__run__"` and the renderer skipped `__*` nodes.

**Changes (`src/bt5151_credit_risk/ui_trace.py`, `tests/test_ui_trace.py`):**
1. Rebuilt the pipeline rail as an execution timeline first: actual trace events are rendered in emitted order, so retry/repair nodes appear beside the stage that triggered them.
2. Replaced the stale coarse stage skeleton with the current main graph path and removed obsolete pending nodes such as `preprocess`, `feature-engineering`, and `recommend-action`.
3. Kept conditional repair nodes out of the pending skeleton; they appear only when the trace actually emits them, or as the immediate next pending step after a failed conditional route.
4. Normalized lifecycle events with `node="__run__"` into visible cards (`run-complete`, `run-failed`, `cache-saved`) so terminal failures are no longer hidden from the sidebar.
5. Increased pipeline card height, padding, width, and connector length so each stage reads as a real block instead of a flat strip.
6. Added regression tests for retry-loop ordering and terminal `run_failed` visibility.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py tests/test_ui_trace.py -q` → 40 passed; `PYTHONPATH=src .venv/bin/python3 -m pytest tests/ -q` → 177 passed. Browser QA on port 7862 confirmed the sidebar order now shows `validate-preprocessing-output` failure → `repair-preprocessing-code` → second inspect/execute/validate attempt → `generate-feature-engineering-code`.

**Must remain true:**
- Conditional retry nodes must be rendered from actual trace order, not appended after the canonical pipeline.
- `run_failed` lifecycle events must be visible in the Developer Trace pipeline rail.
- Obsolete coarse nodes must not reappear unless the graph emits those exact nodes again.

## 2026-04-17 — Developer Trace sticky pipeline with normal log page scroll

**Context:** Tab 3 should let users keep the pipeline progress visible while reading a long live/historical log, without putting the log into an inner scroll panel.

**Changes (`app.py`, `tests/test_app.py`):**
1. Moved custom Gradio CSS into `_app_css()` so layout contracts are testable.
2. Changed Developer Trace sticky behavior from the inner `#trace-pipeline` HTML component to the whole `#trace-left-col` wrapper. This keeps the pipeline/sidebar visible while the right log continues to use normal document scroll.
3. Root-cause fix from browser QA: Gradio's outer `.gradio-container` had `overflow: hidden`, which blocks CSS sticky even when the sidebar computes as `position: sticky`. The app CSS now overrides that container to `overflow: visible`.
4. Constrained the left pipeline rail with `max-height: calc(100dvh - 112px)` and `overflow-y: auto`. The pipeline itself can scroll when it is taller than the viewport; the log remains normal page content.
5. Follow-up browser QA found a second Gradio layout bug: the separate `gr.Markdown("#### Pipeline")` heading and `gr.HTML(elem_id="trace-pipeline")` body were laid out side-by-side inside the sticky column, then clipped by `overflow-x: hidden`, making the sidebar look empty. The title and dynamic pipeline body are now wrapped into one `gr.HTML` via `_wrap_pipeline_html()`.
6. Added CSS guards so `#trace-log` and right-column wrappers stay `overflow: visible`; no `max-height` or `overflow-y: auto` is introduced for the log.
7. Added regression tests that assert the left pipeline column is sticky, the log remains normal page content, and the pipeline title/body stay in one HTML component.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q` → 27 passed. Browser QA on port 7862 confirmed that after scrolling to `scrollY=1300`, `#trace-left-col` stays at `top=88px`; `#trace-log` remains `overflow=visible` and `max-height=none`; `.trace-pipeline-body` sits inside the left rail (`contentInside=true`).

**Must remain true:**
- Developer Trace log uses normal page scroll, not an inner scroll container.
- Sticky behavior belongs to `#trace-left-col`, not only the inner pipeline HTML.
- Mobile layouts disable sticky so the sidebar does not cover the log.

## 2026-04-17 — Tab 1 inference UX + skill prompt overhaul

**Context:** Three independent problems were addressed together: (1) `_run_explain_step` was discarding the `recommended_action` computed inside `explain_risk_node` and calling the weak `recommend-action.md` skill separately with minimal context; (2) `explain-risk.md` listed analysis bundle fields but gave no explicit instructions on how to chain each layer's evidence to the specific customer's prediction; (3) `recommend-action.md` was a 3-line placeholder with no schema, principles, or output contract.

**Changes:**
1. **`app.py` — architecture fix**: `_run_explain_step` now returns both `risk_explanation` and `recommended_action` from `explain_risk_node` (the node already pops the action from the single LLM response). Removed `_run_recommend_step` and its call in `cb_predict`. Saves 5-10s per inference request by eliminating a redundant LLM call.
2. **`app.py` — PDP position chart**: New `_pdp_position_fig()` builder renders per-class PDP curves for the top SHAP features with a vertical marker at the customer's actual feature value. Falls back to a percentile bar chart when full curve data is absent. Added `pdp_position_plot` component to Tab 1.
3. **`app.py` — casebook comparison chart**: New `_casebook_comparison_fig()` builder renders a side-by-side SHAP waterfall comparing this customer to the nearest casebook archetype (looked up from `state.local_xai_cases` by `row_index`). Added `casebook_comparison_plot` component to Tab 1.
4. **`app.py` — `_build_action_md`**: Now surfaces the `urgency` field from the new recommended_action schema.
5. **`app.py` — `cb_predict` yield tuples**: Extended from 6 to 8 items (added `pdp_position_plot`, `casebook_comparison_plot`). Removed step 5 (recommend LLM call); step 5 is now chart rendering. Progress bar recalibrated for ~20-25s total.
6. **`skills/recommend-action.md`**: Complete rewrite — full decision matrix (predicted class × caution level × casebook signal), output schema with `action`, `urgency`, `rationale`, `key_evidence`, `monitoring_conditions`, `information_gaps`. Explicit evidence-first posture; rationale must cite named fields with actual values.
7. **`skills/explain-risk.md`**: Added "How to use the analysis bundle for THIS customer" section with layer-by-layer chain-of-evidence instructions. Added `bundle_link` field to `key_drivers` schema. Added `curve_steepness` field to `pdp_context`. Added `customer_evidence` and `customer_relevance` fields to `hypothesis_validation`. Strengthened `recommended_action.rationale` rule to require at least one specific cited value.

**Tradeoffs:**
- The embedded `recommended_action` from `explain-risk.md` has full context (bundle, casebook, PDP, hypothesis chain). The standalone `recommend-action.md` still exists as a fallback skill but is no longer called during Tab 1 inference.
- PDP position chart requires `global_xai_results.pdp` in state — will render the fallback percentile bar chart when PDP was not computed (ALE-only gating).

**Must remain true:**
- `_run_explain_step` must return a 2-tuple `(risk_explanation, recommended_action)`.
- `cb_predict` must yield 8-item tuples; the outputs list must have the same 8 components in the same order.
- `recommend-action.md` decision matrix must be the floor, not the ceiling — evidence can elevate the action above the matrix default.
- Never fabricate SHAP values, PDP positions, or similarity scores in either skill prompt.

## 2026-04-17 — Tab 2 staged evidence loading to avoid Gradio payload stalls

**Context:** After converting Tab 2 plots to static images, the browser could still freeze when clicking the evidence refresh button after an app restart. The server-side callback completed, but one callback still returned metrics, confusion matrix, SHAP, PDP/ALE, casebook, dependence, and hypothesis markdown together. On slower local browsers this could stall while Gradio hydrated several output components from one response.

**Changes (`app.py`, `tests/test_app.py`):**
1. Kept a single top-right `Load / Refresh Evidence` button for Tab 2. One user action now starts the staged evidence refresh; there is no separate advanced-chart button for graders to understand.
2. Split the backend work into sequential callbacks: summary first, then `cb_model_shap_chart()` → `cb_model_pdp_ale_charts()` → `cb_model_dependence_chart()`. The UI chains them with `.then(...)`, so one click still works but the payload arrives as smaller responses instead of one large multi-chart response.
3. Added per-run chart caching for the individual chart callbacks and cleared it when the pipeline cache is invalidated.
4. Updated app tests for generator-style Business View callbacks and added regression coverage that each heavy chart can load independently.

**Verification:**
- `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q` → 25 passed.
- Real-cache timing check: summary 4.27s including first cache load, SHAP 0.41s, PDP/ALE 1.45s, dependence 0.54s.

**Must remain true:**
- Tab 2 must not use `demo.load()` or `tab.select()` for model evidence refresh.
- Summary and advanced charts must remain staged backend callbacks behind one visible button.
- Advanced charts should stay sequentially chained or individually callable, not merged back into one large multi-output response.

## 2026-04-16 — Fix FE deferred categorical encoding failures

**Root cause:** `generate_feature_engineering_code()` and `repair_feature_engineering_code()` did not receive `deferred_categorical_columns` from state. The LLM had to infer which columns were deferred by inspecting object-dtype in the 5-row train sample — it consistently generated interaction features but skipped encoding Occupation/Payment_of_Min_Amount/Payment_Behaviour. All 3 repair attempts hit the same validation error (`[train_fully_numeric] non-numeric/non-bool columns`) because the repair prompt named the error but not the specific columns or the exact encoding contract.

**Changes:**
1. `feature_engineering.py`: Added `deferred_categorical_columns: dict | None = None` parameter to both `generate_feature_engineering_code()` and `repair_feature_engineering_code()`. When non-empty, included in LLM payload as `deferred_categorical_columns`.
2. `graph.py`: `generate_feature_engineering_code_node` and `repair_feature_engineering_code_node` now pass `state.deferred_categorical_columns`. Generate node also logs the deferred columns before calling the LLM.
3. `skills/generate-feature-engineering-code.md`: Added "MANDATORY: Encode deferred categorical columns" section before Technical guardrails. Provides the cardinality-based encoding table and frequency-encode code pattern explicitly.
4. `skills/repair-feature-engineering-code.md`: Added "MANDATORY: Handle deferred categorical columns first" section before Reasoning steps. Provides cardinality rules, frequency encoding pattern, and assert pattern for both views.

**Must remain true:**
- `deferred_categorical_columns` must be passed to both generate and repair (not just one)
- Encoding is view-appropriate: one-hot for linear, frequency for tree at medium cardinality
- FE function must assert no object-dtype columns before writing each view's CSV

## 2026-04-16 — Fix Tab 2 / Tab 3 frozen in Gradio 6.7+ (lazy tab loading)

**Root cause:** Gradio 6.7 introduced lazy loading of inactive tab components. `demo.load()` callbacks targeting components inside inactive tabs (Tab 2, Tab 3) fire on page load but the outputs cannot be applied to lazily-loaded components — creating permanent loading spinners that block the tabs from becoming interactive. Additionally, `gr.Timer` was placed inside `with gr.Tab("Developer Trace"):`, making it tab-scoped: the timer only ticked when Tab 3 was visible, accumulating a backlog of events that all fired at once when the tab was clicked, freezing the UI.

**Changes (`app.py`):**
1. Removed `demo.load(cb_model_overview, ...)` from inside Tab 2. Replaced with `tab_evidence.select(cb_model_overview, ...)` — fires when Tab 2 is first selected, not on global page load. Tab is captured as `with gr.Tab("Model Evidence") as tab_evidence:`.
2. Removed `demo.load(cb_poll_trace, ...)` from inside Tab 3 (no longer needed; timer handles polling).
3. Moved `gr.Timer(value=5)` outside all `gr.Tab()` contexts to the Blocks level. Added `trigger_mode="always_last"` (discards queued ticks, only processes most recent) and `concurrency_limit=1` (prevents parallel executions) to `timer.tick()`.

**Verification:** App restarted at PID 493363, HTTP 200. `/cb_poll_trace_1` duplicate endpoint gone (was two registrations; now correctly one). `cb_model_overview` API returns charts in ~0.1s. 21 app tests pass.

**Must remain true:**
- `gr.Timer` must be at Blocks level, never inside `gr.Tab`
- Tab-specific initial loads use `tab.select()`, not `demo.load()`
- `trigger_mode="always_last"` must stay on the timer tick to prevent queue buildup during long idle periods

## 2026-04-16 — EDA MI now includes label-encoded categoricals

**Context:** The programmatic EDA computed mutual information only on numeric columns, so ordinal-encoded categoricals (Credit_Mix, Month, Payment_of_Min_Amount) were invisible to the MI ranking. Credit_Mix is the dominant SHAP feature (0.5374 mean |SHAP|) but never appeared in the EDA discriminative features list, causing the hypothesis generation node to form predictions without knowing the top signal exists.

**Changes (`src/bt5151_credit_risk/eda.py`):**
1. `_compute_discriminative_features`: now accepts `categorical_cols` parameter. Each categorical is label-encoded and included alongside numerics in a single MI computation. Output rows tagged with `column_type: "numeric"` or `"categorical"`.
2. `_compute_class_separability`: now accepts `categorical_cols`; adds `categorical_class_modes` section with class-conditional top-3 value distributions per categorical column.
3. `_compute_categorical_association` (new): computes Cramér's V + chi-squared for each categorical vs target. Added as `categorical_association` section in the EDA report.
4. Added `scipy.stats.chi2_contingency` import.

**Why:** LLM hypothesis generation reads the EDA report. If Credit_Mix (the dataset's near-proxy for Credit_Score) is absent from the MI ranking, tested predictions will be wrong before any model runs. Fix ensures the unified MI ranking shows all features the model will actually see.

**Tradeoff:** Label encoding for MI is order-agnostic — MI doesn't care about numeric assignment, only mutual information. Cramér's V complements this with a scale-free measure that doesn't require encoding. No ordinal assumptions are baked into the MI step.

**Verification:** 169 tests pass. Smoke-tested on mini dataset: Credit_Mix appears in `top_discriminative_features` with `column_type: "categorical"`.

**Must remain true:**
- `top_discriminative_features` must include both numeric and categorical columns in a unified ranking
- Each result must have a `column_type` field so the LLM knows which columns needed encoding
- `categorical_association` section must exist in the EDA report

## 2026-04-16 — Gradio Model Evidence callback handles nested hypothesis validation

**Context:** In the real cached pipeline state, `training_diagnostics["hypothesis_validation"]` is a dict with `tested` and `supported` sections, not a flat list. The Gradio `cb_model_overview()` callback assumed list indexing and crashed with `KeyError: 0`, which made Tab 2 hang/fail and destabilized the tabbed UI experience.

**Changes:**
1. **Normalize hypothesis-validation shape in the UI callback** (`app.py`): `cb_model_overview()` now accepts either shape:
   - flat list of rows
   - nested dict with `tested` / `supported`
   The nested form is flattened into a markdown table with a `tier` column so the UI can render real cached pipeline outputs without crashing.
2. **Regression coverage for the real cache shape** (`tests/test_app.py`): Added a focused test proving the callback renders a nested `hypothesis_validation` dict and includes both `tested` and `supported` rows in the output markdown.

**Verification:**
- `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q`
- Direct callback check against the real cache via `cb_model_overview()` completed successfully and returned the expected tuple/markdown output.

**Must remain true:**
- `cb_model_overview()` must tolerate both flat-list and nested-dict `hypothesis_validation` payloads
- UI rendering code must not assume test fixtures match the exact persisted cache shape

## 2026-04-16 — Contract hardening: cross-field invariants, feature lineage, shortcut audit

**Context:** Run 012 and 013 shipped but the pipeline had three structural blind spots: (a) preprocessing only checked one-sided role invariants — a column tagged `numeric_continuous` that stayed string would pass the validator; (b) feature engineering had no way to prove what each engineered column actually computed — the FE report was a human-readable list of formula strings, not a replayable manifest; (c) XAI outputs like `Month` top-ranked by SHAP and `Credit_Mix` dominating the importance plot were flagged manually in discussion, with no systematic detection + ablation in-run. Plan: [docs/plans/2026-04-16-contract-hardening-lineage-shortcut-audit.md](../plans/2026-04-16-contract-hardening-lineage-shortcut-audit.md).

**Changes:**
1. **Preprocessing numeric dtype gate + cross-field invariants** ([preprocess.py](../../src/bt5151_credit_risk/preprocess.py), [tests/test_preprocess.py](../../tests/test_preprocess.py)): combined the one-sided `if numeric and ...` branches into a single block that fails *both* ways — if a role-declared `numeric_count`/`numeric_continuous` column is still object dtype, it now records `not_numeric_dtype`; non-negative, NaN, and inf checks live under the numeric branch. Added `validate_semantic_invariants()`: Age ∈ [18, 100] with output cardinality ≥30% of raw (gated to frames ≥100 rows to avoid tripping on toy fixtures), Credit_History_Age ∈ [0, 1000] months with ≤ (Age−18)·12·1.1, detection of non-multiple-of-12 values indicating month-component was dropped, missing-sentinel mutual exclusivity, and Not-Specified-alongside-_missing duplicate encoding. Return dict now includes `cross_field_violations`.
2. **Preprocessing contract report persisted** ([preprocess.py](../../src/bt5151_credit_risk/preprocess.py), [graph.py](../../src/bt5151_credit_risk/graph.py)): `validate_preprocessing_output()` writes `preprocessing_contract_report.json` to the workspace with merged role+invariant violations. Repair escalation signature now includes invariant violations so the capability-ceiling check catches repeated semantic failures.
3. **Feature lineage manifest** ([skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md), [skills/repair-feature-engineering-code.md](../../skills/repair-feature-engineering-code.md), [feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py), [tests/test_feature_engineering_validation.py](../../tests/test_feature_engineering_validation.py)): FE codegen prompt now requires `feature_lineage.json` with structured entries (`operation`, `input_stage`, `inputs`, `drop_reason`) drawn from enumerated vocabularies — no free-form formula strings. Added `validate_feature_lineage()` with replay: samples 20 rows, recomputes ratio/product/sum/difference/log1p from declared raw parents, asserts `|expected − actual| ≤ max(1e-4, 1e-4·|expected|)`. Rules enforced: `lineage_artifact_present`, schema, operation/input_stage/drop_reason enums, `ratios_use_raw_parents` (arithmetic ops must cite `pre_fe_raw_numeric`), `lineage_coverage_complete` (every engineered column accounted for), `top_mi_drop_requires_justification` (top-5 MI can only be dropped by `leakage` or `deterministic_duplicate`), `lineage_replay_matches`. `feature_contract_report.json` persisted to workspace. Covers Phases 2 + 3 of the plan.
4. **Shortcut-feature audit node** ([shortcut_audit.py](../../src/bt5151_credit_risk/shortcut_audit.py), [graph.py](../../src/bt5151_credit_risk/graph.py), [state.py](../../src/bt5151_credit_risk/state.py), [skills/interpret-global-xai.md](../../skills/interpret-global-xai.md), [tests/test_shortcut_audit.py](../../tests/test_shortcut_audit.py)): new `shortcut-feature-audit` node sits between `local-xai` and `interpret-global-xai`. Detects suspects via three signals — SHAP/PFI rank divergence (SHAP ≤5 but PFI >10), top-10 SHAP share dominance (>20%), and calendar-pattern names (`Month`, `Day*`, `Year`, `Index`, `Week`, `Quarter`). Ablates up to 2 suspects by zero-out (median for numeric, 0 for non-numeric) in the test view — no retrain — and computes ΔmacroF1. Verdicts: `|Δ| < 0.005 → weak_signal`, `Δ < −0.02 → real_signal`, else `inconclusive`. `shortcut_audit.json` persisted; `interpret-global-xai` now receives the audit and instructs the LLM to treat weak-signal features as untrusted in hypothesis validation.
5. **Test fixtures updated** ([tests/test_feature_engineering.py](../../tests/test_feature_engineering.py), [tests/test_graph.py](../../tests/test_graph.py)): existing "passing" FE fixtures and the end-to-end compiled-graph fake now emit a minimal `feature_lineage.json` so they satisfy the new contract. 149/149 tests pass (notebook smoke excluded as before).

**Tradeoffs:**
- Lineage enum accepted `one_hot` and `frequency_encode` beyond the strict plan enum because real FE outputs one-hot expansions, and rejecting them would cause false repair loops. `bin` and `interaction` without a declared operator are in the enum but skipped by replay (the operator isn't deterministically replayable without extra schema).
- Shortcut audit ablates by zero-out rather than retrain to stay within the plan's cost cap; verdict thresholds are absolute (0.005 / 0.02) rather than dataset-adaptive — future work if false-verdict rate bites.
- Age cardinality check is gated to ≥100 rows; below that, toy frames where `Age.nunique()` is naturally small would generate spurious violations.

**Must remain true:**
- `preprocessing_contract_report.json` and `feature_contract_report.json` must be persisted every run, regardless of pass/fail.
- Every engineered column must appear in `feature_lineage.json` as derived, passthrough, or one-hot expansion of a declared parent — no unaccounted columns.
- Ratios/products/sums/differences/interactions must declare `input_stage=pre_fe_raw_numeric`; log-transformed parents fail `ratios_use_raw_parents`.
- Top-5 MI raw features in `dropped_features` must carry `drop_reason ∈ {leakage, deterministic_duplicate}`; any other reason fails validation.
- `shortcut-feature-audit` runs before `interpret-global-xai` and its output is passed through the interpretation prompt so weak-signal features are de-emphasized in explanations.
- Replay sample uses `np.random.RandomState(42)` so lineage violations are reproducible across runs on the same data.

## 2026-04-16 — Three-tab UI: failed-run status, cold-start layout, trace poll efficiency

**Context:** Three bugs found in the live demo after shipping the three-tab UI.

**Changes:**
1. **Bug (High): Failed runs marked completed** ([run_stage.py](../../run_stage.py)): `run_stage()` catches all pipeline exceptions internally and returns `None`. The outer `main()` had a `try/except` that would call `mark_run_failed`, but it only fires for exceptions that *escape* `run_stage()` — which never happens. Added an explicit `result is None` check after the call: if `None`, call `mark_run_failed` and return early, never reaching `mark_run_completed`. Tradeoff: `run_stage()` still swallows the exception for logging purposes; `main()` now correctly reflects that outcome in `active_run.json`.
2. **Bug (High): Cold-start layout frozen** ([app.py](../../app.py)): `has_cache = state is not None` was captured once at `build_app()` time. Tabs 1 and 2 were rendered inside `if not has_cache / else` branches, so if the app started cold, no interactive components were created, and later cache reload had nothing to populate. Fix: removed `has_cache` branching entirely — all components always rendered. All callbacks (`cb_load_customer`, `cb_predict`, `cb_model_overview`) already handle `state is None` gracefully by returning placeholder data.
3. **Bug (Medium): Full log re-parse every poll** ([app.py](../../app.py)): `cb_developer_trace()` called `parse_stage_log()` on every 5-second tick regardless of whether the log had grown. Added module-level `_last_trace_log_path`, `_last_trace_log_size`, `_last_trace_md`. On each tick, compares `path.stat().st_size` to last-seen size; if unchanged, returns cached markdown immediately. Only re-parses when the file has actually grown.

**Must remain true:**
- `mark_run_completed` must never be called when `run_stage()` returned `None`
- `active_run.json` status must reflect actual pipeline outcome (completed vs failed)
- Tab 1 and Tab 2 components must be created unconditionally at app startup
- Developer Trace re-parse is gated on file growth, not wall-clock time

## 2026-04-16 — MI-first tiebreaker for correlated feature drops (validator + prompt)

**Context:** Run 013 dropped `Monthly_Inhand_Salary` (MI=0.5187, #2 ranked feature) because its correlation with `Annual_Income` exceeded |r|>0.95 after imputation. The existing FE skill prompt said "keep the one with higher variance" — the wrong criterion. Variance measures spread, not class-separating signal. MI directly measures class-separating signal, which is what matters for feature selection.

**Changes:**
1. **Skill prompt rule rewrite** ([skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md)): Rule 2 changed from "keep the higher-variance feature" to "keep the higher-MI feature." Added absolute protection: "NEVER drop a top-5 MI raw feature in favor of a lower-MI proxy unless the pair is a deterministic duplicate (identical values) or a leakage feature (derived from target)." The correlation drop rule now only applies when both features have MI rank > 5.
2. **Programmatic validator** ([src/bt5151_credit_risk/feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py)): Added `check_top_mi_not_dropped()` inside `validate_feature_engineering_output()`. Checks that each top-5 MI feature is present in the output (as itself or an explicit derived form like `_log`, `_bin`). Generates a repair-ready error message naming the rule and restoration path.
3. **Graph wiring** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)): `validate_feature_engineering_node()` now extracts top-5 MI features from `state.eda_report["top_discriminative_features"]` and passes them to the validator via the new `top_mi_features` parameter.

**Must remain true:**
- When |r| > 0.95, higher-MI feature wins — never higher-variance
- Top-5 MI raw features are protected from the correlation drop rule unless the pair is a deterministic duplicate or leakage feature
- Validator error messages must be repair-ready (name the rule and restoration path, not just "feature missing")

## 2026-04-16 — Three-tab Gradio UI with live developer trace

**Context:** The previous app was a single-tab prediction demo with no visibility into the training pipeline or model evidence. This change ships a full three-tab UI: Tab 1 is a business-first prediction view, Tab 2 exposes model evidence (confusion matrix, per-class metrics, global SHAP, hypothesis validation table), and Tab 3 is a live developer trace that polls the active run log while training and falls back to the cached run log when idle.

**Changes:**
1. **Cache provenance contract** ([src/bt5151_credit_risk/state.py](../../src/bt5151_credit_risk/state.py), [src/bt5151_credit_risk/cache.py](../../src/bt5151_credit_risk/cache.py), [tests/test_cache.py](../../tests/test_cache.py)): added `cache_log_path`, `cache_bundle_path`, `cache_saved_at` fields to `CreditRiskState`; `save_cache()` now accepts a `metadata` dict that merges these provenance fields into the persisted payload; `run_id` added to `CACHE_KEYS`. Both tabs and the trace view use the exact log bound at cache-save time, never "latest log on disk."
2. **Active run status contract** ([src/bt5151_credit_risk/run_status.py](../../src/bt5151_credit_risk/run_status.py), [tests/test_run_status.py](../../tests/test_run_status.py)): new module with `write_active_run`, `mark_run_completed`, `mark_run_failed`, `read_active_run`. Stores `pid` + `pid_start_time` (not just PID) to handle Linux/WSL PID reuse; `read_active_run` auto-rewrites stale-PID "running" records to "failed" before returning.
3. **run_stage.py provenance binding** ([run_stage.py](../../run_stage.py)): `run_stage.py` now writes `active_run.json` at run start and marks it completed/failed at exit. `save_cache` receives provenance metadata (`cache_log_path`, `cache_bundle_path`) bound to the run_id. `app.py` does NOT write `active_run.json` — `run_stage.py` is the sole writer.
4. **Raw log parser** ([src/bt5151_credit_risk/ui_trace.py](../../src/bt5151_credit_risk/ui_trace.py), [tests/test_ui_trace.py](../../tests/test_ui_trace.py)): pure-function module with `parse_stage_log` (builds node cards with pass/warn/error/repair status), `read_log_tail` (incremental byte-offset tail for live watching), `summarize_log_card`, `build_trace_markdown`. No Gradio dependencies.
5. **Three-tab app.py** ([app.py](../../app.py), [tests/test_app.py](../../tests/test_app.py)):
   - Global row selector and Train button above tabs; concurrent-run guard (PID + start_time) prevents double spawning
   - Cold-start: Tabs 1 and 2 show "No trained model yet" instead of crashing when no cache exists
   - Tab 1 Business View: business-first layout (decision hero card, confidence badge, explanation, action; SHAP in accordion below)
   - Tab 2 Model Evidence: confusion matrix heatmap, per-class metrics, model justification + XAI methods used, hypothesis validation table (EDA tested predictions vs. actual results — closes the loop for rubric)
   - Tab 3 Developer Trace: live log watch (5s poll via `gr.Timer`); routes to `active_run.log_path` when PID is alive, falls back to `state.cache_log_path` when idle; friendly "log not found" instead of crash
   - Cache auto-reloads on `running → completed` status transition; `threading.Lock` around `_state` for Gradio concurrency safety

**Must remain true:**
- `run_stage.py` is the sole writer of `active_run.json`; `app.py` never writes it
- Developer Trace log routing must use the explicitly bound log path, not the newest file in `lab/logs/`
- PID liveness check must verify both PID existence AND `pid_start_time` (1s tolerance) to handle Linux/WSL PID reuse
- `save_cache(result, metadata=None)` must accept and persist provenance fields; old callers without `metadata` must not break
- Gradio polling interval is 5s; never 1s (1800 reads for a 30-minute run is wasteful)

## 2026-04-15 — Run artifact alignment, duplicate-column guard, complementary ALE/PDP, and machine-readable confidence stats

**Context:** The latest full run completed end-to-end, but the saved artifacts exposed four remaining design gaps: the preserved stage log in `lab/logs/` did not stay aligned with the bundle timestamp, preprocessing was still vulnerable to repeated multi-hot duplicate-column failures, ALE was being treated as a gated alternative to PDP instead of a complement, and inference-time caution logic was reading a prose-only confidence summary.

**Changes:**
1. **Run-id aligned artifacts** ([run_stage.py](../../run_stage.py), [src/bt5151_credit_risk/state.py](../../src/bt5151_credit_risk/state.py), [src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)): stage runs now carry a shared `run_id` from CLI entry to state, and `package-analysis-bundle` uses that same id when writing `lab/logs/analysis_bundle_<run_id>.json`.
2. **Preprocessing duplicate-column guard** ([src/bt5151_credit_risk/preprocess.py](../../src/bt5151_credit_risk/preprocess.py), [skills/generate-preprocessing-code.md](../../skills/generate-preprocessing-code.md), [skills/repair-preprocessing-code.md](../../skills/repair-preprocessing-code.md)): validation now explicitly fails if pandas name-mangled duplicate columns like `.1` appear in the feature frame, and the worker/repair prompts now call out the two recurring pandas failure modes: `DataFrame.groupby(..., axis=1)` and unsupported `Series.str.get_dummies(prefix=...)`.
3. **Complementary PDP + ALE** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py), [skills/interpret-global-xai.md](../../skills/interpret-global-xai.md)): global XAI now attempts both PDP and ALE on the same top continuous features when feasible, so ALE becomes a second view on feature shape rather than a mutually-exclusive fallback.
4. **Machine-readable confidence diagnostics** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py), [tests/test_graph.py](../../tests/test_graph.py)): `training_diagnostics.confidence_analysis` is now wrapped into a structured object with an LLM `summary` plus programmatic `by_model` confidence stats, including per-class correct/wrong mean confidence. Inference-time caution logic now reads that structure instead of depending on prose.
5. **Inference/runtime cleanup** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py), [src/bt5151_credit_risk/business.py](../../src/bt5151_credit_risk/business.py), [README.md](../../README.md)): nearest-casebook similarity now reads the new SHAP waterfall shape correctly, the dead standalone `recommend_action()` helper was removed, and repo docs/examples now match the merged explain-risk flow.

**Must remain true:**
- Every stage log and its bundle should share one stable run identity.
- Preprocessing must fail loudly on duplicate logical dummy columns instead of silently accepting pandas-mangled names.
- PDP and ALE should be treated as complementary evidence when runtime allows, not as a forced either/or gate.
- Inference-time confidence/caution logic must consume machine-readable stats, not prose.

## 2026-04-15 — Preprocessing representation contract and FE ratio-safety tightened

**Context:** After the first fully coherent overhaul run, the artifact-level deep dive showed the current regression was more consistent with preprocessing damage than FE damage: unbounded heavy-tail numerics survived preprocessing, a structured duration field collapsed to a constant, and FE ratios still used `/(denominator + 1e-6)` patterns that can manufacture million-scale spikes when denominators are legitimately zero.

**Changes:**
1. **Column-transform-spec strengthened** ([skills/column-transform-spec.md](../../skills/column-transform-spec.md)): added explicit guidance that preprocessing should preserve a compact canonical base table, prefer scalar/ordinal representations when they preserve semantics, and avoid brittle structured-string parsing that collapses real variation.
2. **Preprocessing worker/repair prompts aligned** ([skills/generate-preprocessing-code.md](../../skills/generate-preprocessing-code.md), [skills/repair-preprocessing-code.md](../../skills/repair-preprocessing-code.md)): the worker prompt now treats compact scalar/ordinal encodings as intentional contract choices, and the repair prompt now explicitly avoids "fixing" broken compact roles by widening them into one-hot encodings.
3. **FE ratio safety contract updated** ([skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md), [skills/repair-feature-engineering-code.md](../../skills/repair-feature-engineering-code.md)): replaced the old blanket epsilon-denominator rule with zero-aware ratio guidance (`np.where(denom > 0, num / denom, ...)`) so generated features preserve semantic meaning instead of creating artifact spikes.
4. **Regression tests** ([tests/test_skill_prompts.py](../../tests/test_skill_prompts.py)): added prompt-contract coverage for compact base-table semantics and epsilon-free ratio guidance.

**Must remain true:**
- Preprocessing should produce a clean semantic base table, not prematurely explode every categorical into one-hot columns.
- Structured duration parsing should preserve variance instead of collapsing to one fallback value.
- FE ratios must use zero-aware logic rather than epsilon hacks that invent giant values.

## 2026-04-15 — Preprocessing convergence hardening and global-XAI robustness

**Context:** The first fully completed post-overhaul run exposed three residual gaps: preprocessing was still being accepted after repeated **major** audit failures, grouped PFI crashed when dual-view feature frames had fresh CSV indices but targets retained original split indices, and PDP could fail on integer-valued features because sklearn built fractional grids against `int64` columns.

**Changes:**
1. **Preprocessing quality routing hardened** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)): the quality-review escape hatch now accepts only residual **minor** audit issues after repeated attempts. Critical/major issues keep the repair loop active instead of silently flowing into FE/training.
2. **PFI subsampling aligned by position** ([src/bt5151_credit_risk/xai.py](../../src/bt5151_credit_risk/xai.py)): grouped PFI now subsamples `test_frame` / `test_target` with `.iloc` position selection rather than `.loc` label selection, so dual-view frames reloaded from CSV no longer crash when target indices still reflect original split rows.
3. **PDP integer-feature coercion** ([src/bt5151_credit_risk/xai.py](../../src/bt5151_credit_risk/xai.py)): partial dependence now casts integer-valued candidate features to float before grid construction, preventing dtype/grid mismatches on count-like features.
4. **Preprocessing prompt guidance tightened** ([skills/generate-preprocessing-code.md](../../skills/generate-preprocessing-code.md), [skills/repair-preprocessing-code.md](../../skills/repair-preprocessing-code.md)): added stronger general rules for percentile clipping on unbounded heavy-tail numeric columns and for connector-tolerant duration parsing (`"X Years and Y Months"`-style fields).
5. **Regression tests** (`tests/test_xai.py`, `tests/test_graph.py`, `tests/test_skill_prompts.py`): added coverage for position-based PFI subsampling, PDP float coercion for integer features, stricter quality-review routing, and the new preprocessing prompt contract wording.

**Must remain true:**
- The graph should never continue past preprocessing with unresolved critical/major audit issues just because the attempt counter is high.
- Global XAI helpers must tolerate view-frame / target index mismatches introduced by CSV round-trips.
- Integer-valued continuous/count features should not knock PDP out of the method set due to dtype coercion issues alone.
- Preprocessing worker prompts should default to safe percentile clipping and connector-tolerant duration parsing when the spec or profile indicates they are needed.

## 2026-04-14 — Validation-policy plumbing for training / early stopping

**Context:** We aligned on a general rule: reasoning models should choose the validation policy, but leakage-sensitive split execution should be deterministic. The existing training path still used row-level validation by default, which was too brittle for grouped or temporal datasets.

**Changes:**
1. **Dataset policy prompt extended** ([skills/dataset-policy-spec.md](../../skills/dataset-policy-spec.md)): added `validation_policy` as a separate contract from the final holdout split. Supported policy types are `iid_stratified`, `grouped_entity`, and `temporal`.
2. **Training policy engine** ([src/bt5151_credit_risk/train.py](../../src/bt5151_credit_risk/train.py)): added deterministic helpers to normalize the chosen policy, align temporal data, build grouped/temporal/IID validation folds, and create the early-stopping holdout without free-form code generation.
3. **Graph metadata plumbing** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py), [src/bt5151_credit_risk/state.py](../../src/bt5151_credit_risk/state.py)): preprocessing success now carries aligned `train_group_values`, `test_group_values`, `train_time_values`, and `test_time_values` so training can execute the policy selected upstream.
4. **Regression tests** (`tests/test_train.py`, `tests/test_graph.py`, `tests/test_preprocess.py`, `tests/test_state.py`): added coverage for grouped holdout disjointness, temporal alignment before validation, prompt contract passthrough, and graph-level policy plumbing.
5. **Architecture doc updated** ([docs/architecture/current-state.md](../architecture/current-state.md)): training section now explains the `validation_policy` boundary and the policy-aware early-stopping flow.

**Must remain true:**
- `dataset-policy-spec` chooses the policy; `train.py` executes it.
- Inner validation for grouped datasets must keep entity groups disjoint.
- Temporal validation must preserve row order rather than shuffle.
- This policy is separate from the final train/test split and should stay auditable in logs.

## 2026-04-14 — FE dual-view architecture support (backward compatible)

**Context:** We aligned on a staged responsibility shift: preprocessing should trend toward canonical base-table cleanup, while the FE node should eventually own model-facing representation. The safest incremental move was to let the existing FE node emit `linear_view` and `tree_view` without breaking the legacy single-view contract.

**Changes:**
1. **Dual-view FE artifacts supported** ([src/bt5151_credit_risk/feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py)): execution/validation now accepts either legacy single-view artifacts or dual-view artifacts plus `view_metadata.json`.
2. **Graph view routing** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)): added helper routing so training, evaluation, SHAP, PFI, PDP/ALE, local XAI, and inference use the correct feature view per model. Default mapping is `logistic_regression -> linear_view`, `random_forest/xgboost -> tree_view`.
3. **State extensions** ([src/bt5151_credit_risk/state.py](../../src/bt5151_credit_risk/state.py)): added `train_views`, `test_views`, `full_feature_frames_by_view`, `feature_columns_by_view`, and `model_view_map`.
4. **Prompt contract updated** ([skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md), [skills/repair-feature-engineering-code.md](../../skills/repair-feature-engineering-code.md)): FE codegen is now instructed to prefer dual-view output when model families need different representations, while staying compatible with the legacy single-view path.
5. **Regression tests** (`tests/test_feature_engineering.py`, `tests/test_graph.py`, `tests/test_skill_prompts.py`, `tests/test_state.py`): added coverage for dual-view artifact validation, model-specific test/inference view selection, and prompt-contract presence.

**Must remain true:**
- Dual views live inside the existing FE node; no extra graph node is required.
- The pipeline must continue to accept legacy single-view artifacts during transition.
- Training/eval/XAI/inference must all use the same view that the model was trained on.
- `view_metadata.json` is the source of truth when dual views are emitted.

## 2026-04-14 — Semantic role contract for preprocessing

**Context:** Run 013 exposed a preprocessing convergence regression — `Type_of_Loan` multi-hot indicators came out in {0,1,2} instead of {0,1} (count vs presence), the prose audit flagged it weakly, and the repair loop failed three times before the graph accepted the bad output via the 3-attempt escape hatch. The failure class is general: LLM spec output lacked an explicit, machine-checkable statement of column semantics, so the validator had nothing concrete to enforce and repair had nothing concrete to act on.

**Changes:**
1. **Spec schema extended** ([skills/column-transform-spec.md](../../skills/column-transform-spec.md)): every column now declares `semantic_role` (one of 12 enumerated roles) and `representation_intent` (encoding choice when the role admits multiple). Cardinality is a property that drives intent, not a role dimension.
2. **Deterministic validator** (`validate_semantic_roles` in [src/bt5151_credit_risk/preprocess.py](../../src/bt5151_credit_risk/preprocess.py)): runs after preprocessing, checks each column's post-encoding output against the invariant implied by its declared role. Emits structured findings `{column, declared_role, violation, observed, expected, likely_cause}`. Any violation fails the validation report's `passed` flag.
3. **Codegen contract** ([skills/generate-preprocessing-code.md](../../skills/generate-preprocessing-code.md)): new "Semantic role contract" section enumerating the key invariants so the non-reasoning codegen model honors them up front.
4. **Repair rendering** ([skills/repair-preprocessing-code.md](../../skills/repair-preprocessing-code.md)): new principle 6 explicitly treats `role_violations` as deterministic contracts and instructs the model to obey `likely_cause` literally.
5. **Graph logging** ([src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)): `validate_preprocessing_output_node` logs each role violation with its declared role and likely cause.
6. **ADR** ([docs/decisions/0001-semantic-role-contract.md](../decisions/0001-semantic-role-contract.md)).

**Tradeoff:** Two extra fields per column in the spec (minor token cost). If the reasoning model mis-assigns a role, the validator enforces the wrong contract — mitigated by the fact that the validator checks *output vs declared role*, not role correctness itself; semantic mis-assignment surfaces as model-performance regression, not as preprocessing failure.

**Must remain true:**
- Role taxonomy stays closed at 12 roles — cardinality and encoding choice live in `representation_intent`, not in the role list.
- Role assignment is reasoning work (stays on a reasoning-capable model); validator enforcement is deterministic (pure Python, no LLM).
- Repair prompts must render `role_violations` as concrete structured findings with their `likely_cause`, not collapse them into prose.
- Repeated same-role violation across repair attempts is a capability-ceiling signal — escalate model per [AGENT.md](../../AGENT.md), do not retry blindly.

## 2026-04-14 — FE prompt contract tightened to protect raw-value interactions

**Context:** Artifact audit showed the FE node was still generating semantically broken features like `log(1+EMI) / Salary`. The main issue was not encoding choice but an unstable FE prompt contract: the prompt both encouraged skew transforms early and separately said interactions must use raw values first. EDA hypotheses were also framed too strongly as directives.

**Changes:**
1. **FE generate prompt clarified** (`skills/generate-feature-engineering-code.md`): Reframed `eda_hypotheses` from "upstream directives" to "prioritized ideas, not directives." Added explicit required code order: drop redundant features → build interactions from raw parents → only then apply log/monotonic transforms → cleanup.
2. **FE repair prompt aligned** (`skills/repair-feature-engineering-code.md`): Added explicit repair guidance that ratios/products must use raw parent columns before any log transform and that semantic correctness takes priority during repairs.
3. **Prompt regression tests** (`tests/test_skill_prompts.py`): Added tests that lock in the new FE contract language so future prompt edits cannot quietly reintroduce the contradictory "transform first" framing.

**Must remain true:**
- EDA hypotheses should influence FE as candidate ideas, not obligations.
- Semantic interactions must always be created from raw parent values before standalone transforms.
- FE prompt changes should be locked by tests, not just chat memory.

## 2026-04-14 — XAI interpret node + EDA→FE hypothesis chain + SHAP dedup + FE ordering guard

**Context:** Run 010 audit revealed: (1) no LLM interpretation of XAI layers 3-4 (global+local XAI produced numbers but no insights), (2) EDA hypotheses never reached the FE node (generated but unused), (3) SHAP computed 3 times redundantly (select_model inline + global_xai recompute), (4) generated FE code applied log1p before computing interaction ratios (making ratios semantically meaningless).

**Changes:**

1. **interpret-xai-evidence node** (`hypotheses.py`, `graph.py`, `state.py`): New LLM node between local-xai and package-analysis-bundle. Receives global SHAP/PFI, local casebook, training diagnostics, EDA hypotheses, FE hypothesis. Produces observations, insights, feature importance consensus, casebook analysis, cross-layer validation, three-tier hypotheses. Uses `_compact_xai_for_llm()` to strip large arrays before LLM call. Pipeline now 26 nodes.

2. **EDA hypotheses→FE chain** (`feature_engineering.py`, `graph.py`, `skills/generate-feature-engineering-code.md`): `generate_feature_engineering_code()` now accepts `eda_hypotheses` parameter. Graph node passes `state.eda_hypotheses`. FE skill prompt updated: new core principle #2 ("EDA hypotheses are upstream directives, not suggestions") and new input documentation for `eda_hypotheses` (tested_predictions + exploratory_leads). Hypothesis output now includes `eda_hypotheses_acted_on` field for traceability.

3. **FE interaction ordering guard** (`skills/generate-feature-engineering-code.md`): New core principle #4: "Interaction features MUST use raw values, not transformed values." Compute ratios/products BEFORE applying log/skew transforms. Prevents meaningless ratios like `log(1+EMI) / Salary`.

4. **SHAP deduplication** (`graph.py`): select_model_node now uses `compute_global_shap()` from xai.py (was inline ~35 lines). Passes full SHAP result via `state.global_xai_results["shap"]`. global_xai_node reuses it instead of recomputing. Eliminates 2 redundant SHAP computations.

5. **PFI subsampling** (`xai.py`): `compute_permutation_importance()` subsamples test set to 5k rows if larger. Prevents PFI from dominating runtime on large test sets.

6. **interpret-xai-evidence skill prompt** (`skills/interpret-xai-evidence.md`): 5 core principles (observations→insights→hypotheses, cross-method disagreement IS the insight, PDP/ALE shape tells the story, local validates global, chain across layers). Downstream contract for package-analysis-bundle→explain-risk. Domain-neutral worked example.

7. **Analysis bundle updated** (`graph.py:package_analysis_bundle_node`): Bundle includes `xai_interpretation`. Summary includes `xai_observations`, `xai_insights`, `xai_consensus`, `xai_casebook_analysis`, `xai_hypotheses`.

**Must remain true:**
- EDA hypotheses must flow to FE node — the tested_predictions are the highest-priority feature requests
- FE interactions must be computed from raw values before any monotonic transforms
- SHAP from select_model must be reused in global_xai (not recomputed)
- interpret-xai-evidence prompt must teach analytical posture (cross-method disagreement, chain validation), not prescribe specific outputs
- `_compact_xai_for_llm()` must strip beeswarm/raw PFI arrays to fit LLM context

## 2026-04-14 — Training time fix: subsample + reduced trees during tuning

**Context:** RF tuning via Optuna took 2h26m on 66k×60 data (run 009). Root cause: unbounded max_depth trees + 500 n_estimators + 15 trials × 5 folds = 75 fits of fully-grown forests. Previous fix (max_depth=15, n_estimators=200, 10 trials) reduced to ~25 min but still far from expected ~3 min.

**Changes:**
1. **Subsample during tuning** (`train.py`): datasets >15k rows are subsampled to 15k via stratified split before Optuna CV. Hyperparameter rankings are stable at this size. Final retrain still uses full data.
2. **100 trees during RF tuning** (`train.py`): RF uses n_estimators=100 during Optuna search (enough to rank configs), final retrain uses the default 200.
3. **Feature-lineage grouping** (`xai.py`): `_group_onehot_columns` now accepts `column_transform_spec` for authoritative one-hot grouping via encoding metadata — eliminates binary one-hot false-negative from 3+ prefix heuristic.
4. **Prompt restructuring** (`skills/generate-eda-hypotheses.md`, `skills/generate-training-diagnostics.md`): Restructured for o4-mini reasoning model — principles-based prompts with worked examples instead of prescriptive checklists.

**Estimated speedup:** RF tuning from ~25 min → ~1.5-2 min (clean, 4 cores). Combined with prior fixes (max_depth cap, n_estimators removal from grid), total improvement is ~75x vs original 2h26m.

**Must remain true:** Final retrain always uses full training data + full n_estimators. Subsampling is tuning-only. Feature-lineage grouping requires column_transform_spec in state; falls back to 3+ prefix heuristic without it.

## 2026-04-14 — XAI overhaul correctness fixes (5 findings)

**Context:** Code review of the XAI overhaul surfaced 2 high, 2 medium, and 1 low severity issues. All fixed.

**Changes:**

1. **HIGH — True grouped PFI** (`xai.py:compute_permutation_importance`): Was using sklearn's per-column `permutation_importance` then summing by prefix — that's post-hoc aggregation, not true grouped permutation. Fixed: custom implementation that permutes all columns in a one-hot group simultaneously using the same random permutation index, then measures f1_macro drop. Raw (per-column) PFI is still computed via sklearn for comparison.

2. **HIGH — Stale global_shap_importance** (`graph.py:select_model_node`): SHAP was computed for the metric-best model before the LLM selection step. If the LLM selected a different model, downstream nodes (global-xai gating, explain-risk) used SHAP from the wrong model. Fixed: if the LLM selects a non-metric-best model, SHAP is recomputed for the selected model via `compute_global_shap`.

3. **MEDIUM — PDP/ALE method gating mismatch** (`graph.py:global_xai_node`): Correlation evidence came from raw EDA column names but candidate XAI features were post-FE names. Engineered features (e.g., `Income_Debt_Ratio`) would always miss the raw correlation set and get routed to PDP instead of ALE. Fixed: `_is_correlated()` helper checks if any raw correlated column name is a substring of the engineered feature name.

4. **MEDIUM — Analysis bundle overwrite** (`graph.py:package_analysis_bundle_node`): Fixed filename was `analysis_bundle.json`, destroying previous runs. Fixed: filename includes UTC timestamp (`analysis_bundle_20260414T120000Z.json`).

5. **LOW — Worst misclassification selection** (`xai.py:select_classification_cases`): Was maximizing `1.0 - true_class_proba` (lowest true-class confidence), not highest predicted-wrong-class confidence. In multiclass, these differ: a sample can spread low true-class prob across many wrong classes vs one with high confidence in a specific wrong class. Fixed: now maximizes `probas[i, predicted_class]` for wrong predictions.

**Must remain true:**
- Grouped PFI must permute all columns in a group simultaneously — summing per-column PFI is not equivalent
- `global_shap_importance` must always belong to `selected_model_name`, not metric-best
- Method gating must handle engineered feature names that don't match raw EDA column names
- Analysis bundle files must not overwrite across runs

## 2026-04-13 — XAI 4-layer hypothesis-driven overhaul (implementation)

**Context:** Implemented the XAI overhaul plan designed in the earlier deep dive. Pipeline grew from 20 to 25 nodes. Every analytical layer now produces bold three-tier hypotheses. Added 4 new XAI methods beyond SHAP. Local XAI moved from single arbitrary row to systematic casebook.

**New files:**
- `src/bt5151_credit_risk/hypotheses.py` — `generate_eda_hypotheses()` and `generate_training_diagnostics()` LLM wrappers
- `src/bt5151_credit_risk/xai.py` — `compute_global_shap()` (refactored from graph.py), `compute_permutation_importance()` (grouped PFI), `compute_partial_dependence()` (per-class PDP), `compute_ale()` (custom implementation), `compute_shap_contributions_for_case()` (refactored from graph.py), `select_classification_cases()` (casebook strategy)
- `skills/generate-eda-hypotheses.md` — three-tier hypothesis generation from EDA statistics
- `skills/generate-training-diagnostics.md` — per-class struggle, capacity analysis, confidence analysis, hypothesis validation
- `tests/test_xai.py` — PFI grouping, casebook selection

**Modified files:**
- `src/bt5151_credit_risk/graph.py` — 5 new nodes (generate-eda-hypotheses, training-diagnostics, global-xai, local-xai, package-analysis-bundle), rewired edges, removed old `_compute_shap_contributions`, `run_inference_node` uses `xai.compute_shap_contributions_for_case`, baseline CV n_jobs=-1
- `src/bt5151_credit_risk/state.py` — 6 new fields: eda_hypotheses, training_diagnostics, global_xai_results, local_xai_cases, analysis_bundle, analysis_bundle_summary
- `src/bt5151_credit_risk/train.py` — RF n_jobs=-1 (was defaulting to 1, causing 2h26m training)
- `src/bt5151_credit_risk/business.py` — `explain_risk()` now accepts `analysis_bundle_summary` instead of separate eda_top_features/fe_hypothesis params; uses skill prompt via `load_skill_prompt()`
- `skills/explain-risk.md` — restructured for three-tier hypothesis validation output, receives compact bundle summary
- `tests/test_graph.py` — 5 new nodes in expected set, monkeypatches for all new functions, assertions for new state fields
- `tests/test_state.py` — assertions for 6 new state fields

**Key design decisions:**
1. **Method gating**: SHAP + grouped PFI always; ALE when EDA shows |r|>0.5 correlation among top features; PDP when features are uncorrelated. Never hard-code "always compute everything."
2. **Analysis bundle**: Persisted JSON artifact with stable schema. Package node builds compact summary for explain-risk (no PDP grids or beeswarm arrays in the summary).
3. **Casebook strategy**: Classification-specific — representative (most confident correct), worst misclassification (most confident wrong), borderline (least confident correct) per class. Up to 9 cases.
4. **ALE is custom**: ~50 lines, no alibi dependency. Quantile binning → finite differences → accumulate + centre.
5. **PFI grouping**: One-hot columns grouped by shared prefix (rsplit on last underscore). Ungrouped PFI on one-hot is misleading.
6. **Regression→classification adaptation**: No residual plots (confidence analysis instead), no under/over-predicted (representative/misclassification/borderline instead), PDP/ALE return per-class probability curves (not single curve).

**Tradeoff:** 5 more nodes add ~30-60s of compute (PFI, ALE, PDP are CPU-bound) and 2 more LLM calls (EDA hypotheses, training diagnostics). Worth it for the analytical depth and hypothesis chain.

**Must remain true:**
- Bold hypotheses must be grounded in observable data, labeled by tier
- PFI must group one-hot columns
- Method gating: SHAP + grouped PFI always; ALE/PDP conditional
- Local XAI uses casebook strategy, not arbitrary row selection
- Analysis bundle is a persisted artifact with stable schema
- explain-risk receives compact summary, not raw PDP/ALE grids
- Global SHAP in select_model_node stays (model selection needs it before global-xai runs)

## 2026-04-13 — XAI hypothesis-driven deep dive design

**Context:** Reviewed Carson's XAI case study (Melbourne housing notebook) against our pipeline. Found major gaps across all 4 layers: EDA (descriptive only, no forward hypotheses), training diagnostics (metrics only, no residual/per-class analysis), global XAI (SHAP only, no PFI/PDP/ALE), local XAI (single row, no case selection strategy). Agreed on a philosophical shift: bold exploratory hypotheses that don't all need closed-loop validation.

**Key decisions:**

1. **Three-tier hypothesis framework**: Every analytical node produces tested hypotheses (closed loop), supported conjectures (partially testable), and exploratory leads (open threads for future work). Exploratory leads are not discarded just because we can't validate them now.
2. **EDA → forward predictions**: EDA node will generate directional predictions about model performance, class-specific struggles, and feature behavior — not just statistics.
3. **Global XAI method selection by task**: PFI (grouped) + SHAP beeswarm + SHAP dependence always. ALE where EDA shows correlated features. PDP only as ALE contrast. LIME excluded (weak for one-hot tabular). ICE only for meaningful subgroups.
4. **Local XAI case selection**: Per-class representative + worst misclassification per class, not single arbitrary row.
5. **Target encoding as testable hypothesis**: One-hot encoding fragments SHAP/PFI and creates impossible perturbation combinations. Target encoding may consolidate signals and improve LR.

**Full analysis:** `lab/analysis/xai-hypothesis-driven-deep-dive.md`

**Must remain true:**
- Bold hypotheses must be grounded in observable data (not fabricated)
- Hypotheses must be labeled by tier (tested / supported / exploratory)
- Method selection must be justified by task characteristics, not convention

## 2026-04-10 — Repair node → o4-mini reasoning model

**Context:** Kept adding specific code patterns to the repair prompt (str.get_dummies strip, str.extract intermediate vars, etc.) but gpt-4o ignored them or half-followed them across 3+ repair attempts. Same pattern as column-transform-spec: the repair task is analytical reasoning (diagnose bug from audit feedback → trace root cause → fix), not instruction-following.

**Changes:**

1. **Repair model → o4-mini** (`.env`, `.env.example`): Repair-preprocessing-code now uses o4-mini reasoning model.
2. **Repair prompt restructured** (`skills/repair-preprocessing-code.md`): From wall-of-patterns command style to principles-based reasoning: (1) diagnose root causes not symptoms, (2) spec is source of truth, (3) row count is sacred, (4) audit feedback is structured — read it, (5) fix everything in one pass. Technical patterns kept as "Reference patterns" section, not mandatory steps.

**Tradeoff:** o4-mini repair will cost ~3× more output tokens and ~2× more time per call. But if it fixes things in 1 attempt instead of 3, net token and time savings are substantial (saves 2 repair + 2 audit rounds).

**Must remain true:**
- Repair prompt should have principles for reasoning, not just code templates
- Technical patterns are reference material, not commands

## 2026-04-10 — str.get_dummies whitespace strip, auditor convergence, FE reasoning model

**Context:** 17:43 run hit the 3-attempt escape hatch. Root causes: (1) `str.get_dummies(sep=',')` doesn't strip whitespace from tokens, creating duplicate columns like `" Home Loan"` and `"Home Loan"`; (2) quality auditor refused to converge on follow-up — kept re-flagging Annual_Income after repair addressed it. Separately, FE hypothesis from gpt-4o was vague; switched to o4-mini.

**Changes:**

1. **Codegen + repair prompts** (`skills/generate-preprocessing-code.md`, `skills/repair-preprocessing-code.md`): Expanded `str.get_dummies` example to 5-line pattern: strip column names, `groupby(level=0, axis=1).max()` to merge duplicates, drop empty-name columns. Added "CRITICAL" comment explaining `str.get_dummies` doesn't strip.

2. **Audit prompt** (`skills/audit-preprocessing.md`): Strengthened follow-up convergence rules — partial improvement counts as progress, do not re-flag values that were reasonably addressed, explicit "do not move the goalposts."

3. **FE prompt restructured** (`skills/generate-feature-engineering-code.md`): Rewrote from command-style (7 heuristic rules) to hybrid reasoning-first approach — 5 core principles, reasoning phase before code, then mandatory technical guardrails and recommended transforms. Same pattern as column-transform-spec.

4. **FE model → o4-mini** (`.env`): Added `OPENAI_MODEL_GENERATE_FEATURE_ENGINEERING_CODE=o4-mini`. Validated: hypothesis now cites MI values and creates domain-grounded interactions.

5. **Target alignment check** (`skills/audit-preprocessing.md`): Changed to exact string match to prevent false positives from similar-prefix columns.

**Must remain true:**
- `str.get_dummies` patterns must always include strip + dedup + empty-column cleanup
- Auditor follow-up reviews must converge — partial improvements pass
- FE prompt must have reasoning phase before code phase when paired with reasoning model

## 2026-04-10 — Preprocessing codegen/repair prompt fixes (Type_of_Loan + Credit_History_Age)

**Context:** Preprocess stage (16:13 run) exhausted all 5 repair attempts. Two root causes: (1) Type_of_Loan multi-value encoding — codegen used `pd.get_dummies` on raw strings (6,310 columns) or repair used `explode` (442k rows vs 100k groups); (2) Credit_History_Age parsing — `str.extract` with 2 groups returns DataFrame, codegen called `.median()` on the original string column.

**Changes:**

1. **Repair prompt** (`skills/repair-preprocessing-code.md`): Added explicit `str.get_dummies(sep=...)` pattern with "NEVER use `explode`" warning explaining why (changes row count, breaks group-based splits). Added concrete Credit_History_Age "Years and Months → total months" code example.

2. **Codegen prompt** (`skills/generate-preprocessing-code.md`): Strengthened Step 6 multi-value guidance with concrete code example (`str.replace` + `str.get_dummies` + `pd.concat`). Added "NEVER use `explode`" to both Step 6 and Common gotchas. Added concrete `str.extract` multi-group code example for Years/Months parsing.

**Result:** 16:42 run converged in 3/5 attempts. Repair model used `str.get_dummies` (not `explode`) and correct `str.extract` with intermediate variables — both matching the new prompt examples exactly.

**Must remain true:**
- Codegen and repair prompts must both have explicit `str.get_dummies` pattern — repair prompt was the one missing it
- Multi-value column guidance must warn against both failure modes: `explode` (row count) and raw `pd.get_dummies` (cardinality)

## 2026-04-10 — Reasoning model validation, bug fixes, prompt restructure

**Context:** Specs stage testing revealed column-transform-spec variance (gpt-4o produced inconsistent clipping bounds and encoding choices across runs). Code review surfaced test set leakage in baseline metrics and unreachable quality review escape hatch.

**Changes:**

1. **o4-mini for column-transform-spec** (`.env`, `skills/column-transform-spec.md`): Validated o4-mini produces stable, domain-plausible decisions. Prompt restructured from hard constraints to principles-based approach — explains *why* (codegen contract, semantic encoding, percentile validity) instead of commanding what not to do. o4-mini internalizes principles and applies them; gpt-4o needed explicit rules.

2. **ANOVA class separability forwarded to column-transform-spec** (`preprocess.py`): EDA computed ANOVA F-stats but didn't pass them downstream. Now included in `eda_insights.class_separability`.

3. **Test set leakage fix** (`graph.py:train_models_node`): Baseline metrics were computed on `state.test_frame` and fed to `reason_hyperparameter_grids()`. Replaced with cross-validated baseline on training data only (`StratifiedKFold`, scoring=f1_macro).

4. **Unreachable escape hatch fix** (`graph.py`): Quality review node now preserves `structural_passed` before overwriting merged `passed` flag. Routing function reads `structural_passed` to correctly trigger the 3-attempt escape hatch.

5. **CV baseline robustness** (`graph.py:train_models_node`): Skip baseline CV when rarest class has < 2 samples (degenerate dataset guard).

6. **`feature_engineering_runs/` added to .gitignore**.

**Must remain true:**
- Baseline metrics must never touch test_frame — only cross-validation on train data
- `structural_passed` must be preserved before quality merge overwrites `passed`
- column-transform-spec prompt must use principles (not commands) when paired with reasoning model

## 2026-04-10 — EDA node, reasoning chain, Optuna, early stopping, learning curves

**Context:** Run 008 had mediocre metrics (RF macro_f1=0.68), no EDA, no reasoning trail, inefficient tuning (RandomizedSearchCV, 111 min for RF), and no early stopping for XGBoost.

**Changes:**

1. **EDA node** (`eda.py`, `graph.py`, `state.py`): New `exploratory-data-analysis` node between dataset-policy-spec and column-transform-spec. Programmatic (no LLM): correlation matrix (|r|>0.8 pairs), ANOVA F-stat + class-conditional means, mutual information ranking, skewness, missing patterns (MNAR detection), cardinality. Report stored in `eda_report` state field.

2. **Optuna Bayesian optimization** (`train.py`, `skills/reason-hyperparameter-grid.md`): Replaced RandomizedSearchCV with Optuna TPESampler (15 trials, 5-fold stratified CV). Grid format changed from list-based to range-based (type/low/high/step/log). XGBoost n_estimators removed from grid — early stopping handles it.

3. **XGBoost early stopping + learning curves** (`train.py`, `graph.py`): CV folds use n_estimators=1000 + early_stopping_rounds=50 with eval_set. Final refit uses 90/10 held-out split for early stopping. Learning curves extracted from `evals_result()` and stored in `learning_curves` state field.

4. **Reasoning chain** — traceable hypothesis from EDA to explanation:
   - **column-transform-spec** (`skills/column-transform-spec.md`, `preprocess.py`): Receives EDA insights (discriminative features, correlation pairs, skewness, cardinality, MNAR). Returns per-column `reasoning` dict.
   - **FE hypothesis** (`skills/generate-feature-engineering-code.md`, `skills/repair-feature-engineering-code.md`, `feature_engineering.py`, `graph.py`): Receives EDA insights. Returns `hypothesis` (interactions_rationale, dropped_features_rationale, expected_impact). Stored in `feature_engineering_hypothesis` state field.
   - **Model selection** (`skills/reason-model-selection.md`, `evaluate.py`, `graph.py`): New `reason_model_selection()` LLM function receives evaluation results, tuning results, SHAP importance, EDA top features, FE hypothesis. Returns justification + hypothesis_validation. Falls back to metric-based selection on failure.
   - **Explain-risk** (`business.py`, `graph.py`): Receives full hypothesis chain (EDA top features, FE hypothesis, global SHAP, selection justification). LLM notes which hypotheses were confirmed/refuted.

5. **Reasoning model config** (`.env.example`): Recommended per-node model assignments — o4-mini for analytical/reasoning nodes, gpt-4o for codegen, gpt-4o-mini for simple text.

**Must remain true:**
- EDA is programmatic (no LLM) — must not add latency from API calls
- Optuna grids must use range-based format (type/low/high), not list-based
- XGBoost early stopping must be in both CV folds and final refit
- LR grids must only include model__C — Optuna _suggest_params strips other LR params as safety net
- reason_model_selection must fall back to choose_best_model on any failure
- FE hypothesis must survive repair cycles (repair skill includes hypothesis in output)

## 2026-04-10 — SHAP XAI, hyperparameter tuning, FE improvements, bug fixes

**Context:** Pipeline needed explainability (SHAP), hyperparameter tuning, stronger feature engineering, and several runtime bug fixes.

**Changes:**

1. **SHAP explainability** (`graph.py`, `business.py`, `state.py`): Added `_compute_shap_contributions()` — TreeExplainer for RF/XGBoost, LinearExplainer for LR Pipeline. Per-prediction top-5 SHAP features ground the explain-risk LLM call. Global SHAP importance (mean |SHAP| on 500 test samples) computed at model selection.

2. **Hyperparameter tuning** (`train.py`, `graph.py`, `skills/reason-hyperparameter-grid.md`): LLM reasons search grids per model given dataset characteristics. `RandomizedSearchCV` (15 iterations, 5-fold stratified CV, f1_macro). Train node: baseline fit → LLM-reasoned grids → tune with fresh models. LR grids filtered to strip l1_ratio/penalty/solver (lbfgs only supports l2).

3. **FE skill prompt improvements** (`skills/generate-feature-engineering-code.md`): Mandatory rules: min 3 interaction features, mandatory inf cleanup, safe binning with extended edges, final NaN cleanup. Max 8 interactions (was 5).

4. **Subprocess bytes fix** (`preprocess.py`, `feature_engineering.py`): `TimeoutExpired.stderr` can be bytes even with `text=True`. Decode to str before JSON serialization.

5. **Quality review tolerance** (`graph.py:_route_after_quality_review`): Accept preprocessing after 3 attempts if structural validation passes but quality review keeps flagging real data characteristics.

6. **Preprocessing timeout** (`preprocess.py`): Increased from 60s to 120s — 100k rows with encoding needs more time.

**Must remain true:**
- SHAP explainers must match model type (Tree for RF/XGB, Linear for LR Pipeline with scaled input)
- LR grids must never include l1_ratio/penalty/solver — lbfgs crashes with elasticnet
- Subprocess stderr must be decoded to str before passing to repair LLM
- Quality review should not exhaust all repair attempts on unfixable data characteristics

## 2026-04-10 — Model evaluation + data quality improvements

**Context:** XGBoost lacked class imbalance handling, evaluation had no confusion matrix, and the column-transform-spec had insufficient data visibility (saw only 5 sample rows, missed garbage values and outliers).

**Changes:**

1. **XGBoost class imbalance** (`graph.py:train_models_node`): Added `compute_sample_weight('balanced', train_target)` passed to XGBoost `.fit()`. LR and RF already use `class_weight='balanced'`. Result: XGBoost macro_f1 +1.3pp, minority class recall +9-24pp.

2. **Confusion matrix + per-class metrics** (`evaluate.py`, `graph.py:evaluate_models_node`): Added `confusion_matrix` to `compute_multiclass_metrics()` output. Log matrix and per-class precision/recall/f1/support for each model.

3. **Enriched column-transform-spec payload** (`preprocess.py`): Added `_build_column_profiles()` — top-10 values per categorical (catches garbage like `________`, `!@9#%8`) and min/max/mean/p1/p99 per numeric (catches outliers like Age=8698, Interest_Rate=5797). Sample rows 5→10. Result: first audit finds 3 issues vs 8-11 previously, preprocessing converges in 3 attempts vs 4+.

4. **FE inf validation** (`feature_engineering.py`): Added `no_infs_in_train`/`no_infs_in_test` checks. Log-transform of columns with zeros produced inf that crashed training (StandardScaler ValueError).

5. **Skill prompt gotchas**: Added `freq='M'`→`'ME'` (pandas 3.x), `mode()[0]` empty series guard, mandatory inf cleanup rule for FE.

**Must remain true:**
- XGBoost must receive `sample_weight` in `.fit()` — it cannot use `class_weight` parameter
- FE validation must check both NaN and inf before passing to training
- Column profiles must include top-10 values for categoricals (garbage detection depends on this)

## 2026-04-07 — Code audit fixes (feature/free-codegen-preprocessing-loop)

**Context:** Full audit of the free-codegen preprocessing refactor surfaced 10 original issues and 6 new codegen-specific issues.

**Changes:**

1. **Sort class_names** (`graph.py:100`): `unique()` → `sorted(...)`. Without this, `label_to_id` could disagree with sklearn's `.classes_` order, causing silent misclassification.

2. **Generalize profile.py**: Replaced hardcoded `"Credit_Score"` with `config.TARGET_COLUMN` parameter. Unblocks reuse on other datasets.

3. **Pass raw_frame_path through state** (`state.py`, `graph.py`): Added `preprocessing_raw_frame_path` to state. Execution node stores the actual path; validation node reads it from state. Previously reconstructed from workspace path, which was the same implicit dependency disguised as an explicit one.

4. **Expand code inspector blocklist** (`preprocess.py`): Added `eval`, `exec`, `__import__`, `compile`, `breakpoint`, `os.popen`, `os.exec*`, `os.spawn*`, `importlib.import_module`, `socket`, `urllib`, `http`, `ftplib`, `smtplib`, `ctypes`, `multiprocessing`. Added 7 parametrized tests.

5. **Workspace cleanup after validation, not before execution** (`preprocess.py`, `graph.py`): `cleanup_old_workspaces()` now runs in the validation node after validation passes, not in `execute_generated_preprocessing`. During repair loops, earlier workspaces are preserved until a successful validation, so repair payloads and artifact checks remain valid.

6. **Gitignore updates**: Added `train.csv` and `generated_preprocessing_runs/` to prevent repo bloat.

7. **LLM JSON retry** (`llm.py`): Up to 3 attempts on `json.JSONDecodeError` before raising.

8. **Repair loop off-by-one** (`graph.py`): Initial generation now sets `preprocessing_attempt_count: 1` so `MAX_REPAIR_ATTEMPTS=3` means 3 total attempts, not 4.

9. **Pin dependency versions** (`requirements.txt`): All 14 dependencies pinned to exact versions.

**Must remain true:**
- `class_names` must always be sorted — sklearn depends on this for `predict_proba` alignment
- Inspector must block any pattern that could escape the subprocess sandbox
- Workspace cleanup must preserve the latest workspace (validation reads artifacts from it)
- LLM retry must not swallow non-JSON errors

## 2026-04-15 — XAI audit follow-ups (PFI dtype, bundle verbatim, split interpret nodes, explain-risk expansion)

**Context:** Audit of run `stage_full_20260415_030032.log` identified: (1) PFI fails with "object dtype" error despite FE artifacts on disk being numeric, (2) `package-analysis-bundle` ships a programmatic compression (`[:5]/[:3]`) to explain-risk that throws away the exact "last 1% insight" callers care about, (3) `interpret-xai-evidence` receives a compacted payload in `hypotheses.py:29` and mixes global+local reasoning in one LLM call, weakening per-case analysis, (4) `explain-risk` skill is only 64 lines, its schema no longer matches what `graph.py` builds, and it only receives the compacted summary, (5) Credit_History_Age parser silently collapses months to 0 due to greedy `.*` regex, (6) Occupation frequency-encoded at preprocess erases category identity.

**Changes:**

1. **PFI bool→int8 cast** (`xai.py:182`): Tree-view test frames contain bool multi-hot columns mixed with float/int. `.values` / `np.asarray` on such a frame promotes to `object` dtype, which XGBoost's `predict` rejects. Cast bool columns to int8 before calling `permutation_importance`. Root cause was a NumPy coercion at the PFI boundary, not bad FE output.

2. **Split `interpret-xai-evidence` into two LLM nodes** (`hypotheses.py`, `graph.py`, `state.py`, new skills `interpret-global-xai.md` / `interpret-local-xai.md`): Global-XAI interpretation handles SHAP/PFI/PDP/ALE cross-method consensus and feature-effect shapes; Local-XAI interpretation handles per-class casebook stories, confusion patterns, and decision-boundary analysis. Second node receives the first node's output as context so it can cite global findings without re-deriving them. Removed `_compact_xai_for_llm` — only raw beeswarm arrays are stripped; everything else passes verbatim.

3. **Package-analysis-bundle: verbatim pass-through** (`graph.py:1103`): Removed the programmatic compression (`observations[:5]`, `insights[:3]`, `pfi_top5`, `{case_type,true,pred}` reduction). `analysis_bundle_summary` now IS the full semantic bundle — explain-risk is the consumer that decides what to surface to the customer. A separate `numeric_artifacts` block holds the raw SHAP importance / PDP grids and is saved to disk but kept out of the LLM payload.

4. **Explain-risk prompt expansion + schema alignment** (`skills/explain-risk.md`): Rewrote from 64 lines to a full contract — explicit three-tier hypothesis validation (with `tier` and `layer` tags per claim), `local_context` section (case profile, boundary proximity, counterfactual), `key_drivers` now carries `raw_value` from `source_record` and `global_rank_context`, `model_context` covers performance / class struggle / confidence reliability. Schema matches the fields the bundle actually carries (`global_xai_interpretation`, `local_xai_interpretation`, `local_casebook`, `feature_engineering_hypothesis`, `selection_justification`).

5. **Credit_History_Age parser hardening** (`skills/generate-preprocessing-code.md`): Added explicit gotcha about greedy `.*` in duration regex. The symptom — parsed column whose unique values are all multiples of 12 — is now a stated failure mode for the codegen LLM to avoid.

6. **Occupation identity-preservation guidance** (`skills/column-transform-spec.md`): Strengthened `unordered_categorical` default section — identity-significant categoricals (occupation, industry, product-type) should almost always be `deferred`; frequency-encoding collapses categories with similar row counts into indistinguishable values. With `OPENAI_MODEL_COLUMN_TRANSFORM_SPEC=o3` now active in `.env`, next run should take this guidance.

7. **Per-node env overrides** (`.env`, `.env.example`): Replaced `OPENAI_MODEL_INTERPRET_XAI_EVIDENCE` with separate `OPENAI_MODEL_INTERPRET_GLOBAL_XAI` and `OPENAI_MODEL_INTERPRET_LOCAL_XAI` hooks. Explain-risk stays on gpt-4o for now — evaluate whether the expanded prompt is sufficient before escalating to o3.

**Tests:** All 87 pass. `test_graph.py` updated with `fake_interpret_global_xai` + `fake_interpret_local_xai` fakes and new node names in the expected set. `test_state.py` updated for the two new state fields.

**Must remain true:**
- PFI must cast bool→int8 before any NumPy-layer promotion
- `analysis_bundle_summary` must be the full semantic bundle, not a programmatic compression — truncation decisions belong to explain-risk
- Global XAI interpretation and local XAI interpretation are two distinct LLM calls; local receives global's output for reference, not a combined payload
- Explain-risk schema must track the bundle schema — if a new interpretation field is added to the bundle, explain-risk's skill prompt must be updated in the same change
- Credit_History_Age parse must yield non-multiple-of-12 unique values when raw data has month-level granularity

## 2026-04-15 — FE bool contract fix, preprocessing timeout bump, and run-artifact tracking

**Context:** The first `o3`-era rerun (`stage_full_20260415_121006.log`) exposed a harness bug rather than a true FE semantic failure: repaired FE code emitted valid dual-view CSVs with boolean dummy columns, but runtime validation and the FE prompt were still using `select_dtypes(exclude='number')`, which rejects pandas `bool`. The same run also showed the first preprocessing attempt timing out under the old 120s subprocess budget.

**Changes:**

1. **Allow bool dummy columns in FE validation** (`feature_engineering.py`): The runtime contract now treats `bool` as a valid model-ready dtype alongside numeric dtypes. Validation still fails loudly on any remaining string/category/object columns.

2. **Align FE prompt wording with runtime** (`skills/generate-feature-engineering-code.md`): The skill now instructs generated code to assert `select_dtypes(exclude=['number', 'bool']).empty` and explicitly states that boolean one-hot / multi-hot columns are valid outputs.

3. **Raise preprocessing subprocess timeout to 180s** (`preprocess.py`): Gives first-pass generated preprocessing code more room before repair, while keeping a bounded timeout and the same isolated execution model.

4. **Track pipeline run artifacts in git** (`.gitignore`): Removed the ignore rules for `logs/` and `analysis_bundle_*.json` so run evidence can be committed when desired.

**Tests:** Added RED tests first for bool-dummy FE acceptance and the new preprocessing timeout, then updated prompt assertions. Full suite remains the source of truth after the implementation pass.

**Must remain true:**
- FE validation must accept `bool` but reject raw string/category/object columns
- FE prompt examples and runtime validator must use the same dtype contract
- Preprocessing timeout changes are a guardrail tuning knob, not a substitute for fixing slow generated code
- Logs and analysis bundles are now versionable artifacts; if they create repo bloat later, use a more selective policy instead of reintroducing blanket ignores

## 2026-04-16 — Deterministic selection and structured Developer Trace artifacts

**Context:** We wanted to borrow the right lessons from `plexe-main` without importing its heavier search/journal architecture. The two highest-value gaps were still open: the final selected model could still be nudged by the LLM, and the Gradio Developer Trace tab was still parsing prose logs instead of binding to a machine-readable run artifact.

**Changes:**

1. **Deterministic final model selection** (`evaluate.py`, `graph.py`, `tests/test_evaluate.py`, `tests/test_graph.py`): `choose_best_model()` is now the sole authority for `selected_model_name`, using held-out `macro_f1` and `weighted_f1`. The `reason-model-selection` skill is still called, but only to explain the metric winner and attach hypothesis-validation context. Any LLM-proposed alternate `model_name` is logged as advisory and ignored.

2. **Structured trace-event artifact** (`trace_events.py`, `run_stage.py`, `run_status.py`, `cache.py`, `state.py`, `tests/test_run_status.py`, `tests/test_trace_events.py`): Added `trace_events_<run_id>.jsonl` as a first-class run artifact. `run_stage.py` now emits run lifecycle events plus node-complete events, `active_run.json` now records `trace_path`, and cache provenance now preserves `cache_trace_path` alongside the existing log and bundle paths.

3. **Developer Trace routing prefers machine-readable provenance** (`ui_trace.py`, `app.py`, `tests/test_ui_trace.py`, `tests/test_app.py`): The Gradio Developer Trace tab now prefers `active_run.trace_path` for live/failed/completed active runs, then `state.cache_trace_path` for cached runs, and only falls back to raw stage logs when no trace artifact exists. The UI parser now renders both structured trace JSONL and raw logs into the same card-based Markdown view.

4. **Failure-path hardening for trace/provenance** (`run_status.py`, `run_stage.py`): `mark_run_completed()` / `mark_run_failed()` now guard against stale `run_id` writes, and successful pipeline runs no longer remain stuck in `running` if optional post-run persistence steps like cache save or `cache_saved` trace append fail.

**Verification:** Targeted slices passed during implementation:
- `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_evaluate.py tests/test_graph.py -q`
- `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_run_status.py tests/test_trace_events.py -q`
- `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_ui_trace.py tests/test_app.py -q`

**Must remain true:**
- The LLM may justify the winner, but it must never override `selected_model_name`
- `trace_events_<run_id>.jsonl` must remain bound to the same `run_id` as the stage log, analysis bundle, and cached model provenance
- `active_run.json` remains single-writer state owned by `run_stage.py`
- Developer Trace must prefer structured trace artifacts first and use raw logs only as fallback evidence

## 2026-04-17 — Model Evidence toolbar cleanup and sticky Developer Trace sidebar

**Context:** After the Tab 2 rendering stabilization, the remaining UX rough edges were layout-related: the evidence refresh button was unnecessarily full-width, and Developer Trace needed a classic sticky-left/sidebar + normal-page-scroll behavior instead of feeling like a nested panel.

**Changes:**

1. **Right-aligned Evidence toolbar** (`app.py`, `tests/test_app.py`): Reworked the top of Tab 2 into a small toolbar row with the explanatory copy on the left and a narrow `Load / Refresh Evidence` button on the right. Added stable `elem_id` hooks so the layout is intentional rather than relying on implicit Gradio sizing.

2. **Sticky left pipeline panel in Developer Trace** (`app.py`, `tests/test_app.py`): Added `elem_id` hooks for the trace layout and applied CSS so the left pipeline column stays visible with `position: sticky` while the right log uses the normal document scroll. This preserves the requested “classic sticky sidebar” UX and avoids introducing an inner scroll region.

**Verification:** `PYTHONPATH=src .venv/bin/python3 -m pytest tests/test_app.py -q`

**Must remain true:**
- The evidence refresh control stays narrow and right-aligned on desktop layouts
- Developer Trace uses normal page scroll; the log should not become an inner scroll container
- The sticky pipeline behavior relies on explicit `elem_id` hooks, so future layout work should preserve those ids or update the CSS/tests together
