# Spec-Driven Preprocessing Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the preprocessing block into a spec-driven pipeline that uses two LLM planning calls for dataset-level and column-level decisions, followed by deterministic execution and audit, while keeping training and evaluation deterministic.

**Architecture:** The current single `preprocess-data` node will be decomposed into a preprocessing block with explicit stages: dataset policy generation, column transform generation, preprocessing execution, and preprocessing audit. LLM calls will generate structured specs, while deterministic Python will apply those specs, run grouped splitting, and verify leakage or schema safety before training begins.

**Tech Stack:** Python 3.12, pandas, scikit-learn, langgraph, pydantic, openai, python-dotenv, pytest, jupyter

---

### Task 1: Expand State For Spec-Driven Preprocessing

**Files:**
- Modify: `src/bt5151_credit_risk/state.py`
- Test: `tests/test_state.py`

**Step 1: Write the failing test**

Add expectations for:
- `dataset_policy_spec`
- `column_transform_spec`
- `preprocessing_execution_report`
- `preprocessing_audit_report`

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_state.py -v`
Expected: FAIL because the new state keys do not exist yet

**Step 3: Write minimal implementation**

Add the new fields to `CreditRiskState`.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_state.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/state.py tests/test_state.py
git commit -m "feat: extend state for preprocessing specs"
```

### Task 2: Add Dataset Policy And Column Transform Spec Functions

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add tests for:
- `generate_dataset_policy_spec(...)`
- `generate_column_transform_spec(...)`

Monkeypatch the LLM call so the functions can be tested without a real API call.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: FAIL because the functions do not exist yet

**Step 3: Write minimal implementation**

Add:
- `_call_preprocess_agent(...)`
- `generate_dataset_policy_spec(...)`
- `generate_column_transform_spec(...)`

Each function should return structured dicts from the LLM layer.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: PASS for the new spec-generation tests

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: add llm preprocessing spec generation"
```

### Task 3: Implement Deterministic Preprocessing Execution And Audit

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add tests for:
- dropping identifiers from executed feature output
- grouped split with zero overlap
- target excluded from features
- audit report marking key checks as passed

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: FAIL because execution and audit functions are missing or incomplete

**Step 3: Write minimal implementation**

Add:
- `execute_preprocessing(...)`
- `audit_preprocessing_output(...)`

Execution should:
- replace placeholder tokens
- drop forbidden columns
- apply simple per-column imputation and drop rules from the spec
- one-hot encode categorical features
- create grouped train/test splits

Audit should:
- verify target not in feature columns
- verify no group overlap between train and test
- verify forbidden columns are removed
- verify feature output is non-empty

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_preprocess.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: add deterministic preprocessing execution and audit"
```

### Task 4: Rewire The Graph To The New Preprocessing Block

**Files:**
- Modify: `src/bt5151_credit_risk/graph.py`
- Test: `tests/test_graph.py`

**Step 1: Write the failing test**

Update graph tests to expect these nodes:
- `dataset-policy-spec`
- `column-transform-spec`
- `execute-preprocessing`
- `audit-preprocessing`
- `train-models`
- `evaluate-models`
- `select-model`
- `run-inference`
- `explain-risk`
- `recommend-action`

Also add an end-to-end test that monkeypatches:
- preprocessing LLM spec generation
- downstream business LLM generation

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_graph.py -v`
Expected: FAIL because the graph still uses the old node layout

**Step 3: Write minimal implementation**

Refactor `graph.py` so the preprocessing block becomes:
- `dataset-policy-spec`
- `column-transform-spec`
- `execute-preprocessing`
- `audit-preprocessing`

Store the generated specs and audit results in state.

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/graph.py tests/test_graph.py
git commit -m "feat: refactor graph to spec-driven preprocessing block"
```

### Task 5: Keep Train And Evaluate Deterministic But Improve Selection Context

**Files:**
- Modify: `src/bt5151_credit_risk/evaluate.py`
- Test: `tests/test_evaluate.py`

**Step 1: Write the failing test**

Add a test that ensures model selection output includes enough context for downstream business and report use.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_evaluate.py -v`
Expected: FAIL because the current selection output is too thin

**Step 3: Write minimal implementation**

Keep metric computation deterministic, but expand the selection result to include:
- selected model name
- metric summary
- explicit justification string

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_evaluate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/evaluate.py tests/test_evaluate.py
git commit -m "feat: strengthen deterministic selection output"
```

### Task 6: Rewire Notebook To The New Graph Shape

**Files:**
- Modify: `bt5151_credit_risk_pipeline.ipynb`
- Test: `tests/test_notebook_smoke.py`

**Step 1: Write the failing test**

Update the notebook smoke test to assert the notebook references the new preprocessing nodes or preprocessing-spec workflow.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_notebook_smoke.py -v`
Expected: FAIL because the notebook still reflects the older graph structure

**Step 3: Write minimal implementation**

Update the notebook so it:
- compiles the new graph
- invokes the graph through the redesigned preprocessing block
- keeps the UI on the real graph path only

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_notebook_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bt5151_credit_risk_pipeline.ipynb tests/test_notebook_smoke.py
git commit -m "feat: update notebook for spec-driven preprocessing graph"
```

### Task 7: Update Skills And README To Match The New Structure

**Files:**
- Create: `skills/dataset-policy-spec.md`
- Create: `skills/column-transform-spec.md`
- Create: `skills/execute-preprocessing.md`
- Create: `skills/audit-preprocessing.md`
- Modify: `README.md`

**Step 1: Write the failing test**

If practical, add a lightweight test that the README references the new graph shape. Otherwise rely on manual verification after edits.

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && PYTHONPATH=src pytest tests/test_notebook_smoke.py -v`
Expected: optional, only if README assertion is added

**Step 3: Write minimal implementation**

Add the four new skill docs and update the README architecture and node descriptions so the team sees the new preprocessing block clearly.

**Step 4: Run verification**

Run:

```bash
sed -n '1,240p' README.md
find skills -maxdepth 1 -type f | sort
```

Expected: the new skill files exist and README reflects the redesigned node chain

**Step 5: Commit**

```bash
git add skills README.md
git commit -m "docs: align skills and readme with spec-driven preprocessing"
```

### Task 8: Full Verification

**Files:**
- No new files required unless fixes are needed

**Step 1: Run full test suite**

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -v
```

Expected: PASS

**Step 2: Run git status**

```bash
git status --short --branch
```

Expected: clean branch except for known untracked local data files

**Step 3: Commit any final fixes**

```bash
git add <files>
git commit -m "test: verify spec-driven preprocessing refactor"
```
