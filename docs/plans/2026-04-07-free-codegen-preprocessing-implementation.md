# Free-Codegen Preprocessing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current spec-driven deterministic preprocessing execution with a free-code generation preprocessing loop that uses `SKILL.md` as runtime system prompt policy, executes generated code in an isolated workspace, validates the artifacts, and repairs failed generations before training.

**Architecture:** Keep `dataset-policy-spec` and `column-transform-spec` as LLM reasoning nodes, and keep `train-models` and `evaluate-models` deterministic. Replace the current `execute-preprocessing` and `audit-preprocessing` nodes with a code-generation loop: `generate-preprocessing-code -> inspect-preprocessing-code -> execute-generated-preprocessing -> validate-preprocessing-output -> repair-preprocessing-code`. Each LLM-driven node must load its corresponding `skills/*.md` file at runtime and inject relevant state into the prompt as task context.

**Tech Stack:** Python 3.12, pandas, scikit-learn, langgraph, pydantic, openai, python-dotenv, pytest, jupyter

---

### Task 1: Extend State For Generated-Code Preprocessing

**Files:**
- Modify: `src/bt5151_credit_risk/state.py`
- Test: `tests/test_state.py`

**Step 1: Write the failing test**

Extend `tests/test_state.py` to expect:

- `preprocessing_code`
- `preprocessing_codegen_metadata`
- `preprocessing_code_review`
- `preprocessing_workspace`
- `preprocessing_artifacts`
- `preprocessing_execution_log`
- `preprocessing_validation_report`
- `preprocessing_attempt_count`

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_state.py -v
```

Expected: FAIL because the new state fields do not exist yet.

**Step 3: Write minimal implementation**

Add the new fields to `CreditRiskState`.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_state.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/state.py tests/test_state.py
git commit -m "feat: extend state for preprocessing codegen loop"
```

### Task 2: Add Runtime Skill Prompt Loading

**Files:**
- Create: `src/bt5151_credit_risk/skill_prompts.py`
- Create: `tests/test_skill_prompts.py`

**Step 1: Write the failing test**

Create tests for:

- loading a skill prompt from `skills/<name>.md`
- raising a clear error when the file does not exist

Example:

```python
from bt5151_credit_risk.skill_prompts import load_skill_prompt


def test_load_skill_prompt_reads_skill_file():
    prompt = load_skill_prompt("train-models")
    assert "train-models" in prompt
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_skill_prompts.py -v
```

Expected: FAIL because the loader does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `load_skill_prompt(skill_name: str) -> str`

This function should read `skills/<skill_name>.md`.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_skill_prompts.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/skill_prompts.py tests/test_skill_prompts.py
git commit -m "feat: load skill prompts at runtime"
```

### Task 3: Implement Preprocessing Code Generation

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add a test for:

- `generate_preprocessing_code(...)`

The test should monkeypatch the LLM call and assert that:

- the generated code string is returned
- the skill prompt is loaded
- `dataset_policy_spec` and `column_transform_spec` are passed as runtime context

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL because `generate_preprocessing_code` does not exist.

**Step 3: Write minimal implementation**

Implement:

- `_call_preprocess_codegen_agent(...)`
- `generate_preprocessing_code(raw_df, dataset_profile, dataset_policy_spec, column_transform_spec) -> dict`

Return a dict like:

```python
{
    "code": "...python code...",
    "entrypoint": "run_preprocessing",
}
```

Use `skills/generate-preprocessing-code.md` as system prompt policy.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS for the new code-generation test.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: add preprocessing code generation"
```

### Task 4: Add Static Inspection For Generated Code

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add tests for:

- rejecting code that imports `subprocess`
- rejecting code that uses `os.system`
- rejecting code that omits the required entrypoint
- accepting safe code that defines the entrypoint

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL because the inspection function does not exist.

**Step 3: Write minimal implementation**

Implement:

- `inspect_preprocessing_code(generated_code: dict) -> dict`

This should:

- parse the code with `ast`
- reject banned imports/modules
- reject banned calls
- verify the required entrypoint exists

Return a structured review like:

```python
{
    "passed": True,
    "errors": [],
}
```

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: inspect generated preprocessing code"
```

### Task 5: Execute Generated Code In An Isolated Workspace

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add a test for:

- `execute_generated_preprocessing(...)`

The test should provide simple generated code that writes the required artifacts into a temp workspace and verify:

- workspace path is created
- generated code file is saved
- expected output artifacts exist

Required artifacts:

- `cleaned_frame.csv`
- `feature_frame.csv`
- `target.csv`
- `split_manifest.json`
- `preprocessing_report.json`

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL because execution support is missing.

**Step 3: Write minimal implementation**

Implement:

- `execute_generated_preprocessing(raw_df, generated_code, run_root) -> dict`

Behavior:

- create isolated run directory
- persist input dataset there
- write generated code to a `.py` file
- execute it with a timeout
- collect stdout/stderr
- return artifact paths and execution log

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: execute generated preprocessing code in isolated workspace"
```

### Task 6: Validate Generated Preprocessing Artifacts

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add tests for:

- `validate_preprocessing_output(...)`

Checks must include:

- target not present in feature columns
- split manifest exists
- group overlap is zero when grouped split is used
- feature frame is non-empty
- target file exists

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL because validator is missing.

**Step 3: Write minimal implementation**

Implement:

- `validate_preprocessing_output(execution_result, dataset_policy_spec, column_transform_spec) -> dict`

This should read the generated artifacts and return:

```python
{
    "passed": True,
    "checks": {...},
    "errors": [],
}
```

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: validate generated preprocessing artifacts"
```

### Task 7: Add Repair Loop For Failed Preprocessing Code

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add a test for:

- `repair_preprocessing_code(...)`

The test should monkeypatch the LLM call and assert that:

- previous code
- inspection errors
- execution log
- validation report

are all passed into the repair prompt.

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL because repair function is missing.

**Step 3: Write minimal implementation**

Implement:

- `repair_preprocessing_code(...) -> dict`

Use `skills/repair-preprocessing-code.md` as the system prompt.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "feat: add preprocessing code repair loop"
```

### Task 8: Rewire The Graph To The Codegen Loop

**Files:**
- Modify: `src/bt5151_credit_risk/graph.py`
- Test: `tests/test_graph.py`

**Step 1: Write the failing test**

Update `tests/test_graph.py` to expect these preprocess nodes:

- `dataset-policy-spec`
- `column-transform-spec`
- `generate-preprocessing-code`
- `inspect-preprocessing-code`
- `execute-generated-preprocessing`
- `validate-preprocessing-output`
- `repair-preprocessing-code`

Add an end-to-end test that monkeypatches:

- preprocess policy LLM
- column transform LLM
- preprocessing codegen LLM
- downstream business LLM

and verifies the graph reaches training and inference.

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_graph.py -v
```

Expected: FAIL because the graph still uses the old preprocess execution nodes.

**Step 3: Write minimal implementation**

Refactor `graph.py` so the preprocess part becomes:

- `dataset-policy-spec`
- `column-transform-spec`
- `generate-preprocessing-code`
- `inspect-preprocessing-code`
- `execute-generated-preprocessing`
- `validate-preprocessing-output`
- `repair-preprocessing-code`

Use a simple repair loop policy:

- if validation passes, continue to `train-models`
- if validation fails and attempts < 2, go to `repair-preprocessing-code`
- if validation fails and attempts >= 2, raise a clear error

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_graph.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/graph.py tests/test_graph.py
git commit -m "feat: rewire graph to preprocessing codegen loop"
```

### Task 9: Add New Skill Files For Runtime Prompt Injection

**Files:**
- Create: `skills/generate-preprocessing-code.md`
- Create: `skills/inspect-preprocessing-code.md`
- Create: `skills/execute-generated-preprocessing.md`
- Create: `skills/validate-preprocessing-output.md`
- Create: `skills/repair-preprocessing-code.md`
- Modify: `README.md`

**Step 1: Write the failing test**

If practical, add a README or skill-loader test asserting that the new skills are discoverable.

**Step 2: Run test to verify it fails**

Run the relevant test if added.

**Step 3: Write minimal implementation**

Add the five skill files.

Important:

- LLM-driven skill files are prompt policy at runtime.
- deterministic skill files still document execution contract for the assignment.

Update README architecture to show the codegen loop.

**Step 4: Run verification**

Run:

```bash
find skills -maxdepth 1 -type f | sort
sed -n '1,240p' README.md
```

Expected: new skill files exist and README reflects the new preprocess architecture.

**Step 5: Commit**

```bash
git add skills README.md
git commit -m "docs: align skills and readme with preprocessing codegen loop"
```

### Task 10: Update Notebook To Match The New Preprocessing Loop

**Files:**
- Modify: `bt5151_credit_risk_pipeline.ipynb`
- Test: `tests/test_notebook_smoke.py`

**Step 1: Write the failing test**

Update notebook smoke test to assert the notebook references:

- `compile_graph`
- `generate-preprocessing-code`

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_notebook_smoke.py -v
```

Expected: FAIL because the notebook still reflects the older preprocess block.

**Step 3: Write minimal implementation**

Update the notebook text and sample result display so it clearly reflects the generated-code preprocessing loop.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_notebook_smoke.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add bt5151_credit_risk_pipeline.ipynb tests/test_notebook_smoke.py
git commit -m "feat: update notebook for preprocessing codegen loop"
```

### Task 11: Remove Old Preprocess Execution Surface

**Files:**
- Modify: `src/bt5151_credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Step 1: Write the failing test**

Add or update tests so the old compatibility functions are no longer required:

- remove dependence on `preprocess_credit_data`
- remove dependence on `_default_dataset_policy_spec`
- remove dependence on `_default_column_transform_spec`

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: FAIL if legacy helpers are still wired into tests or runtime.

**Step 3: Write minimal implementation**

Delete:

- `_default_dataset_policy_spec`
- `_default_column_transform_spec`
- `preprocess_credit_data`

Leave only the codegen-driven preprocess path.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
PYTHONPATH=src pytest tests/test_preprocess.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/bt5151_credit_risk/preprocess.py tests/test_preprocess.py
git commit -m "refactor: remove fallback preprocessing path"
```

### Task 12: Full Verification

**Files:**
- No new files unless fixes are needed

**Step 1: Run full suite**

```bash
source .venv/bin/activate
PYTHONPATH=src pytest -v
```

Expected: PASS.

**Step 2: Sanity check branch**

```bash
git status --short --branch
```

Expected: clean branch except for known untracked local data files.

**Step 3: Commit final verification if needed**

```bash
git add <files>
git commit -m "test: verify preprocessing codegen loop"
```
