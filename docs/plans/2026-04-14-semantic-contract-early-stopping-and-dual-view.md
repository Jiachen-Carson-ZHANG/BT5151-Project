# Plan: Semantic Contract, Early Stopping Policy, and Dual-View Representation

## Context

The pipeline is now past the "just patch the latest bug" stage. We have aligned on a stronger architecture:

- reasoning models should decide semantics and policy
- worker/codegen nodes should implement those decisions
- deterministic code should enforce binary contracts and leakage-sensitive invariants
- repeated failure on the same contract is a capability signal, not just "try again"

Recent work fixed the immediate feature-engineering prompt contradiction and introduced `semantic_role` plus deterministic preprocessing validation. That gives us a stronger foundation, but the responsibility boundary across `column-transform-spec`, preprocessing, feature engineering, and training is still only partially reflected in the code.

This plan makes that boundary explicit and orders the remaining patches so we do not mix stabilization with redesign again.

## Core decisions

1. **`column-transform-spec` remains the brain for per-column semantics.**
It should decide what a column *is*, what risks it carries, and what kind of representation family is appropriate.

2. **Preprocessing remains a worker, not a second reasoning layer.**
It should clean, parse, split, and enforce the semantic contract. It should not become the place where we keep inventing model-specific representation logic.

3. **Feature engineering becomes the representation builder.**
It should stay as one graph node, but over time it should become responsible for model-facing views such as `linear_view` and `tree_view`.

4. **Leakage-sensitive mechanics are deterministic.**
LLM chooses policy. Code enforces fit-on-train, apply-to-test, artifact alignment, and validation invariants.

5. **No dataset-specific hardcoding.**
Policies should be general enough to apply to grouped, IID, or temporal datasets without credit-risk-specific logic.

## Responsibility adjustment

There **is** a responsibility adjustment, but it is staged rather than a big-bang rewrite.

| Component | Responsibility now | Target responsibility |
|---|---|---|
| `dataset-policy-spec` | task type, target, group column, split strategy | same, plus `validation_policy` for early stopping / inner validation |
| `column-transform-spec` | per-column action, cleaning, encoding choice | per-column semantic contract: `semantic_role`, leakage risk, cleaning/imputation intent, and representation hints |
| Preprocessing node | cleaning + encoding + split + feature frame generation | canonical base-table worker: parse, normalize, split, fit safe preprocessing, validate semantic-role contract |
| FE node | codegen for interactions/transforms on model-ready frame | free-code generation over base train/test tables; produce model-facing views (`linear_view`, `tree_view`) plus shared engineered features |
| Train node | tune models on one shared feature frame | deterministic execution of `validation_policy`; train each model on the correct view |

## Reasoning vs non-reasoning split

### Reasoning-model responsibilities

- `dataset-policy-spec`
- `column-transform-spec`
- any future `validation_policy` reasoning extension
- EDA / FE / training interpretation nodes

These prompts should explain principles, tradeoffs, and downstream consequences.

### Worker / codegen responsibilities

- `generate-preprocessing-code`
- `repair-preprocessing-code`
- `generate-feature-engineering-code`
- `repair-feature-engineering-code`

These prompts should be explicit about execution order, required artifacts, and non-negotiable postconditions.

### Deterministic enforcement responsibilities

- semantic-role validator
- structural artifact validation
- train-fit / test-apply invariants
- early stopping policy executor
- alignment checks across train/test/model views

This is where leakage prevention lives.

## Workstream 1: Finish preprocessing contract stabilization

### Goal

Make the new semantic-role contract fully operational, not just documented.

### Changes

1. **Repeated contract failure should trigger escalation, not blind looping**
- Track violation signatures such as `(column, declared_role, violation)` across preprocessing attempts
- If the same signature repeats twice in a row, escalate to a stronger repair/codegen model
- Keep this generic: no column-specific logic

2. **Tighten role-validation routing**
- Ensure `role_violations` remain part of the repair context
- Keep deterministic findings compact and structured so a worker model can act on them literally

3. **Bring the architecture docs up to date**
- Update `docs/architecture/current-state.md` to describe:
  - `semantic_role`
  - `representation_intent`
  - deterministic role validation
  - stronger-model escalation rule

### Success criteria

- repeated identical role failure changes model behavior, not just retry count
- a `multi_value_set` binary-membership regression fails loudly and repairs deterministically
- docs describe the live contract accurately

## Workstream 2: Generalize early stopping policy

### Goal

Make early stopping trustworthy across dataset shapes without hardcoding this dataset.

### Design

Add `validation_policy` to `dataset_policy_spec`, for example:

```json
{
  "validation_policy": {
    "type": "iid_stratified | grouped_entity | temporal",
    "group_column": "Customer_ID",
    "time_column": null,
    "stratify_target": true
  }
}
```

### Responsibilities

- `dataset-policy-spec` decides the policy type
- preprocessing preserves the metadata needed to execute that policy
- `train.py` executes the policy deterministically for early stopping and any inner validation split

### Why this is the right boundary

The LLM is good at recognizing whether a dataset is IID, grouped, or time-ordered.
The LLM should **not** free-generate the leakage-sensitive split mechanics every run.

### Changes

1. Extend `dataset-policy-spec` schema with `validation_policy`
2. Persist required metadata through preprocessing artifacts / state
3. Refactor early-stopping split builder in `train.py` to dispatch by policy type
4. Log policy choice and resulting split stats

### Success criteria

- no credit-risk-specific branching in training code
- grouped datasets can use grouped inner validation
- time-ordered datasets can use temporal validation
- logs make the chosen policy auditable

## Workstream 3: Shift representation-building from preprocessing to FE

### Goal

Move from "preprocessing makes one model-ready frame for everyone" to "preprocessing makes a canonical base table, FE makes model-facing views."

### Important constraint

This should happen **inside the existing FE node**, not by adding a new graph node.

### Transitional stance

Do **not** move everything at once.

#### Phase 3A: Keep the current graph shape, but clarify the boundary

Preprocessing should own:
- placeholder cleanup
- parsing dirty numeric / temporal strings
- target extraction
- identifier/group handling
- split
- canonical base-table semantics

Feature engineering should own:
- interactions
- monotonic/log transforms
- redundancy pruning
- model-facing representation building

#### Phase 3B: FE node emits two aligned views

The FE node should eventually output:

- `engineered_train_linear.csv`
- `engineered_test_linear.csv`
- `engineered_train_tree.csv`
- `engineered_test_tree.csv`
- `feature_engineering_report.json`
- `view_metadata.json`

### Encoding architecture

The LLM should not choose raw encoders from scratch every run with no structure.
Instead:

1. `column-transform-spec` declares semantics and representation hints
2. FE codegen uses those hints to build:
   - `linear_view`
   - `tree_view`
3. deterministic validation checks:
   - fit learned transforms on train only
   - apply to test without refitting
   - row alignment preserved
   - train/test columns aligned within each view

### Long-term spec direction

`column-transform-spec` can evolve toward:

```json
{
  "semantic_role": "unordered_categorical",
  "representation_intent": "preserve_category_identity",
  "view_hints": {
    "linear_view": "one_hot_or_safe_target_encoding",
    "tree_view": "ordinal_or_binary_membership"
  }
}
```

This keeps the reasoning layer semantic and the worker layer executable.

### Success criteria

- preprocessing no longer needs to compromise between LR and tree models
- FE can free-generate model-appropriate views without changing graph shape
- validators can still enforce leakage-sensitive invariants deterministically

## Workstream 4: Evaluation and experiment closure

### Goal

Do not claim architectural improvement without a clean run and evidence.

### Required evidence after each major workstream

1. **Preprocessing stabilization rerun**
- convergence behavior across attempts
- whether repeated violations escalate correctly

2. **Early stopping policy rerun**
- chosen `validation_policy`
- learning-curve behavior
- final best iteration / early-stopping decision

3. **Dual-view FE rerun**
- view artifacts written correctly
- each model trained on the intended view
- metric comparison versus shared-frame baseline
- XAI artifact quality after representation shift

### Experiment records

Each significant rerun should write:
- a new `lab/experiments/*.md` record
- a short entry in `docs/changes/implementation-log.md`

## Recommended order

1. Finish preprocessing contract stabilization
2. Rerun and record evidence
3. Implement `validation_policy` for early stopping
4. Rerun and record evidence
5. Redesign FE node to emit dual views inside the existing node
6. Rerun and compare against the single-frame baseline

## Out of scope for this plan

- dropping LR immediately
- forcing target encoding as the default
- adding extra graph nodes just to separate concerns
- rewriting the whole preprocessing system before the current contract is stable

Those may become valid later, but they should be decisions supported by experiment evidence, not mixed into the current stabilization work.
