# Contract-Hardening, Feature Lineage, and Shortcut Audit

**Date:** 2026-04-16
**Status:** Planned
**Motivation docs:** [feature-review-diagnosis](../../lab/analysis/pipeline-diagnostics-qa-insights.md), this session's log-audit of runs 010–20260416_012744

## Problem statement

Six consecutive full runs produced XGB macro_f1 in the 0.68–0.70 band on grouped entity split. No prompt change moved the number. Three structural reasons:

1. **Preprocessing codegen is unstable by design.** `generate_preprocessing_code` at [preprocess.py:87](../../src/bt5151_credit_risk/preprocess.py#L87) and `execute_generated_preprocessing` at [preprocess.py:263](../../src/bt5151_credit_risk/preprocess.py#L263) run a fresh pandas program every run. Prompt fixes never become stable behavior. Most recent run had `Age` surviving as object dtype with negative values.
2. **Numeric-role validation is one-sided.** [preprocess.py:620](../../src/bt5151_credit_risk/preprocess.py#L620) and [preprocess.py:630](../../src/bt5151_credit_risk/preprocess.py#L630) guard value-level checks with `if numeric and …`. If the role is `numeric_continuous` but the dtype is `object`, the validator silently skips. That is exactly how `Age-as-string` survived.
3. **FE validation is structural, not semantic.** [validate_feature_engineering_output](../../src/bt5151_credit_risk/feature_engineering.py#L244) checks row counts, NaN/inf, feature cap, top-MI presence, low-cardinality identity. It does not verify that `EMI_to_Salary_Ratio` was built from raw parents, or that any derived formula matches the hypothesis. We have inferred feature meaning from logs for four runs without ever proving it.

Scope: this is a **contract-hardening** pass, not a rewrite. We keep codegen for both stages and tighten validators enough that unreliable outputs cannot pass.

## Design principles

- **Trust the pipeline, not the prose.** Every run must emit contract-report artifacts that can be diffed across runs. Logs are not the source of truth.
- **Structured lineage, not formula strings.** Lineage is a JSON graph of `{operation, inputs, input_stage}`, not a Python expression to eval.
- **Data-aware thresholds over brittle absolutes.** Prefer fractional/relative checks to hardcoded magic numbers.
- **Deterministic verdicts for suspect features.** SHAP/PFI disagreement, dominance share, and ablation are numbers, not LLM prose.
- **Cap cost aggressively on every new check.** No new phase may double training time.

## Architecture after changes

```
raw CSV
  │
  ▼
column-transform-spec (LLM reasoning only)
  │
  ▼
preprocessing codegen
  │
  ▼
preprocessing validator  ── emits preprocessing_contract_report.json
  ├─ structural checks
  ├─ dtype contract (role → required dtype family)
  ├─ role-invariant checks (existing)
  ├─ cross-field invariants (new: Age, Credit_History_Age, missing-exclusivity)
  ▼
feature engineering codegen  ── must emit feature_lineage.json
  │
  ▼
feature validator  ── emits feature_contract_report.json
  ├─ structural checks (existing)
  ├─ top-MI preservation (existing, extended to consume lineage drop_reason)
  ├─ lineage replay (new: recompute derived features from raw parents, compare)
  ├─ raw-parent rule for ratios (new: input_stage must be pre_fe_raw_numeric)
  ▼
training
  │
  ▼
global XAI
  │
  ▼
shortcut-feature-audit  ── emits shortcut_audit.json
  ├─ SHAP/PFI rank disagreement detection
  ├─ dominance-share check
  ├─ capped ablation (top-2 suspects, reuse params, selected model only)
  ▼
interpretation (consumes shortcut audit)
  │
  ▼
analysis bundle
```

Four new artifacts persisted per run alongside existing bundle JSON:
- `preprocessing_contract_report.json`
- `feature_lineage.json` (written by FE codegen, consumed by validator)
- `feature_contract_report.json`
- `shortcut_audit.json`

## Phase 0 — Honest baseline (no code changes)

**Goal:** have one comparable number to judge later phases against.

**Work:**
1. Create scratch branch. Revert `feature_engineering.py`, `preprocess.py`, FE/preprocess skills to Run 009 state. Keep grouped entity split. Rerun full pipeline.
2. Record resulting XGB macro_f1.
3. Record current-HEAD XGB macro_f1 on same split (already captured: **0.6943**).
4. Write both numbers + delta into `lab/experiments/013-grouped-split-baseline.md`.

**Deliverable:** the only honest anchor for every later phase.

**Cost:** one pipeline run (~45 min).

## Phase 1 — Close the numeric-role validation hole

**Goal:** `Age-as-string` or any other wrong-dtype numeric column must fail validation deterministically.

### 1.1 Dtype contract gate

**File:** [src/bt5151_credit_risk/preprocess.py:620](../../src/bt5151_credit_risk/preprocess.py#L620)

Before the existing `if numeric and …` value-level checks for `numeric_count` and `numeric_continuous`, add a shared dtype gate:

```python
elif role in {"numeric_count", "numeric_continuous"}:
    if not numeric:
        record(
            column=m, role=role,
            violation="not_numeric_dtype",
            observed=f"dtype={series.dtype}",
            expected="numeric (int/float)",
            likely_cause=(
                "column remained non-numeric after cleaning. Verify order: "
                "(1) replace garbage tokens → NaN, (2) strip non-numeric artifacts, "
                "(3) pd.to_numeric(errors='coerce'), (4) impute. "
                "Do not run imputation on an object-dtype series."
            ),
        )
        continue  # skip value-level checks when dtype is wrong
    # existing numeric-only checks follow …
```

**Note on role taxonomy:** only `numeric_continuous` and `numeric_count` exist per [current-state.md:77](../../docs/architecture/current-state.md#L77). Do not invent `numeric_ratio`.

### 1.2 Cross-field semantic invariants

**File:** [src/bt5151_credit_risk/preprocess.py](../../src/bt5151_credit_risk/preprocess.py) — new function `validate_semantic_invariants(df, spec, row_count)`, called from `validate_preprocessing_output` after `validate_semantic_roles`.

Checks:

| # | Invariant | Rationale |
|---|-----------|-----------|
| 1 | `Age` numeric, `18 ≤ Age ≤ 100`, `Age.nunique() ≥ 10` | catches Age-as-string and single-value collapse |
| 2 | `Credit_History_Age` numeric, `0 ≤ val ≤ 1000` (months), `nunique / raw_nunique ≥ 0.3` | catches year-only parsing when raw had >200 unique strings |
| 3 | `Credit_History_Age % 12 != 0` for at least 5% of non-null rows | catches regex that discards month component |
| 4 | `Credit_History_Age ≤ (Age − 18) × 12 × 1.1` holds for ≥ 95% of rows | catches obvious parse errors without rejecting legitimate edge cases |
| 5 | For every `<col>_missing` sentinel, if `==1` then all `<col>_*` siblings must be 0 (exact) | catches double-encoded missingness |
| 6 | No `<col>_Not Specified` dummy when `<col>_missing` exists | forbids duplicate missingness representation |

All six emit structured `{column, violation, observed, expected, likely_cause}` findings using the existing record pattern.

**Data-aware thresholds:** invariant 2 uses `raw_nunique` from the raw frame (passed in from `validate_preprocessing_output`), not a magic `100`. Invariant 3 uses a 5% fraction, not "never all multiples of 12" — allows legitimate years-only borrowers without collapsing the whole column.

### 1.3 Persist contract report artifact

**File:** [src/bt5151_credit_risk/preprocess.py](../../src/bt5151_credit_risk/preprocess.py) in `validate_preprocessing_output`

At end of the function, after collecting all findings, write `preprocessing_contract_report.json` to the run workspace:

```json
{
  "run_id": "...",
  "timestamp": "...",
  "passed": true,
  "role_violations": [...],
  "cross_field_violations": [...],
  "dtype_violations": [...],
  "structural_checks": {...}
}
```

### 1.4 Tests

**File:** [tests/test_preprocess.py](../../tests/test_preprocess.py)

- `test_numeric_role_rejects_object_dtype` — build a frame with `Age` as object dtype declared `numeric_count`, expect `not_numeric_dtype` finding
- `test_credit_history_age_all_year_multiples_flagged` — synthetic column where every value is a multiple of 12, expect violation 3
- `test_type_of_loan_missing_sentinel_mutual_exclusivity` — row has `Type_of_Loan_missing==1` AND `Type_of_Loan_Personal Loan==1`, expect violation 5
- `test_age_plausibility_catches_single_value` — Age column is a single constant, expect `nunique` violation

**Scope estimate:** ~80 LoC in `preprocess.py`, ~60 LoC in tests, 4 new test cases.

## Phase 2 — Feature lineage manifest + replay validation (+ log-ratio guard merged)

**Goal:** no derived feature enters training without a verifiable lineage entry, and no ratio can be built from log-transformed parents.

### 2.1 Lineage manifest schema

**Decision:** structured JSON, not code-string formulas. No `eval`. Deterministic replay.

**File:** [skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md) and [skills/repair-feature-engineering-code.md](../../skills/repair-feature-engineering-code.md)

Require the FE entrypoint to write `feature_lineage.json` alongside `engineered_train.csv` / `engineered_test.csv`:

```json
{
  "derived_features": [
    {
      "feature": "EMI_to_Salary_Ratio",
      "operation": "ratio",
      "inputs": ["Total_EMI_per_month", "Monthly_Inhand_Salary"],
      "input_stage": "pre_fe_raw_numeric",
      "fill_strategy": "zero_aware_branching",
      "clip_strategy": "p99_upper",
      "note": "optional free-text rationale"
    },
    {
      "feature": "Annual_Income_log1p",
      "operation": "log1p",
      "inputs": ["Annual_Income"],
      "input_stage": "pre_fe_raw_numeric",
      "fill_strategy": null,
      "clip_strategy": null
    }
  ],
  "dropped_features": [
    {
      "feature": "Payment_of_Min_Amount_No",
      "drop_reason": "deterministic_duplicate",
      "correlated_with": "Payment_of_Min_Amount_Yes"
    }
  ],
  "passthrough_features": [
    "Credit_Mix", "Outstanding_Debt", "..."
  ]
}
```

Allowed enumerations (enforced by validator):
- `operation`: `ratio`, `product`, `sum`, `difference`, `log1p`, `bin`, `interaction`, `passthrough`
- `input_stage`: `pre_fe_raw_numeric`, `pre_fe_encoded`, `fe_derived`
- `drop_reason`: `leakage`, `deterministic_duplicate`, `constant`, `correlation_with_higher_mi_feature`

### 2.2 Replay validator

**File:** [src/bt5151_credit_risk/feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py) — new function `validate_feature_lineage(lineage, train_frame_pre_fe, train_frame_post_fe, top_mi_features)`, called from `validate_feature_engineering_output`.

Checks:

| # | Check | Action on failure |
|---|-------|-------------------|
| 1 | `feature_lineage.json` exists and parses | fail `lineage_artifact_present` |
| 2 | Every non-passthrough column in `engineered_train.csv` has exactly one `derived_features` entry | fail `lineage_coverage_complete` |
| 3 | For each `derived_features` entry with `operation ∈ {ratio, product, sum, difference, interaction}`: `input_stage == "pre_fe_raw_numeric"` | fail `ratios_use_raw_parents` (this subsumes Phase-5) |
| 4 | **Replay check:** sample 20 rows. For each sampled row, recompute the feature from `inputs` (read from `train_frame_pre_fe`) using the declared `operation`. Compare to actual output with `rtol=1e-4` | fail `lineage_replay_matches` with offending rows |
| 5 | Every `dropped_features` entry with a top-5 MI input requires `drop_reason ∈ {leakage, deterministic_duplicate}` | fail `top_mi_drop_requires_justification` |

**On replay check:** `operation="ratio"` means `inputs[0] / inputs[1]` with the declared `fill_strategy`. If the codegen computed something different (e.g. divided log-transformed values), the replay mismatches and fails. This is the structural answer to "is EMI_to_Salary_Ratio actually raw EMI over raw salary?"

**Implementation note:** `train_frame_pre_fe` is already available — it's `state.train_frame` before FE runs. Thread it through the validator signature.

### 2.3 Persist contract report artifact

**File:** [src/bt5151_credit_risk/feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py)

At end of `validate_feature_engineering_output`, write `feature_contract_report.json` next to the engineered CSVs. Same schema shape as preprocessing contract report.

### 2.4 Tests

**File:** [tests/test_feature_engineering_validation.py](../../tests/test_feature_engineering_validation.py)

- `test_lineage_replay_catches_log_transformed_parents` — lineage claims `input_stage="pre_fe_raw_numeric"` but the actual output equals `log1p(EMI) / log1p(Salary)`; replay must fail
- `test_lineage_missing_artifact_fails` — delete `feature_lineage.json`, expect `lineage_artifact_present` violation
- `test_top_mi_drop_without_justification_fails` — drop `Monthly_Inhand_Salary` with `drop_reason="correlation_with_higher_mi_feature"`, expect failure
- `test_ratio_with_encoded_inputs_fails` — `input_stage="pre_fe_encoded"` on a ratio, expect `ratios_use_raw_parents` violation

**Scope estimate:** ~150 LoC in `feature_engineering.py`, ~100 LoC in tests, 2 prompt edits, 4 new test cases.

## Phase 3 — Correlation-drop keeper rule (lineage-aware)

**Goal:** top-5 MI features cannot be dropped by correlation heuristic alone.

### 3.1 Keeper rule in FE prompt (already mostly done)

**File:** [skills/generate-feature-engineering-code.md](../../skills/generate-feature-engineering-code.md)

Verify (from this session's MI-first edit) the hierarchy is present:
- leakage → always drop
- deterministic duplicate → drop either
- otherwise → higher-MI keeps
- never drop top-5 MI raw feature without explicit leakage or deterministic_duplicate evidence

### 3.2 Enforcement via lineage metadata

**File:** [src/bt5151_credit_risk/feature_engineering.py](../../src/bt5151_credit_risk/feature_engineering.py) — extend the existing `check_top_mi_not_dropped`:

If a top-5 MI feature is absent from output columns AND it appears in `feature_lineage.json` `dropped_features` with a `drop_reason` other than `leakage` or `deterministic_duplicate`, fail with:

> Top-MI feature 'Monthly_Inhand_Salary' was dropped with reason 'correlation_with_higher_mi_feature'.
> Rule: top-5 MI raw features cannot be dropped by correlation heuristic. Only `leakage` or `deterministic_duplicate` justifies dropping a top-5 MI feature. Restore it or escalate the drop_reason.

**Scope estimate:** ~30 LoC, depends on Phase 2 manifest being wired.

## Phase 4 — Shortcut-feature audit (capped cost)

**Goal:** deterministic verdicts on Month, Credit_Mix dominance, and any future shortcut feature, without doubling training time.

### 4.1 Suspect detection

**File:** new [src/bt5151_credit_risk/shortcut_audit.py](../../src/bt5151_credit_risk/shortcut_audit.py) (keeps `xai.py` from bloating)

`detect_shortcut_suspects(shap_ranks, pfi_ranks, shap_importance) → list[dict]` flags a feature when ANY of:

| Signal | Threshold |
|--------|-----------|
| SHAP rank ≤ 5 AND PFI rank > 10 | divergence |
| Single feature's `mean\|SHAP\|` share > 20% of total top-10 `mean\|SHAP\|` | dominance |
| Feature name matches calendar/index pattern (`^Month$`, `^Day`, `^Year$`, `^Index$`, etc.) AND appears in SHAP top-10 | calendar shortcut |

Each suspect gets: `{feature, signals: [...], shap_rank, pfi_rank, shap_share}`.

### 4.2 Capped ablation

`ablate_suspects(model, X_test, y_test, suspects, max_ablations=2) → list[dict]`

Rules (cost cap):
- Maximum 2 ablations per run (top-2 suspects by severity)
- No retraining from scratch. **Reuse fitted model params**, refit with the one feature zeroed out (not removed from the dataframe — set to median for numeric, mode for categorical). This avoids schema changes in the fitted model.
- Alternative if zeroing is not meaningful for tree models: use `permutation_importance` single-feature result as a cheaper proxy.
- Selected model only. No LR/RF re-ablation.

Each result: `{feature, delta_macro_f1, delta_per_class_recall, verdict}`.

Verdict rule (deterministic):
- `|Δ macro_f1| < 0.005` → `"weak_signal"` (model does not actually depend on it)
- `Δ macro_f1 < -0.02` → `"real_signal"` (dropping hurts)
- otherwise → `"inconclusive"`

### 4.3 Graph wiring

**File:** [src/bt5151_credit_risk/graph.py](../../src/bt5151_credit_risk/graph.py)

- New node `shortcut-feature-audit` between `global-xai` and `interpret-global-xai`
- Writes `shortcut_audit.json` to the run workspace
- Adds `shortcut_audit: dict | None = None` to [state.py](../../src/bt5151_credit_risk/state.py)

### 4.4 Interpretation consumes audit

**File:** [skills/interpret-global-xai.md](../../skills/interpret-global-xai.md)

Add instruction: "If `shortcut_audit` contains a suspect with `verdict: weak_signal`, treat that feature's SHAP rank as untrusted and do not cite it as a primary driver."

### 4.5 Tests

**File:** new [tests/test_shortcut_audit.py](../../tests/test_shortcut_audit.py)

- `test_detect_shap_pfi_divergence` — SHAP rank 3, PFI rank 15 → flagged `divergence`
- `test_detect_dominance` — single feature 40% share → flagged `dominance`
- `test_detect_calendar_shortcut` — `Month` in SHAP top-10 → flagged `calendar_shortcut`
- `test_ablation_cap_respected` — 5 suspects, only 2 ablated
- `test_verdict_thresholds` — synthetic deltas map to correct verdicts

**Scope estimate:** ~120 LoC in `shortcut_audit.py`, ~80 LoC in tests, 1 graph node, 1 state field, 1 prompt edit.

## Files summary

### New (4)
| File | Purpose |
|------|---------|
| `src/bt5151_credit_risk/shortcut_audit.py` | Suspect detection + capped ablation |
| `tests/test_shortcut_audit.py` | Phase 4 tests |
| `lab/experiments/013-grouped-split-baseline.md` | Phase 0 deliverable |
| `docs/plans/2026-04-16-contract-hardening-lineage-shortcut-audit.md` | This file |

### Modified (~10)
| File | Change |
|------|--------|
| `src/bt5151_credit_risk/preprocess.py` | Dtype gate, cross-field invariants, contract report |
| `src/bt5151_credit_risk/feature_engineering.py` | Lineage replay, raw-parent rule, contract report, MI-drop enforcement |
| `src/bt5151_credit_risk/graph.py` | Shortcut-audit node, lineage wiring, contract report persistence |
| `src/bt5151_credit_risk/state.py` | `shortcut_audit` field |
| `skills/generate-feature-engineering-code.md` | Lineage manifest contract |
| `skills/repair-feature-engineering-code.md` | Lineage manifest contract |
| `skills/interpret-global-xai.md` | Consume shortcut audit |
| `tests/test_preprocess.py` | Phase 1 tests |
| `tests/test_feature_engineering_validation.py` | Phase 2 tests |
| `docs/architecture/current-state.md` | Architecture diagram update |
| `docs/changes/implementation-log.md` | Log phase completions |

## Execution order

| Phase | Depends on | Independent? | Estimated scope |
|-------|-----------|--------------|-----------------|
| 0 — honest baseline | none | yes | 1 run |
| 1 — numeric dtype gate + invariants + contract report | none | yes | ~140 LoC + 4 tests |
| 2 — lineage manifest + replay + raw-parent rule | prompt + validator | partially (needs prompt edit first) | ~250 LoC + 4 tests + 2 prompt edits |
| 3 — MI drop enforcement via lineage | Phase 2 manifest | no | ~30 LoC |
| 4 — shortcut audit + capped ablation | none (but most useful after Phase 2) | yes | ~200 LoC + 5 tests |

Phases 0, 1, 4 can proceed in parallel. Phases 2 → 3 must be sequential.

## Verification

After all phases:

1. `pytest tests/` — all tests pass (current: 128; after: ~145)
2. Full pipeline run on grouped split. Verify artifacts exist:
   - `preprocessing_contract_report.json`
   - `feature_lineage.json`
   - `feature_contract_report.json`
   - `shortcut_audit.json`
3. Open `feature_lineage.json` from the run. Confirm `EMI_to_Salary_Ratio.input_stage == "pre_fe_raw_numeric"` and replay passed.
4. Compare XGB macro_f1 against Phase 0 baseline. Report delta in `lab/experiments/014-post-contract-hardening.md`.

## Must remain true

- Codegen stays for both preprocessing and FE. This is contract hardening, not a rewrite.
- Numeric roles cannot pass validation with non-numeric dtype.
- No ratio can be computed from log-transformed parents.
- Top-5 MI features can only be dropped with `drop_reason ∈ {leakage, deterministic_duplicate}`.
- Shortcut audit is deterministic (numbers, not LLM prose).
- Ablation cost is bounded: max 2 ablations per run, selected model only, no re-tuning.
- All contract verdicts are persisted to disk as JSON artifacts, not left in logs.
- `numeric_ratio` is not invented — only the 12 existing semantic roles in [current-state.md:77](../../docs/architecture/current-state.md#L77) are referenced.
- Phase 0 ships before Phase 1 to establish the honest anchor.

## What this plan does NOT do

- Does not rewrite preprocessing or FE as deterministic transforms.
- Does not promise a macro_f1 improvement. Promises observability: we will finally know whether the FE ratios are what we think they are, and whether `Month` is a real signal or a shortcut.
- Does not add new preprocessing rules — the dtype gate is the only new rule, and it already follows from the existing role taxonomy.
