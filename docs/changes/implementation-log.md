# Implementation Log

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
