# Plan: XAI Overhaul — 4-Layer Hypothesis-Driven Analysis

## Context

Run 012 (XGB macro_f1=0.802) works end-to-end but has major XAI gaps identified in our deep dive discussion (`lab/analysis/xai-hypothesis-driven-deep-dive.md`). The pipeline produces numbers but no interpretation. Key problems:

1. **EDA outputs statistics, not hypotheses** — MI/ANOVA/correlations are computed but never interpreted into directional predictions about model behavior
2. **Training has no diagnostics** — no per-class struggle analysis, no capacity diagnosis, no learning curve interpretation
3. **XAI is SHAP-only** — no PFI (grouped), no PDP, no ALE. Missing critical perspectives on feature importance
4. **Local XAI is a single arbitrary row** — no systematic case selection (per-class representative + worst misclassification)
5. **Explain node doesn't validate hypotheses** — no three-tier framework (tested/supported/exploratory)
6. **Training is single-threaded** — RF takes 2h26m because n_jobs defaults to 1

**Philosophical shift**: Every analytical node should produce bold directional hypotheses in three tiers — tested (closed loop), supported (partially testable), and exploratory (open threads that open doors even without proof).

## Design Principle: Port the Posture, Not the Notebook

The XAI case study (Melbourne housing) is a **thinking model** — it teaches analytical posture, hypothesis chaining, and method selection reasoning. It is NOT a template to copy. The case study is regression; our pipeline is multiclass classification. Some things translate directly, some need fundamental adaptation.

**What we port**: the discipline of forming bold hypotheses at each layer, chaining observations across layers, and selecting methods by task characteristics.

**What we adapt to classification**:

| Case Study (Regression) | Our Pipeline (Multiclass Classification) | Why |
|---|---|---|
| Residual plots (predicted - actual) | **Confidence analysis** (how confident when right vs wrong) + **confusion flow** (which class pairs confuse, symmetric or asymmetric?) + **calibration** (predicted probabilities vs actual frequencies) | Classification has no continuous residual — the equivalent diagnostic is "where and how confidently is the model wrong" |
| Under-predicted / representative / over-predicted | **Representative** (most confident correct per class) + **worst misclassification** (most confident wrong per class) + **borderline** (least confident correct per class — reveals decision boundary thinness) | Regression error is directional (too high/low); classification error is categorical (which wrong class, how confident) |
| PDP/ALE → single curve (feature → predicted value) | PDP/ALE → **one curve per class** (feature → predicted probability for class K). Interpretation: how does this feature shift probability mass between classes? | Regression predictions are scalar; classification predictions are probability vectors across K classes |
| Segment analysis by property type (R² per segment) | Segment analysis by **class confusion pairs** — which feature ranges cause Good→Standard confusion vs Standard→Poor confusion. Segments defined by confusion matrix structure, not external categories | Regression segments are predefined groups; classification segments emerge from the model's error patterns |

**What we do NOT hard-code**:
- Not all case study artifacts become pipeline requirements — methods are selected at runtime based on task characteristics (number of classes, feature types, correlation structure)
- The architecture must remain reusable across classification tasks — the specific credit risk hypotheses are examples, not templates
- Skill prompts should instruct the LLM on the analytical posture (form bold hypotheses, chain observations, select methods by task) rather than prescribing specific outputs

## Design Principle: Analysis Bundle as Central Artifact

All analytical outputs from the hypothesis/XAI layers accumulate into a single **analysis bundle** — a structured, serializable artifact that:
- Gets saved to disk as JSON alongside the experiment record
- Has a stable schema (not ad-hoc dict blobs shaped by whatever the LLM returns)
- Is the single input to explain-risk (compact summary, not 4 separate state fields)
- Enables cross-run comparison ("did grouped PFI and SHAP become more aligned after FE v2?")
- Enables hypothesis tracking across runs ("exploratory lead from run N → tested in run N+1")

```
analysis_bundle:
  ├─ eda_hypotheses          # three-tier predictions from EDA
  ├─ training_diagnostics    # capacity analysis, confusion flow, hypothesis validation
  ├─ global_xai              # SHAP + PFI (always), PDP/ALE (conditional)
  └─ local_casebook          # representative / misclassification / borderline per class
```

A dedicated `package-analysis-bundle` node after local-xai serializes the bundle to disk and stores a compact summary in state for explain-risk. This creates an explicit boundary between evidence-building and explanation.

**Persistence layers** (distinct concerns):
- `state` = live orchestration memory for this run
- `analysis_bundle` = persisted analytical evidence from this run (JSON artifact)
- `model artifacts` = fitted model, preprocessing outputs
- `experiment record` = human-readable summary in `lab/experiments/`

## Design Principle: Method Gating

XAI methods are not all mandatory. Selection depends on runtime task characteristics:

| Method | When | Why |
|---|---|---|
| SHAP (global + per-case) | **Always** | Universal feature attribution, works for all model types |
| Grouped PFI | **Always** | Model-agnostic importance, groups one-hot columns |
| ALE | **Conditional**: when EDA shows |r| > 0.5 pairs among top features | PDP is biased for correlated features; ALE corrects this |
| PDP | **Conditional**: for top continuous features without high correlation | Intuitive but misleading when features are correlated |
| ICE | **Optional**: for meaningful subgroup stories (if domain suggests it) | Per-instance PDP curves; useful for heterogeneity |
| LIME | **Out of scope** | Weak for one-hot tabular; SHAP is strictly better here |

The `global-xai` node checks EDA correlation structure at runtime to decide which conditional methods to execute, rather than hard-coding "always compute everything."

## Design Principle: Task-Aware Case Selection (Casebook Strategy)

Local XAI case selection is framed as a **strategy**, not a hard-coded heuristic:

- **Classification strategy** (this project): representative correct, confident wrong, ambiguous borderline — per class
- **Regression strategy** (future): under-predicted, representative, over-predicted — by error magnitude

The architecture defines the interface (select N cases with rationale, compute SHAP per case), and this project implements the classification strategy. The strategy pattern keeps the architecture open without over-engineering.

## Graph After Changes

```
START → dataset-policy-spec → EDA (programmatic, existing)
  → generate-eda-hypotheses (LLM, NEW) → column-transform-spec
  → [preprocessing loop] → [FE loop]
  → train-models → evaluate-models → training-diagnostics (LLM, NEW)
  → select-model → global-xai (programmatic, NEW) → local-xai (programmatic, NEW)
  → package-analysis-bundle (NEW) → run-inference
  → explain-risk (UPDATED) → recommend-action → END
```

5 new nodes, 1 updated node, 1 speed fix. 2 new Python modules, 3 new skill files.

---

## Workstream 1: Training Speed Fix

**Why**: RF takes 2h26m on 80k×48 because everything is single-threaded.

**Changes:**
- `src/bt5151_credit_risk/train.py:36` — add `n_jobs=-1` to `RandomForestClassifier()`
- `src/bt5151_credit_risk/graph.py:489` — change baseline `cross_val_score(..., n_jobs=1)` to `n_jobs=-1`
- Optuna `study.optimize(n_jobs=1)` stays — TPE is inherently sequential

---

## Workstream 2: EDA Hypothesis Generation

**Why**: EDA computes MI=0.18 for Annual_Income but never says "this predicts LR should hit 0.55+ from top-2 features alone." The gap between statistics and directional predictions is where bold hypotheses live.

**New file**: `src/bt5151_credit_risk/hypotheses.py`
```python
def generate_eda_hypotheses(eda_report: dict, dataset_profile: dict) -> dict:
    """LLM interprets EDA stats → three-tier hypotheses."""

def generate_training_diagnostics(...) -> dict:
    """LLM interprets training results, validates EDA hypotheses."""
```

**New file**: `skills/generate-eda-hypotheses.md` — instructs LLM to produce:
- `tested_predictions`: e.g., "LR macro_f1 >= 0.55 from top-2 MI features alone"
- `supported_conjectures`: e.g., "Standard class will have lowest recall — class-conditional means overlap with both neighbors"
- `exploratory_leads`: e.g., "Monthly_Balance × Num_Credit_Inquiries interaction may separate borderline Standard from Good"
- `model_selection_prediction`: directional prediction on which model wins and why
- `class_struggle_prediction`: which class will struggle most

**State**: Add `eda_hypotheses: dict | None = None`

**Graph**: Insert `generate-eda-hypotheses` between `exploratory-data-analysis` and `column-transform-spec`. Remove edge EDA→column-transform-spec, add EDA→generate-eda-hypotheses→column-transform-spec.

---

## Workstream 3: Training Diagnostics

**Why**: After training, we report "XGB macro_f1=0.802" but not "LR is capacity-limited (C=0.016 means heavy regularization), RF max_depth=27 means interaction-heavy signal, XGB didn't early-stop → still learning at round 1000."

**Adaptation from case study**: The case study uses residual plots and R² per segment (regression). We adapt to classification equivalents:
- **Confidence analysis** replaces residual analysis — how confident is the model when right vs wrong? A model that's 0.95 on correct but 0.80 on misclassifications is different from one at 0.55 on both.
- **Confusion flow analysis** replaces error distribution — which class pairs confuse? Is it symmetric (Good↔Standard equally) or asymmetric (Standard→Good much more than Good→Standard)?
- **Per-class struggle** replaces per-segment R² — which classes are hardest, and why (overlap? minority? feature profile?)

**New file**: `skills/generate-training-diagnostics.md` — LLM receives evaluation_results, tuning_results, learning_curves, eda_hypotheses, fe_hypothesis. Produces:
- `per_class_analysis`: per-class struggle level + diagnosis + confusion flow direction
- `capacity_analysis`: per-model capacity diagnosis
- `confidence_analysis`: confidence distribution on correct vs incorrect predictions
- `learning_curve_interpretation`: convergence analysis
- `hypothesis_validation`: which EDA tested_predictions were confirmed/refuted with actual values
- `new_hypotheses`: new three-tier hypotheses generated from training observations

**State**: Add `training_diagnostics: dict | None = None`

**Graph**: Insert `training-diagnostics` between `evaluate-models` and `select-model`.

**Note**: The programmatic data for confidence analysis (predict_proba distributions) is computed in the node function before passing to the LLM. The LLM interprets, not computes.

---

## Workstream 4: Global XAI Expansion

**Why**: SHAP alone shows "what features matter" but not "how they matter" (PDP/ALE) or "what happens when features are grouped" (PFI grouped for one-hot). PDP vs ALE divergence on correlated features is itself an insight.

**Adaptation from case study**: The case study computes PDP/ALE as single curves (feature → predicted value). For multiclass, PDP/ALE produce **one curve per class** (feature → predicted probability for class K). The interpretation shifts from "how does this feature move the prediction" to "how does this feature shift probability mass between classes." A feature where PDP shows Poor probability rising while Good probability falls is doing something different from one where Standard rises at the expense of both neighbors.

**New file**: `src/bt5151_credit_risk/xai.py`
- `compute_global_shap()` — refactored from inline code in graph.py select_model_node. Returns importance list + beeswarm data + dependence data
- `compute_permutation_importance()` — grouped PFI: groups one-hot columns by shared prefix, permutes together. Custom wrapper around sklearn's `permutation_importance`
- `compute_partial_dependence()` — PDP for top-5 continuous features via sklearn. Returns per-class probability curves (not a single predicted value curve)
- `compute_ale()` — custom ALE implementation (~50 lines, no alibi dependency). Quantile binning → finite differences → accumulate + center. Returns per-class ALE curves
- `compute_shap_contributions_for_case()` — refactored from `graph.py:_compute_shap_contributions`

**Key design decisions:**
- ALE is custom (alibi is too heavy for one function)
- PFI grouping uses column name prefix matching (columns like `Type_of_Loan_Home Loan` → group `Type_of_Loan`)
- Global SHAP in select_model_node stays (feeds model selection reasoning). global-xai node computes PFI/PDP/ALE + expanded SHAP (beeswarm/dependence) for selected model only
- SHAP is recomputed in global-xai rather than cached (TreeExplainer on 500 samples < 5s)
- PDP and ALE both return per-class curves — the divergence between PDP and ALE for a given feature+class is itself diagnostic (correlated feature bias)

**Method gating logic** (in node function):
```python
# Always
results["shap"] = compute_global_shap(...)
results["pfi"] = compute_permutation_importance(...)  # grouped

# Conditional — check EDA correlation structure
high_corr_features = {p["col_a"] for p in eda_high_pairs} | {p["col_b"] for p in eda_high_pairs}
top_continuous = [f for f in top_shap_features if f is continuous and f not in one_hot_columns]

if top_continuous:
    correlated_features = [f for f in top_continuous if f in high_corr_features]
    uncorrelated_features = [f for f in top_continuous if f not in high_corr_features]
    if correlated_features:
        results["ale"] = compute_ale(..., correlated_features)  # ALE where correlation exists
    if uncorrelated_features:
        results["pdp"] = compute_partial_dependence(..., uncorrelated_features)  # PDP where safe

results["methods_used"] = list(results.keys())  # track what was gated in/out
```

**Graph**: Insert `global-xai` after `select-model`. Edge: select-model → global-xai → local-xai.

---

## Workstream 5: Local XAI Casebook

**Why**: One arbitrary row tells you nothing systematic. Systematic case selection reveals the model's behavior space.

**Casebook strategy (classification)**: The case study selects under-predicted / representative / over-predicted (regression direction). For classification, error is categorical and probabilistic, not directional. Our classification case selection strategy:
- **Representative** (most confident correct per class) — shows what the model "understands well"
- **Worst misclassification** (most confident wrong per class) — shows where the model is confidently wrong and which class it confuses with
- **Borderline** (least confident correct per class) — reveals decision boundary thinness, the cases where one feature shift would flip the prediction

This gives up to 9 cases (3 per class × 3 classes), covering the full behavior space.

**Architecture note**: Case selection is implemented as a strategy function in `xai.py`:
```python
def select_classification_cases(test_frame, test_target, model, class_names) -> list[dict]:
    """Classification casebook strategy: representative + worst + borderline per class."""

# Future: def select_regression_cases(...) for regression tasks
```

The `local-xai` node calls the strategy function, then computes SHAP waterfall (top-10 features) for each selected case.

**Graph**: Insert `local-xai` between `global-xai` and `package-analysis-bundle`.

---

## Workstream 6: Package Analysis Bundle

**Why**: Create an explicit packaging boundary between evidence-building and explanation. The analysis bundle is a persisted artifact that enables cross-run comparison and hypothesis tracking.

**Node logic** (`package_analysis_bundle_node` in `graph.py`):
1. Collect `eda_hypotheses`, `training_diagnostics`, `global_xai_results`, `local_xai_cases` from state
2. Assemble into `analysis_bundle` dict with stable schema + metadata (run timestamp, model name, dataset hash)
3. Save to disk as `analysis_bundle.json` in the experiment workspace
4. Build a compact summary for explain-risk (top-line findings, not the full PDP/ALE grids)
5. Store both `analysis_bundle` (full) and `analysis_bundle_summary` (compact) in state

**State**: Add `analysis_bundle: dict | None = None`, `analysis_bundle_summary: dict | None = None`

**Graph**: Insert `package-analysis-bundle` between `local-xai` and `run-inference`.

---

## Workstream 7: Explain-Risk Hypothesis Chain

**Why**: explain-risk currently gets fragments of the hypothesis chain. It should receive the compact bundle summary and produce structured three-tier validation.

**Changes to `business.py:explain_risk()`**: Replace the 4 separate hypothesis parameters with a single `analysis_bundle_summary` parameter. The compact summary contains the essential findings without PDP grids or beeswarm arrays.

**Changes to `skills/explain-risk.md`**: Restructure to request:
- `hypothesis_validation`: which tested hypotheses confirmed/refuted with evidence
- `supported_findings`: which conjectures gained evidence
- `exploratory_threads`: which leads remain open and what would test them

**Changes to `graph.py:explain_risk_node()`**: Pass `state.analysis_bundle_summary` to `explain_risk()`.

---

## Files Summary

### New files (5)
| File | Type | Purpose |
|------|------|---------|
| `src/bt5151_credit_risk/hypotheses.py` | Module | `generate_eda_hypotheses()`, `generate_training_diagnostics()` |
| `src/bt5151_credit_risk/xai.py` | Module | SHAP, grouped PFI, PDP, ALE, casebook strategy, per-case SHAP |
| `skills/generate-eda-hypotheses.md` | Skill | Three-tier EDA hypothesis generation prompt |
| `skills/generate-training-diagnostics.md` | Skill | Training diagnostics + hypothesis validation prompt |
| `tests/test_xai.py` | Test | PFI grouping, ALE accumulation, method gating, casebook selection |

### Modified files (8)
| File | Change |
|------|--------|
| `src/bt5151_credit_risk/train.py` | n_jobs=-1 on RF |
| `src/bt5151_credit_risk/graph.py` | 5 new nodes, new edges, baseline n_jobs=-1, refactor SHAP into xai.py |
| `src/bt5151_credit_risk/state.py` | New fields: eda_hypotheses, training_diagnostics, global_xai_results, local_xai_cases, analysis_bundle, analysis_bundle_summary |
| `src/bt5151_credit_risk/business.py` | explain_risk() receives analysis_bundle_summary instead of 4 separate params |
| `skills/explain-risk.md` | Three-tier hypothesis validation output |
| `tests/test_graph.py` | 5 new nodes in expected set, monkeypatch new functions |
| `tests/test_state.py` | New fields |
| `docs/changes/implementation-log.md` | Log all changes |

---

## Implementation Order

| Step | Workstream | Depends On |
|------|-----------|-----------|
| 1 | WS1: Training speed fix (train.py, graph.py) | None |
| 2 | WS2: hypotheses.py + EDA hypotheses skill + node | None |
| 3 | WS4: xai.py module (all compute functions + method gating + casebook strategy) | None |
| 4 | WS3: Training diagnostics skill + node | WS2 |
| 5 | WS4: global-xai node wiring in graph.py | WS3, xai.py |
| 6 | WS5: local-xai node (casebook) | xai.py |
| 7 | WS6: package-analysis-bundle node | WS4-6 |
| 8 | WS7: explain-risk updates (consumes bundle summary) | WS7 |
| 9 | Tests + docs | All |

Steps 1, 2, 3 can proceed in parallel.

---

## Verification

1. `pytest tests/` — all tests pass
2. Full pipeline run: `PYTHONPATH=src .venv/bin/python3 run_stage.py full 42`
3. Check logs for:
   - EDA hypotheses: tested predictions, supported conjectures, exploratory leads logged
   - Training diagnostics: per-class analysis, confidence analysis, capacity diagnosis, hypothesis validation
   - Global XAI: methods_used logged (which methods were gated in/out), PFI grouped, SHAP beeswarm
   - Local XAI: up to 9 cases (3 per class: representative, worst misclassification, borderline), SHAP waterfall per case
   - Analysis bundle: saved to disk as JSON, compact summary logged
   - Explain-risk: three-tier hypothesis validation in output
   - Training time: RF should drop from 2h26m to ~30min with n_jobs=-1
4. `analysis_bundle.json` artifact exists and is valid JSON with stable schema
5. Experiment record in `lab/experiments/`

## Must Remain True

- Bold hypotheses must be grounded in observable data (not fabricated)
- Hypotheses must be labeled by tier (tested/supported/exploratory)
- PFI must group one-hot columns — ungrouped PFI on one-hot is misleading
- **Method gating**: SHAP + grouped PFI always; ALE when EDA shows correlation; PDP when features are uncorrelated; ICE optional. Never hard-code "always compute everything"
- Local XAI uses casebook strategy pattern — classification strategy selects representative + worst misclassification + borderline per class
- explain-risk receives compact bundle summary, not raw PDP grids or beeswarm arrays
- explain-risk must not present exploratory hypotheses with same confidence as tested ones
- Optuna trials must stay sequential (TPE depends on prior trials)
- Global SHAP in select_model_node must stay (model selection needs it before global-xai runs)
- **Analysis bundle is a persisted artifact** — saved to disk, stable schema, enables cross-run comparison
- **Port the posture, not the notebook** — the case study is a thinking model for analytical discipline, not a template. Regression artifacts (residuals, under/over-predicted, single-curve PDP) must be adapted to classification equivalents (confidence analysis, confusion flow, per-class probability curves). Skill prompts teach the analytical approach, not prescribe specific outputs
- Architecture must remain reusable across classification tasks — credit-risk-specific hypotheses are examples in skill prompts, not hard-coded logic
