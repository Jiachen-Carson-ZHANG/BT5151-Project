# Analysis Bundle as Iteration Architecture

**Date**: 2026-04-13
**Context**: Design discussion for XAI overhaul. External reviewer suggested a "two-plane architecture" that, after refinement, revealed a deeper insight about iterative ML experimentation. This document captures the idea for reuse in future projects — particularly hypothesis-driven iteration across training methods, feature engineering variants, and analytical methods.

---

## The Core Insight

In an ML pipeline, there are four distinct persistence layers with different lifetimes and purposes:

| Layer | What | Lifetime | Purpose |
|---|---|---|---|
| **State** | Live orchestration memory | Single run | Wire nodes together during execution |
| **Analysis bundle** | Persisted analytical evidence | Across runs | Compare, iterate, validate hypotheses |
| **Model artifacts** | Fitted model, preprocessing outputs | Until replaced | Serve predictions |
| **Experiment record** | Human-readable summary | Permanent | Lab notebook, audit trail |

Most pipelines only have state and model artifacts. The analysis bundle is the missing layer — it's what enables **hypothesis-driven iteration** across runs.

## What Goes in the Analysis Bundle

```
analysis_bundle:
  metadata:
    run_id, timestamp, dataset_hash, model_name, pipeline_version
  eda_hypotheses:
    tested_predictions, supported_conjectures, exploratory_leads
    model_selection_prediction, class_struggle_prediction
  training_diagnostics:
    per_class_analysis, capacity_analysis, confidence_analysis
    learning_curve_interpretation, hypothesis_validation, new_hypotheses
  global_xai:
    shap_importance, pfi_grouped, pdp (if computed), ale (if computed)
    methods_used, method_gating_reasons
  local_casebook:
    cases[] with row_index, true_label, predicted_label, probabilities,
    shap_contributions, case_type, selection_rationale
```

## Why This Matters for Iteration

After one run produces an analysis bundle, you can:

1. **Compare two runs' bundles side by side**
   - "Did grouped PFI and SHAP importance become more aligned after feature engineering v2?"
   - "Did the Good↔Standard confusion rate drop after adding the interaction features?"
   
2. **Track hypotheses across runs**
   - Run N produces an exploratory lead: "Monthly_Balance × Num_Credit_Inquiries interaction may separate borderline Standard from Good"
   - Run N+1 implements that interaction and the bundle shows whether it became a tested (confirmed/refuted) hypothesis
   - An exploratory lead from run N can become a tested prediction in run N+1

3. **Reuse the bundle without recomputing everything**
   - Try a different explanation prompt against the same bundle
   - Generate alternative business narratives without retraining
   - Run only the downstream analysis (XAI methods) on a cached model

4. **Iterate analytical methods**
   - Run N uses SHAP + PFI. The bundle records methods_used.
   - Run N+1 adds ALE because the bundle from run N showed high-correlation features that make PDP unreliable.
   - The method gating decision is itself informed by prior bundles.

5. **Iterate training methods and FE based on analytical evidence**
   - Bundle shows LR is capacity-limited → try polynomial features or switch to GAM
   - Bundle shows XGB didn't early-stop at 1000 rounds → raise ceiling to 2000
   - Bundle shows top-5 SHAP features are all one-hot fragments → try target encoding
   - Bundle shows Per-class analysis: Standard struggles because feature overlap with Good → engineer a Good-vs-Standard discriminative feature

## The Two-Plane Idea (For Projects With a Frontend/Inference Path)

The original suggestion was to separate a "training/evaluation plane" from a "runtime/inference plane":

```
Training / Evaluation Plane:
  raw data → EDA → hypotheses → preprocess → FE → train → evaluate
  → diagnostics → model selection → global XAI → local casebook
  → package-analysis-bundle (saved to disk)

Runtime / Inference Plane:
  input row → preprocess → predict → local explanation
  → explain-risk (using compact bundle summary) → recommend-action
```

**When this separation matters:**
- When you have a frontend that serves repeated inference requests (not just one demo row)
- When the heavy XAI computation (PFI, PDP, ALE, casebook) should run once during training, not per request
- When the inference path needs to be fast and lightweight
- When you want to update the model without recomputing all XAI evidence

**When it doesn't matter:**
- Single-run pipelines that execute start-to-finish once per experiment
- Course projects where the "inference" is a single demonstration row

**The pragmatic middle ground** (what we adopted for BT5151):
- Keep one linear graph
- Add one explicit packaging boundary (`package-analysis-bundle` node)
- The bundle is saved to disk and a compact summary flows to explain-risk
- If a frontend is added later, it reads the persisted bundle instead of recomputing

## For Hypothesis-Driven Iteration Across Projects

The analysis bundle pattern generalizes beyond XAI. For any project where you're iterating hypotheses:

1. **Define your hypothesis schema** — tested predictions (with testable criteria), supported conjectures (with evidence needed), exploratory leads (with what would test them)
2. **Every analytical node outputs to the bundle** — not just to the next node's state
3. **The bundle is the unit of comparison** — not individual metrics or logs
4. **Hypothesis tracking is across bundles** — exploratory in bundle N, tested in bundle N+1
5. **Method selection is informed by prior bundles** — what to compute next is determined by what we learned last time

This turns an ML pipeline from "train → evaluate → report" into "hypothesize → train → evaluate → validate hypotheses → generate new hypotheses → iterate."

## Stable Schema Matters

The bundle must have a stable schema so cross-run comparison works. If run N's bundle has `global_xai.pfi_grouped` and run N+1 renames it to `global_xai.feature_importance_grouped`, comparison breaks.

Define the schema once. Version it. Validate LLM outputs against it. Add new fields as optional, never rename or remove without a migration.

---

**Source**: Discussion between Carson, external reviewer, and Claude during BT5151 XAI overhaul planning. The two-plane idea was initially rejected as over-architecture for a single-run pipeline, but the underlying insight — analysis bundle as an iteration artifact — is the genuinely reusable idea.
