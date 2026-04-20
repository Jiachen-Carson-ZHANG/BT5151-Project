# Two-Phase Architecture: Offline Analysis + Lightweight Gradio Inference

**Date**: 2026-04-13
**Status**: Planned — depends on XAI overhaul (analysis bundle) being completed first
**Context**: Assignment requires a Gradio frontend (Section 4.1 point 6). Current implementation runs the entire LangGraph pipeline per Gradio request (~4 hours per slider click). This is impractical for an interactive demo.

---

## The Problem

The current Gradio interface (`bt5151_credit_risk_pipeline.ipynb` Cell 6) calls `run_pipeline(row_index)` which executes the full graph from START to END — EDA, LLM specs, codegen, preprocessing, FE, training 3 models with Optuna, evaluation, model selection, XAI, inference, explain, recommend.

With the XAI overhaul adding PFI/PDP/ALE/casebook generation, this becomes even heavier. An interactive demo needs < 5 seconds per request, not hours.

## The Solution: Two-Phase Separation

Not two deployed systems. Two execution phases within the same codebase.

### Phase 1: Offline Analysis (runs once per experiment)

```
dataset-policy-spec → EDA → hypotheses → preprocess → FE → train
→ evaluate → diagnostics → select-model → global-xai → casebook
→ package-analysis-bundle
→ PERSIST to disk:
    - trained_model.pkl (fitted model object)
    - preprocessing_pipeline.pkl (fitted scaler, encoder, etc.)
    - feature_engineering_pipeline.pkl (fitted FE transforms)
    - analysis_bundle.json (hypotheses, diagnostics, global XAI, casebook)
    - metadata.json (class_names, feature_columns, label_to_id, id_to_label)
```

Triggered by: `run_stage.py full 42` or equivalent. Takes ~30min-1hr with n_jobs fix.

### Phase 2: Lightweight Gradio Inference (runs per user request)

```
load persisted artifacts (model, preprocessing, FE, bundle, metadata)
→ user picks row (or enters new data)
→ apply preprocessing pipeline to that row
→ apply FE pipeline to that row
→ model.predict_proba()
→ compute local SHAP for this prediction (TreeExplainer, < 1 second)
→ explain-risk (LLM call using prediction + compact bundle summary, ~2 seconds)
→ recommend-action (LLM call, ~1 second)
→ display in Gradio
```

Total per-request time: < 5 seconds (dominated by LLM calls for explain + recommend).

## What Needs to Happen

### Persistence layer (after XAI overhaul)

1. **Model serialization**: `joblib.dump()` the selected trained model after `select-model` node
2. **Preprocessing pipeline serialization**: The current preprocessing is LLM-generated code, not a fitted sklearn Pipeline. Two options:
   - Option A: Save the generated preprocessing code + the fitted artifacts it produces (feature_frame columns, split manifest). At inference time, apply the same transformations to a new row.
   - Option B: Refactor preprocessing into a fitted sklearn ColumnTransformer that can be pickled. This is a bigger change.
   - **Recommendation**: Option A is pragmatic. The preprocessing code already produces a deterministic transform — we just need to extract the transform logic (without the train/test split) into a reusable function.
3. **Feature engineering serialization**: Same situation — LLM-generated code. Save the transform function.
4. **Analysis bundle**: Already persisted to disk by `package-analysis-bundle` node (from XAI overhaul plan).
5. **Metadata**: Save class_names, feature_columns, label_to_id, id_to_label as JSON.

### Inference graph (or function)

A lightweight `build_inference_graph()` or a simple function that:
1. Loads all persisted artifacts from a run directory
2. Takes a raw input row (or row index into the dataset)
3. Applies preprocessing → FE → predict → SHAP → explain → recommend
4. Returns the same output structure as the full pipeline's last 3 nodes

### Gradio interface update

```python
# Load once at startup
artifacts = load_inference_artifacts("path/to/latest/run")

def predict_from_row(row_index):
    row = raw_df.iloc[row_index]
    result = run_inference(artifacts, row)
    return (
        result["predicted_label"],
        result["risk_summary"],
        result["recommended_action"],
        result["action_reason"],
        result["selected_model_name"],
        result["confidence"],
        # NEW: hypothesis-grounded explanation
        result["hypothesis_notes"],
    )
```

### Enhanced Gradio interface (opportunity)

With the analysis bundle available, the Gradio interface can be much richer:
- Show the analysis bundle summary alongside predictions (global feature importance, model selection reasoning)
- Show the local SHAP waterfall for each prediction
- Show how this prediction compares to the casebook cases (is it closer to a representative case or a borderline case?)
- Show the three-tier hypothesis validation

## Dependencies

- **Requires**: XAI overhaul completed (analysis bundle exists and is persisted)
- **Requires**: Decision on preprocessing serialization approach (Option A vs B)
- **Blocked by**: Nothing after XAI overhaul is done

## Why This Matters Beyond the Assignment

This is the same "analysis bundle as iteration artifact" pattern from `lab/analysis/analysis-bundle-iteration-architecture.md`. The two-phase separation:
- Lets us iterate the explanation strategy without retraining (load bundle, try different explain-risk prompts)
- Lets us compare runs by loading different bundles into the same Gradio interface
- Lets us add a "lighter inference-only path" later if needed (the friend's original suggestion)
- Makes the Colab demo actually interactive (assignment requirement)

## Estimated Scope

Small workstream once the XAI overhaul is done:
- Model/metadata serialization: ~1 hour
- Inference function (load + predict + explain): ~2 hours
- Gradio interface update: ~1 hour
- Testing: ~1 hour
