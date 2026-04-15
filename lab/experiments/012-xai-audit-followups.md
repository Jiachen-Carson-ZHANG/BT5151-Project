# 012 — XAI audit follow-ups: PFI dtype, bundle verbatim, split interpret, explain-risk expansion

## Motivation
Audit of run `stage_full_20260415_030032.log` surfaced six issues that together explain (a) the PFI failure and (b) why explain-risk inherits generic content despite a rich upstream hypothesis chain:

1. PFI errored with "object dtype" at [graph.py:962](../../src/bt5151_credit_risk/graph.py) despite FE CSVs on disk being fully numeric.
2. `package-analysis-bundle` compressed the interpretation before handing it to explain-risk (`observations[:5]`, `insights[:3]`, `pfi_top5`, casebook reduced to `{case_type,true,pred}`).
3. `interpret-xai-evidence` (a) received a further-compacted payload in [hypotheses.py:29](../../src/bt5151_credit_risk/hypotheses.py) and (b) mixed global and local reasoning in one LLM call, flattening per-case stories.
4. `skills/explain-risk.md` was only 64 lines, its schema didn't match what graph.py actually built, and [business.py:18](../../src/bt5151_credit_risk/business.py) only received the compressed summary.
5. `Credit_History_Age` parser in generated preprocessing had greedy `.*` between year and month groups — unique values collapsed to multiples of 12 (34 values vs 405 in Run 009).
6. Occupation was frequency-encoded at preprocess, erasing category identity.

Run 012 macro_f1 was 0.6930 vs Run 009's 0.8017 — the three data-quality regressions above (plus a lighter tuning budget) are the most-supported explanation.

## Changes
- **PFI bool→int8 cast** in `xai.py` — fixes the NumPy-boundary dtype promotion that broke PFI on tree-view frames containing bool multi-hot columns.
- **Split interpretation into two LLM nodes**: `interpret-global-xai` (SHAP/PFI/PDP/ALE consensus, feature-effect shapes, cross-layer validation) and `interpret-local-xai` (per-class stories, confusion patterns, boundary analysis). Second node gets the first's output as reference. New skills under `skills/`.
- **Bundle verbatim**: removed all programmatic truncation. `analysis_bundle_summary` now equals the full semantic bundle. Numeric-only artifacts (beeswarm arrays, PFI raw, PDP grids, ALE grids) saved to disk in a separate `numeric_artifacts` block — not shipped to explain-risk.
- **Explain-risk rewrite**: new skill prompt (175 lines) with schema aligned to bundle fields, `local_context` section, evidence-tagged `hypothesis_validation`, `key_drivers` enriched with `raw_value` and global rank context.
- **Credit_History_Age regex gotcha** added to `skills/generate-preprocessing-code.md` with the "all multiples of 12" symptom stated explicitly.
- **Occupation identity-preservation guidance** added to `skills/column-transform-spec.md`.
- **Env updates**: split `OPENAI_MODEL_INTERPRET_XAI_EVIDENCE` into per-node overrides for the two interpretation nodes.

## Expected impact
- PFI will compute on every run → grouped PFI becomes a reliable second opinion alongside SHAP.
- Local-XAI interpretation will cite specific rows (e.g. `row=13466` Poor→Good conf=0.895) with the SHAP drivers that pushed the wrong way, giving explain-risk concrete material.
- Explain-risk output will carry evidence-tagged hypothesis validation (each confirmed/refuted claim points to layer+tier), a counterfactual when decision-boundary evidence supports one, and raw feature values alongside SHAP contributions.
- With `OPENAI_MODEL_COLUMN_TRANSFORM_SPEC=o3` already set in `.env`, next run's spec should preserve Credit_History_Age precision and defer Occupation → macro_f1 should close most of the gap to Run 009.

## Verification plan
- `pytest tests/` → 87 passed
- Next full pipeline run: confirm
  - PFI methods_used includes `pfi_grouped` (no "object dtype" warning)
  - two interpret nodes fire, second cites row indices from the casebook
  - `analysis_bundle_*.json` contains `global_xai_interpretation`, `local_xai_interpretation`, `numeric_artifacts` sections
  - explain-risk JSON output contains `local_context.counterfactual`, `hypothesis_validation.confirmed[].layer`, `key_drivers[].raw_value`
  - `feature_frame.csv` for Credit_History_Age has nunique > 100 (not 34)
  - Occupation in `feature_frame.csv` is object-dtype (deferred), not float

## Status
Implemented. Awaiting full pipeline run to measure delta.
