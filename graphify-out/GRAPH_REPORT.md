# Graph Report - .  (2026-04-19)

## Corpus Check
- 49 files · ~66,842 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 456 nodes · 806 edges · 19 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 132 edges (avg confidence: 0.8)
- Token cost: 22,000 input · 9,000 output

## Community Hubs (Navigation)
- [[_COMMUNITY_XAI Reasoning|XAI Reasoning]]
- [[_COMMUNITY_Preprocessing Contracts|Preprocessing Contracts]]
- [[_COMMUNITY_XAI Reasoning|XAI Reasoning]]
- [[_COMMUNITY_Preprocessing Contracts|Preprocessing Contracts]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Preprocessing Contracts|Preprocessing Contracts]]
- [[_COMMUNITY_XAI Reasoning|XAI Reasoning]]
- [[_COMMUNITY_Training and Selection|Training and Selection]]
- [[_COMMUNITY_Run Trace and Cache|Run Trace and Cache]]
- [[_COMMUNITY_EDA Hypotheses|EDA Hypotheses]]
- [[_COMMUNITY_XAI Reasoning|XAI Reasoning]]
- [[_COMMUNITY_Graphify Governance|Graphify Governance]]
- [[_COMMUNITY_Run Trace and Cache|Run Trace and Cache]]
- [[_COMMUNITY_Graphify Governance|Graphify Governance]]
- [[_COMMUNITY_Core Runtime|Core Runtime]]
- [[_COMMUNITY_Graphify Governance|Graphify Governance]]
- [[_COMMUNITY_Core Runtime|Core Runtime]]
- [[_COMMUNITY_Core Runtime|Core Runtime]]
- [[_COMMUNITY_Core Runtime|Core Runtime]]

## God Nodes (most connected - your core abstractions)
1. `Runtime Skill Prompt Loading` - 22 edges
2. `load_skill_prompt()` - 17 edges
3. `cb_predict()` - 16 edges
4. `call_json_response()` - 13 edges
5. `_get_state()` - 12 edges
6. `train_models_node()` - 12 edges
7. `cb_poll_trace()` - 11 edges
8. `main()` - 10 edges
9. `build_eda_report()` - 10 edges
10. `_log_outputs()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Preprocessing Semantic Normalization` --references--> `validate_preprocessing_output()`  [INFERRED]
  docs/architecture/current-state.md → /home/tough/BT5151 GroupProject/src/bt5151_credit_risk/preprocess.py
- `Class Label Encoding Contract` --references--> `train_models_node()`  [INFERRED]
  docs/architecture/current-state.md → /home/tough/BT5151 GroupProject/src/bt5151_credit_risk/graph.py
- `LangGraph State Pipeline` --references--> `CreditRiskState`  [EXTRACTED]
  docs/architecture/current-state.md → /home/tough/BT5151 GroupProject/src/bt5151_credit_risk/state.py
- `Developer Trace Artifacts` --references--> `cb_poll_trace()`  [EXTRACTED]
  docs/architecture/current-state.md → /home/tough/BT5151 GroupProject/app.py
- `Runtime Skill Prompt Loading` --references--> `load_skill_prompt()`  [EXTRACTED]
  docs/architecture/current-state.md → /home/tough/BT5151 GroupProject/src/bt5151_credit_risk/skill_prompts.py

## Hyperedges (group relationships)
- **Preprocessing Contract Loop** — preprocessing_codegen_loop, semantic_role_contract, preprocessing_repair_escalation, role_violation_validator [EXTRACTED 0.90]
- **Feature Engineering Contract Loop** — feature_engineering_loop, feature_views_contract, feature_lineage_replay_contract, deterministic_feature_engineering_mode [EXTRACTED 0.85]
- **XAI to Business Explanation Chain** — four_layer_hypothesis_chain, xai_method_stack, analysis_bundle_contract, run_inference_skill, explain_risk_skill [INFERRED 0.80]
- **Graphify Governance** — graphify_canonical_scope, graphify_exclusion_policy, graphify_maintenance_rules, agents_graphify_usage, claude_graphify_first [EXTRACTED 0.90]

## Communities

### Community 0 - "XAI Reasoning"
Cohesion: 0.04
Nodes (84): _app_css(), _beeswarm_fig(), _build_action_md(), build_app(), _build_casebook_context_md(), _build_explanation_md(), _build_hypothesis_md(), _build_key_drivers_md() (+76 more)

### Community 1 - "Preprocessing Contracts"
Cohesion: 0.04
Nodes (71): Evaluation Evidence Rule, Analysis Bundle Contract, Confidence Diagnostics Contract, Dataset Policy Skill, Deterministic Model Selection, choose_best_model(), compute_multiclass_metrics(), Evaluate Models Skill (+63 more)

### Community 2 - "XAI Reasoning"
Cohesion: 0.06
Nodes (47): _apply_inline_bold(), build_pipeline_html(), _build_pipeline_items(), build_trace_markdown(), _extract_level(), _extract_message(), _finalise_card(), _format_llm_call_line() (+39 more)

### Community 3 - "Preprocessing Contracts"
Cohesion: 0.08
Nodes (42): Audit Preprocessing Skill, Capability Ceiling Escalation, Column Transform Spec Skill, Generate Preprocessing Code Skill, column_transform_spec_node(), generate_preprocessing_code_node(), repair_preprocessing_code_node(), review_preprocessing_quality_node() (+34 more)

### Community 4 - "Feature Engineering"
Cohesion: 0.08
Nodes (38): Deterministic Feature Engineering Mode, _apply_lineage_operation(), _build_fe_artifact_paths(), _build_feature_stats(), _call_fe_codegen_agent(), deterministic_feature_engineering_fallback_code(), _eval_formula_ast(), _evaluate_feature_formula() (+30 more)

### Community 5 - "Preprocessing Contracts"
Cohesion: 0.1
Nodes (30): _build_provenance_metadata(), _finalize_successful_run(), _log_evaluate(), _log_fe(), _log_full(), _log_outputs(), _log_preprocess(), _log_specs() (+22 more)

### Community 6 - "XAI Reasoning"
Cohesion: 0.11
Nodes (22): _call_json_agent(), explain_risk(), generate_eda_hypotheses(), generate_training_diagnostics(), interpret_global_xai(), interpret_local_xai(), LLM interprets programmatic EDA statistics into three-tier directional hypothese, LLM interprets training results, validates EDA hypotheses, generates new ones. (+14 more)

### Community 7 - "Training and Selection"
Cohesion: 0.16
Nodes (19): Class Label Encoding Contract, _get_train_frame_for_model(), train_models_node(), Reason Hyperparameter Grid Skill, build_candidate_models(), _build_cv_splits(), _build_holdout_indices(), extract_learning_curves() (+11 more)

### Community 8 - "Run Trace and Cache"
Cohesion: 0.2
Nodes (15): Observability Rule, Developer Trace Artifacts, build_trace_event_path(), _extract_artifacts(), _extract_metrics(), _extract_warnings(), _flatten_numeric_dict(), _infer_status() (+7 more)

### Community 9 - "EDA Hypotheses"
Cohesion: 0.24
Nodes (12): build_eda_report(), _compute_cardinality(), _compute_categorical_association(), _compute_class_separability(), _compute_correlations(), _compute_discriminative_features(), _compute_missing_patterns(), _compute_skewness() (+4 more)

### Community 10 - "XAI Reasoning"
Cohesion: 0.22
Nodes (12): shortcut_audit_node(), ablate_suspects(), _calendar_match(), detect_shortcut_suspects(), _predict(), Shortcut-feature audit: deterministic verdicts on suspect top-ranked features., Return a copy of `view` with `feature` replaced by its median (numeric)     or 0, Zero-out ablation for up to max_ablations suspects. Returns list of     {feature (+4 more)

### Community 11 - "Graphify Governance"
Cohesion: 0.38
Nodes (11): Classification, classify_paths(), _extract_paths(), flush(), main(), _normalise_path(), paths_from_hook_stdin(), read_recorded_paths() (+3 more)

### Community 12 - "Run Trace and Cache"
Cohesion: 0.17
Nodes (10): BaseModel, list_caches(), load_cache(), Pipeline state cache — serialize and reload trained pipeline artifacts.  After a, Return all named cache files sorted by mtime (newest first)., Load cached pipeline state from path, or CACHE_FILE if path is None.      Return, Serialize pipeline state to CACHE_FILE.      Args:         result: Dict returned, save_cache() (+2 more)

### Community 13 - "Graphify Governance"
Cohesion: 0.4
Nodes (5): Graphify Usage Rules, Documentation Source of Truth, Claude Graphify First Rule, Canonical Graph Scope, Graphify Exclusion Policy

### Community 14 - "Core Runtime"
Cohesion: 1.0
Nodes (1): BT5151 credit risk monitoring package.

### Community 15 - "Graphify Governance"
Cohesion: 1.0
Nodes (2): Graph as Architecture Map, Not Source Truth, Graphify Maintenance Rules

### Community 16 - "Core Runtime"
Cohesion: 1.0
Nodes (0):

### Community 17 - "Core Runtime"
Cohesion: 1.0
Nodes (1): First Principles Design Rules

### Community 18 - "Core Runtime"
Cohesion: 1.0
Nodes (1): Architecture-Aware Change Policy

## Knowledge Gaps
- **131 isolated node(s):** `Run individual pipeline stages for development and debugging.  Each stage runs t`, `Persist post-run artifacts without letting optional failures stall terminal stat`, `Stream node-by-node, accumulating state updates, stop after target node.`, `Log outputs cumulatively — later stages include all earlier outputs.`, `Build the cache provenance metadata dict from run identifiers.` (+126 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Core Runtime`** (2 nodes): `__init__.py`, `BT5151 credit risk monitoring package.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Graphify Governance`** (2 nodes): `Graph as Architecture Map, Not Source Truth`, `Graphify Maintenance Rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Runtime`** (1 nodes): `config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Runtime`** (1 nodes): `First Principles Design Rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Runtime`** (1 nodes): `Architecture-Aware Change Policy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_inference_node()` connect `Preprocessing Contracts` to `XAI Reasoning`, `Training and Selection`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Why does `_run_inference_step()` connect `XAI Reasoning` to `Preprocessing Contracts`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `explain_risk_node()` connect `Preprocessing Contracts` to `XAI Reasoning`, `XAI Reasoning`?**
  _High betweenness centrality (0.133) - this node is a cross-community bridge._
- **Are the 20 inferred relationships involving `Runtime Skill Prompt Loading` (e.g. with `Dataset Policy Skill` and `Column Transform Spec Skill`) actually correct?**
  _`Runtime Skill Prompt Loading` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `load_skill_prompt()` (e.g. with `ValueError` and `generate_dataset_policy_spec()`) actually correct?**
  _`load_skill_prompt()` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `call_json_response()` (e.g. with `_call_preprocess_agent()` and `_call_preprocess_codegen_agent()`) actually correct?**
  _`call_json_response()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Run individual pipeline stages for development and debugging.  Each stage runs t`, `Persist post-run artifacts without letting optional failures stall terminal stat`, `Stream node-by-node, accumulating state updates, stop after target node.` to the rest of the system?**
  _131 weakly-connected nodes found - possible documentation gaps or missing edges._