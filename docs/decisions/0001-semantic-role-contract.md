# ADR 0001: Semantic role contract for preprocessing

**Status:** Accepted
**Date:** 2026-04-14

## Context

Run 013 exposed a recurring preprocessing quality regression: the LLM codegen produced a multi-hot encoding for `Type_of_Loan` whose indicator columns took values in {0, 1, 2} (counts of occurrences) instead of {0, 1} (presence). The audit flagged it as "value range unusual," the repair loop tried three times without fixing it, and the graph accepted the output via the 3-attempt escape hatch.

The failure mode is general: the audit is prose-based and the LLM receives weakly-typed feedback. When a column's *intended semantics* (binary multi-hot, ordered ordinal, binary flag, etc.) are not explicit in the spec, the validator has nothing concrete to enforce and the repair prompt has nothing concrete to act on. Convergence depends on the LLM's willingness to re-read its own output for drift, which is unreliable.

## Decision

Every column in `column_transform_spec.transforms` carries two new fields:

- `semantic_role` — one of 12 enumerated roles describing what the column *is* (not how it is encoded): `identifier`, `group_identifier`, `target`, `numeric_continuous`, `numeric_count`, `ordered_categorical`, `unordered_categorical`, `binary_flag`, `multi_value_set`, `temporal_feature`, `free_text`, `leakage_risk_feature`.
- `representation_intent` — optional string declaring how the column is encoded when the role admits multiple valid encodings (e.g. `binary_membership` vs `count_membership` for `multi_value_set`; `one_hot` vs `target_encoded` for `unordered_categorical`).

A deterministic validator (`validate_semantic_roles` in [preprocess.py](../../src/bt5151_credit_risk/preprocess.py)) runs after the generated preprocessing code and checks each column's output against the invariant implied by its declared role. Violations are emitted as structured findings — `{column, declared_role, violation, observed, expected, likely_cause}` — and merged into the validation report under `role_violations`. Any role violation blocks the `passed` flag and triggers the repair loop.

The repair prompt renders these findings as concrete instructions with a named likely cause, so even a non-reasoning model can act on them without inferring what "value range unusual" means.

## Design principles

1. **Role is intrinsic, intent is a choice.** Cardinality, correlation structure, or any other *property* of the data drives the choice of `representation_intent` — it does not split the role taxonomy. The 12-role list stays closed.
2. **Role assignment is reasoning work.** The `column-transform-spec` skill is the only place a role is assigned; it is invoked on a reasoning-capable model because the decision requires interpretation of profile + EDA signals.
3. **Enforcement is deterministic.** The validator is Python, not an LLM. The same column, same role, same output always produces the same finding. Findings are designed to be concrete enough that a non-reasoning model can act on them.
4. **Escalate before looping.** If the same role violation appears twice in a row, that is capability-ceiling signal — per the model escalation policy in [AGENT.md](../../AGENT.md), surface it rather than retrying with the same model.

## Alternatives considered

- **Keep prose-based audit feedback only.** Rejected: this is what Run 013 demonstrated does not converge.
- **Split roles by cardinality (`unordered_categorical_low_cardinality` vs `unordered_categorical_high_cardinality`).** Rejected: conflates role (what it is) with treatment choice (how to encode). Cardinality drives `representation_intent`, not role.
- **Assign roles in a separate LLM node before `column-transform-spec`.** Rejected: adds a graph node and duplicates context already available to `column-transform-spec`. Bundling role assignment with the existing spec generation keeps the graph compact.

## Consequences

**Positive**
- Role violations fail loudly with named root cause, not vaguely as "quality issue."
- Same contract works across datasets without project-specific checks.
- Enables the model-escalation policy: repeated violation of the same contract is an objective trigger.
- Role field is a durable attribute of each column usable by downstream nodes (e.g. FE hypothesis generation, XAI explain-risk).

**Negative**
- LLM spec output must include two new fields per column — slight token overhead.
- If the reasoning model mis-assigns a role (e.g. calls an ordered column unordered), the validator will enforce the wrong contract. Mitigation: the validator checks *contract violation*, not semantic correctness of the role choice itself; mis-assigned roles will surface downstream in model performance and trigger spec revision.

## References

- [skills/column-transform-spec.md](../../skills/column-transform-spec.md) — role taxonomy + invariant table
- [skills/generate-preprocessing-code.md](../../skills/generate-preprocessing-code.md) — codegen contract section
- [skills/repair-preprocessing-code.md](../../skills/repair-preprocessing-code.md) — structured-finding consumption
- [src/bt5151_credit_risk/preprocess.py](../../src/bt5151_credit_risk/preprocess.py) — `validate_semantic_roles`
- [AGENT.md](../../AGENT.md) — Model selection & escalation
