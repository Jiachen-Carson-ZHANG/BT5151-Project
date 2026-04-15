# AGENT.md — Project-Level Instructions

## Project context

BT5151 Advanced Analytics and Machine Learning group project (AY 2025/2026, NUS).

We are building a **multi-agent credit risk classification pipeline** where every ML stage (preprocessing, training, evaluation, selection, inference) is a distinct LangGraph node governed by its own `skills/*.md` file. Downstream nodes (explain-risk, recommend-action) translate model output into business-facing language via a Gradio interface.

Dataset: Kaggle credit risk dataset (100k rows, 28 columns, 3-class target: Good/Standard/Poor).

## Deliverables

1. Jupyter notebook runnable in Google Colab
2. One `skills/*.md` per pipeline stage
3. Technical report (2500-3000 words): architecture, business case, ML methodology, evaluation, SKILL.md rationale, system evaluation (3+ end-to-end test cases), AI usage declaration
4. Presentation slides + 20-min recorded video

## Rubric priorities (100 pts total)

| Component | Weight | Key requirement |
|---|---|---|
| ML Development & Evaluation | 30 | 2+ models compared fairly, all required metrics (per-class P/R/F1, macro F1, weighted F1), confusion matrix, critical interpretation |
| SKILL.md Design Quality | 20 | Every skill specifies inputs/outputs from agent state, business-readable |
| Technical Implementation | 20 | Pipeline runs end-to-end, ML output meaningfully consumed by downstream agents |
| Business Case Analysis | 15 | Why ML is right for this problem, how model performance affects business outcomes |
| Critical Reflection | 10 | 3+ documented pipeline runs (including edge cases), 2+ real failures analyzed, improvement proposals |
| Presentation | 5 | All members on camera, Gradio demo, model walkthrough |

## Documentation structure

| Folder | Purpose | When to update |
|---|---|---|
| `docs/architecture/` | System-of-record: current pipeline structure | On material architecture changes |
| `docs/changes/` | Implementation log: what changed, why, tradeoffs | After meaningful implementation work |
| `docs/decisions/` | ADRs: major technical decisions with alternatives | On major design decisions |
| `docs/plans/` | Implementation plans before starting work | Before non-trivial implementation |
| `lab/experiments/` | Experiment records: hypothesis, changes, results, insights | After every experiment run |
| `lab/analysis/` | Any valuable insight: design trade-offs, non-obvious reasoning, why approaches do/don't work | When reasoning is non-obvious or a natural assumption turns out wrong |
| `lab/backlog.md` | Ideas discussed but not yet tested | When an idea is proposed but deferred |
| `lab/logs/` | Raw pipeline logs (symlink to `logs/`) | Automatically generated |

## Experiment discipline

Every pipeline run that tests a hypothesis or validates a change must have an experiment record in `lab/experiments/`. The record must capture:

- **What changed** from the previous state
- **Why** — the hypothesis or assumption behind the change
- **Results** — concrete metrics, pass/fail, token usage, repair rounds
- **What broke** and what succeeded
- **Insights** — why things happened, not just what happened
- **Next steps** — what to try based on these results

Ideas that are discussed but not tested go in `lab/backlog.md` with enough context to act on later.

This is critical for the report: Component 5 (Critical Reflection, 10pts) requires documented runs with actual failures and analysis. Our experiment records are the primary source material for that section.

## Skill prompt rules

1. All `skills/*.md` must be **dataset-agnostic**. No credit-risk-specific column names, values, or domain terms in examples. Use cross-domain examples (medical, retail, etc.) to avoid answer leakage.
2. Every skill must have YAML frontmatter (`name`, `description`), reasoning steps, input/output format, and at least one example.
3. Critical rules (like the `inplace=True` ban) go at the top of the skill, not buried in notes.
4. Follow-up review modes must converge, not escalate — the audit skill uses two-mode review for this.

## Technical constraints

- **pandas 3.x**: `inplace=True` raises `ChainedAssignmentError`. Always use assignment. Enforced by AST inspection.
- **LLM model**: gpt-4o for correctness-critical nodes, gpt-4o-mini for simple text generation.
- **Temperature**: Currently default (1.0). Experiment with lower values if stochastic variation causes convergence issues.
- **No hardcoded API keys** in any committed file. Use `.env` (gitignored) or Google Secrets for Colab.

## Model selection & escalation

Two tiers of LLM work live in this pipeline, and the right model differs by tier:

- **Reasoning work** — hypothesis generation, semantic typing (e.g. assigning `semantic_role` to columns), open-ended interpretation, method gating, analytical synthesis. Use a reasoning model (o4-mini or stronger).
- **Instruction-following work** — emitting code to a strict contract, JSON to a schema, mechanical transforms under a validator. Use a non-reasoning model (gpt-4o or stronger).

**Escalation policy.** When an LLM-driven loop (preprocessing codegen, FE codegen, repair, audit) fails to converge and the bottleneck is model capability — not prompt clarity or missing context — escalate to a stronger model of the appropriate tier rather than burning tokens on more retries with the same model. Token budget is not the constraint; capability is.

Triggers that indicate a capability ceiling (not a prompt issue):
- Same deterministic-contract violation on two consecutive repair attempts.
- LLM output repeatedly ignores an explicit, clearly-stated constraint.
- Structured audit findings are passed in but not acted on in the next iteration.

When any trigger fires, **proactively flag it** rather than switching models silently — hitting a capability ceiling is itself signal about where the pipeline's weak spots are, and is worth surfacing.

## AI usage policy (from project brief)

AI tools may support code writing, debugging, refactoring, and grammar checking. AI must NOT be used to:
- Select or configure models (these are core analytical decisions)
- Generate report text, SKILL.md content, or presentation scripts
- Interpret evaluation results

All AI usage must be declared transparently in the report. The pipeline's LLM-driven preprocessing codegen is part of the system design (not hidden AI assistance) and should be documented as such.
