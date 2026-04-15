# Lab

Experiment records, analysis, and raw logs for the BT5151 credit risk pipeline.

This folder is the **single home for all experiment-related artifacts**. It is separate from `docs/` (which holds system-of-record architecture, decisions, and implementation logs).

## Structure

```
lab/
  experiments/    One file per experiment. Records what changed, why, hypothesis,
                  results, what broke, what worked, and next steps.
  analysis/       Insights, reasoning, and lessons learned. Not limited to experiment
                  comparisons — includes design trade-off analysis, Q&A where the
                  reasoning was non-obvious, and why certain approaches do/don't
                  apply to our system.
  backlog.md      Ideas discussed but not yet tested, prioritized.
  logs/           Raw pipeline logs (symlinked from project root).
```

## How to write an experiment record

Use the template:

```markdown
# Experiment NNN: <title>

Date: YYYY-MM-DD
Stage: <which pipeline stage(s) were tested>
Dataset: <dataset used>

## Hypothesis
What we expected to happen and why.

## Changes
What was changed from the previous state (code, prompts, config).

## Results
Concrete outcomes: metrics, pass/fail, token usage, number of repair rounds.

## Observations
What actually happened — surprises, failures, regressions.

## Insights
What we learned. Why things succeeded or failed.

## Next steps
What to try next based on these results.
```

## Naming convention

- `001-topic-slug.md` — experiments, numbered sequentially
- `topic-slug.md` — analysis docs, named by theme

## Relationship to the report

Experiment records feed directly into:
- **Component 5: Critical Reflection** (10pts) — documented runs, failures, reflections
- **Component 1: ML Methodology** (30pts) — preprocessing decisions, model comparison rationale
- **AI Usage Declaration** — transparent record of what the pipeline did autonomously vs what we directed
