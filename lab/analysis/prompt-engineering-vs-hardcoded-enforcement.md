# Prompt engineering vs hardcoded enforcement: when each works

Date: 2026-04-09

## The question

We discovered that gpt-4o ignored the `inplace=True` ban in the prompt for 3 consecutive code generations, but AST inspection caught and rejected it every time. This raises the question: when should we rely on prompt instructions vs hardcoded enforcement?

## Evidence from our experiments

| Approach | Rule: "never use inplace=True" | Result |
|---|---|---|
| Prompt instruction only (Critical Rules section) | gpt-4o ignored it 3/3 times in Run A | Failed |
| AST inspection (hardcoded rejection) | Caught all 3 violations, forced repair | Succeeded |
| Both together | Prompt reduces frequency, AST catches remaining | Most reliable |

| Approach | Rule: "clip outliers on both sides" | Result |
|---|---|---|
| Prompt instruction (Step 4 in workflow) | gpt-4o clipped lower only for 6 columns | Partially failed |
| No hardcoded enforcement available | Can't AST-check for "reasonable upper bounds" | N/A |
| LLM quality review | Caught 4 of 6 one-sided clipping issues | Partially succeeded |

## The pattern

**Hardcoded enforcement works for binary, mechanical rules:**
- `inplace=True` — either present or not, detectable by AST
- Forbidden imports — either imported or not
- Target column in feature frame — either present or not

These rules have:
- Clear true/false conditions
- No judgment required
- Deterministic detection possible
- The fix is always the same (remove it, replace with assignment)

**Prompt engineering works for judgment calls:**
- "Clip to domain-reasonable bounds" — requires understanding what the column means
- "Split multi-value columns before encoding" — requires recognizing delimiter patterns
- "Use pd.to_numeric instead of .astype(float)" — requires knowing real-world data has string artifacts

These require:
- Domain reasoning
- Context-dependent decisions
- No single correct answer
- Understanding intent, not just syntax

## The lesson

Don't rely on prompts for rules the LLM can just... forget. LLMs are probabilistic — even with a "Critical Rules" section, a rule that says "never do X" will occasionally be ignored, especially in long code generation where attention drifts.

**Rule of thumb**: If you can write a 10-line Python function to detect the violation, enforce it in code. If detecting the violation requires understanding the data's meaning, enforce it in the prompt (and accept partial compliance).

The combination of both is most reliable: the prompt reduces violation frequency (so the repair loop runs fewer times), and the hardcoded check catches the remaining cases.

## Update (2026-04-10): Third dimension — reasoning models

The original analysis assumes a single model type (gpt-4o). With reasoning models (o4-mini), a third approach emerges:

**Principles-based prompts for reasoning models** work for judgment calls that gpt-4o handles inconsistently. Example: "clip outliers to domain-reasonable bounds" failed with gpt-4o (used raw p99=849 for credit cards). With o4-mini and a principle explaining *why* p99 can be inflated by corrupted tails, the model correctly reasoned to [0, 30].

Updated rule of thumb:
- Binary/mechanical rules → hardcoded enforcement (AST check, validator)
- Judgment calls with instruction-following model → explicit prompt rules (accept partial compliance)
- Judgment calls with reasoning model → principles with rationale (model internalizes and applies)

See `lab/analysis/reasoning-model-for-analytical-nodes.md` for the full A/B comparison.
