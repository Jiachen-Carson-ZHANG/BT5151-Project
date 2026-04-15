# When to use hardcoded validation vs LLM review

Date: 2026-04-09

## The question

Our pipeline has two layers of validation after preprocessing:
1. **Hardcoded structural validation** — checks artifact existence, target exclusion, feature frame non-empty, group overlap
2. **LLM quality review** — checks data quality, encoding patterns, distribution sanity

Why not just use one or the other?

## The trade-off

| Property | Hardcoded validation | LLM quality review |
|---|---|---|
| Speed | Instant (<1s) | Slow (5-15s + API cost) |
| Cost | Free | ~12k tokens per call |
| Determinism | 100% reproducible | Stochastic — may vary between runs |
| Scope | Only checks what we coded | Can reason about anything in the data |
| False positives | Zero (by construction) | Non-zero — may flag intentional decisions |
| False negatives | Many — can't reason about data quality | Fewer — but still misses encoding artifacts |
| Failure mode | Silent pass on novel issues | May hallucinate issues or miss subtle ones |

## When each is appropriate

**Hardcoded validation** is right for **binary, structural checks** where the answer is objectively true/false:
- Does the file exist?
- Is the target column in the feature frame? (yes/no)
- Are train and test groups disjoint? (yes/no)

These checks are cheap, deterministic, and have zero false positives. They should always run first and gate the more expensive LLM review.

**LLM quality review** is right for **judgment calls** that require domain reasoning:
- Is a max value of 5797 for Interest_Rate plausible?
- Should this column have been split on a delimiter?
- Does this encoding strategy make sense for the column's semantics?

These questions don't have hardcoded answers — they depend on understanding what the data means.

## Our architecture

```
execute → hardcoded validation → LLM quality review → train
              ↓ (fail)              ↓ (fail)
           repair ←←←←←←←←←←←← repair
```

Hardcoded runs first (fast, free, deterministic). If it fails, skip the LLM call entirely — no point reviewing quality when files don't exist. If structural validation passes, LLM review checks the harder questions.

Both feed into the same repair loop: hardcoded errors and LLM quality issues are merged into the validation report that the repair skill sees.

## The lesson

Don't choose between hardcoded and LLM validation — layer them. Use hardcoded checks as a fast gate for structural correctness, and LLM review for nuanced quality judgment that requires reasoning about the data's meaning.

The original design question ("should we use hardcoded validation or LLM review?") was a false dichotomy. The answer is: both, in sequence, with the cheap deterministic check first.
