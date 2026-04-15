# Feature engineering: codegen vs template

**Date:** 2026-04-09

## The question

Should the feature engineering node use free codegen (LLM generates code) or template code (hardcoded transforms)?

## What's templateable (formulaic, no reasoning needed)

- Drop constant features (nunique ≤ 1)
- Drop one of highly correlated pairs (r > 0.95)
- Log-transform highly skewed columns (skewness > threshold)

These are ~15 lines of deterministic code. No LLM needed.

## What needs reasoning (LLM adds value)

- **Which** columns to combine into ratios (debt-to-income, utilization-to-limit)
- **Which** columns have semantic meaning that suggests interaction
- **Whether** a log transform makes domain sense (log of a count vs log of a ratio)
- Identifying that two columns are semantically redundant (e.g., monthly salary and annual income)

## Why template alone is insufficient

A template that does "log-transform all columns with skewness > 2" will blindly transform things that shouldn't be transformed. The LLM can reason: "Annual_Income is right-skewed continuous revenue — log transform makes sense; Num_Bank_Accounts is a small count — log doesn't help here."

## Why pure codegen alone is risky

The LLM might skip the boring-but-important heuristic checks (constant features, correlation pruning) in favor of flashy domain features. Or it might not do them consistently.

## Recommendation: free codegen with heuristic rules in the skill prompt

The skill prompt should mandate:
1. Always check for and drop constant/near-constant features
2. Always check for and handle highly correlated pairs
3. Consider log transforms for right-skewed continuous columns
4. Consider domain-meaningful interaction features based on column semantics
5. Report what you did and why

This gives heuristic guarantees (correlation pruning always happens) plus LLM flexibility (domain-aware feature creation).

## Risk assessment

Feature engineering codegen is **lower risk** than data cleaning codegen because:
- You're adding/removing columns, not fixing messy raw data
- The input is already clean (post-preprocessing)
- Failures are less catastrophic — a bad feature is noise, not a crash
- The existing codegen loop (generate → inspect → execute → validate → repair) is already proven
