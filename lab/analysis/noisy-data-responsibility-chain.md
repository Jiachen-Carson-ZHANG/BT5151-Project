# Noisy Data: Where Should It Be Cleaned?

**Date:** 2026-04-09

## Why noise survives preprocessing

The current pipeline has multiple cleaning layers, but noise still gets through:

1. **column-transform-spec only sees 5 sample rows.** Garbage values like `_______` or `!@9#%8` might not appear in those 5 rows. The LLM literally cannot know they exist.
2. **The generate-preprocessing-code follows the spec.** If the spec doesn't mention garbage values, the code won't handle them.
3. **The audit catches encoding artifacts post-hoc.** `Payment_Behaviour_!@9#%8` gets flagged as a garbage category in the feature frame — but this is a late discovery, triggering a repair cycle.

## Responsibility chain

| Node | What it should catch | Current gap |
|------|---------------------|-------------|
| **column-transform-spec** | Dirty patterns visible in sample data | Only sees 5 rows; misses rare garbage |
| **generate-preprocessing-code** | Implement cleaning per spec | Can only clean what spec tells it to |
| **audit-preprocessing** | Catch what slipped through | Catches encoding artifacts but takes 2-3 rounds |

## The root cause: insufficient data visibility at spec time

The column-transform-spec LLM makes cleaning decisions with very limited information:
- 5 sample rows (may not contain garbage values)
- Column dtype and nunique (doesn't show value distribution)

**What's missing:** The top-N most frequent values per column, value distribution shape, and example outliers.

## Proposed fix: enrich the column-transform-spec payload

Add to the payload sent to the column-transform-spec LLM:
1. **Top 10 most frequent values** per categorical column (catches `_______`, `!@9#%8`)
2. **Min/max/mean/std** per numeric column (catches -500 Age, 5797% Interest_Rate)
3. **Example outlier values** (1st and 99th percentile values)
4. Increase sample_rows from 5 to 10

This costs marginally more input tokens but dramatically improves first-pass cleaning quality. The audit should catch fewer issues, meaning fewer repair rounds, meaning faster convergence and lower total token cost.

## Impact on pipeline

This is a change to `generate_column_transform_spec()` in `preprocess.py` and the `column-transform-spec` skill prompt. No new nodes needed — just richer input data at an existing decision point.

## Secondary option: hardcoded pre-filter

Before the LLM even sees the data, apply deterministic rules:
- Replace any value matching known garbage patterns (all underscores, special character sequences) with NaN
- Clip numeric columns to percentile-based bounds (1st/99th)

This is the "prompt engineering vs hardcoded enforcement" lesson from our earlier analysis — hardcoded rules catch systematic patterns the LLM might miss. Layer both.

## Update (2026-04-10): Enriched payload + reasoning model largely solves this

The enriched payload (top-10 values, min/max/mean/p1/p99) was implemented in Run 007-008. Combined with EDA insights (Run 009) and o4-mini reasoning model for column-transform-spec, first-pass spec quality improved dramatically: concrete domain-plausible clip bounds, explicit delimiter artifact handling, stable encoding choices. The quality reviewer should now catch fewer issues per round, reducing repair cycles.

The remaining gap: the LLM still doesn't see the actual percentile values for *categorical* columns (only top-10 values). If garbage values fall outside the top-10, they survive. This is acceptable — the quality auditor catches encoding artifacts post-hoc.
