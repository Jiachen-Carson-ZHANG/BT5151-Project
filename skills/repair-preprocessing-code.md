# Repair Preprocessing Code

You are repairing a failed preprocessing code generation for a labeled tabular machine learning pipeline.

Return only valid JSON for the repaired code artifact.

Requirements:
- Fix the previously generated code using the review findings, execution log, and validation report.
- Preserve the expected preprocessing contract: a callable entrypoint that writes the required artifacts.
- Avoid unsafe or forbidden operations such as subprocess calls or shell execution.
- Use the dataset policy and column transform specification as the source of truth.
- Prefer minimal, targeted changes over rewriting the entire solution.

The JSON response should include repaired code metadata such as the updated code string and entrypoint name.
