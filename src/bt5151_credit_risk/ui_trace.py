"""Trace artifact parsers and log-tail helpers for the Developer Trace tab.

All functions are pure (no Gradio dependencies) so they can be unit-tested
without a running Gradio server.
"""

import json
import re
import html
from pathlib import Path
from typing import Any

# Regex patterns for log line parsing
_NODE_START_RE = re.compile(r">>> ([\w-]+)")
_LLM_CALLS_RE = re.compile(r"Total LLM calls:\s*(\d+)")
_TOTAL_TOKENS_RE = re.compile(r"Total tokens:\s*(\d+)")
_TOKEN_SUMMARY_START = "--- Token usage summary ---"
_RUN_STAGE_START_RE = re.compile(r"=== Stage '([^']+)' \(stop_after=.*\) ===")
_RUN_STAGE_COMPLETE_RE = re.compile(r"=== Stage '([^']+)' completed in ([^=]+) ===")

# Main-path pipeline stage order — used for pending-node visualization.
# Conditional repair nodes are rendered from actual trace events instead of
# being pre-rendered as grey boxes, otherwise retry loops look like permanent
# bottom-of-pipeline stages.
_PIPELINE_STAGES = [
    "dataset-policy-spec",
    "exploratory-data-analysis",
    "generate-eda-hypotheses",
    "column-transform-spec",
    "generate-preprocessing-code",
    "inspect-preprocessing-code",
    "execute-generated-preprocessing",
    "validate-preprocessing-output",
    "review-preprocessing-quality",
    "generate-feature-engineering-code",
    "inspect-feature-engineering-code",
    "execute-feature-engineering",
    "validate-feature-engineering",
    "train-models",
    "evaluate-models",
    "training-diagnostics",
    "select-model",
    "global-xai",
    "local-xai",
    "shortcut-feature-audit",
    "interpret-global-xai",
    "interpret-local-xai",
    "package-analysis-bundle",
    "run-inference",
    "explain-risk",
]
_PIPELINE_STAGE_INDEX = {stage: idx for idx, stage in enumerate(_PIPELINE_STAGES)}
_CONDITIONAL_NEXT_STAGE = {
    "repair-preprocessing-code": "inspect-preprocessing-code",
    "repair-feature-engineering-code": "inspect-feature-engineering-code",
    "review-preprocessing-quality": "repair-preprocessing-code",
    "validate-feature-engineering": "repair-feature-engineering-code",
    "inspect-preprocessing-code": "repair-preprocessing-code",
    "inspect-feature-engineering-code": "repair-feature-engineering-code",
}
_TERMINAL_LIFECYCLE_EVENTS = {"run_failed", "run_complete", "cache_saved"}
_VISIBLE_LIFECYCLE_EVENTS = {"run_complete", "run_failed", "cache_saved"}

# Colour palette per status
_NODE_COLORS = {
    "pass":    ("#27ae60", "#ffffff"),   # green, white text
    "warn":    ("#f39c12", "#ffffff"),   # amber
    "error":   ("#e74c3c", "#ffffff"),   # red
    "repair":  ("#8e44ad", "#ffffff"),   # purple
    "running": ("#2980b9", "#ffffff"),   # blue — last seen node
    "pending": ("#ecf0f1", "#7f8c8d"),   # light grey
}

# LLM call line: detect the header and the prediction bullets
_LLM_CALL_RE = re.compile(r"LLM call \[([^\]]+)\]")
_PREDICTION_TAG_RE = re.compile(r"(\[(tested|supported|exploratory)\])")

# Bold the first "label:" on a line.  Two cases:
#   1. Special no-colon keywords: "validation FAILED", "validation passed"
#   2. Generic "word(s):" pattern — catches feature_frame:, policy spec keys:,
#      Customer_ID:, verdict:, role violation:, interactions_rationale:, etc.
_BOLD_LEAD_RE = re.compile(
    r"^(⚠\s+)?("
    r"validation FAILED|validation passed"  # no-colon keywords
    r"|[\w][\w _]{0,45}:"                  # generic label: (word chars, spaces, underscores)
    r")",
    re.IGNORECASE,
)

# Bold [content] bracket references selectively
_BRACKET_CONTENT_RE = re.compile(r"\[([^\]]+)\]")
_TIER_TAGS = frozenset({"tested", "supported", "exploratory"})
_IDENT_RE = re.compile(r"^[\w][\w\s_-]{0,39}$")


def _apply_inline_bold(line: str) -> str:
    """Bold diagnostic keywords and column/field references in a summary line."""
    m = _BOLD_LEAD_RE.match(line)
    if m:
        warn = m.group(1) or ""
        keyword = m.group(2)
        rest = line[m.end():]
        line = f"{warn}**{keyword}**{rest}"

    def _bold_bracket(bm: re.Match) -> str:
        content = bm.group(1)
        if content.lower() in _TIER_TAGS:
            return bm.group(0)  # tier tags handled as bullet prefixes
        if "'" in content:
            return bm.group(0)  # Python list repr like ['col1', 'col2'] — skip
        if "," in content and "/" not in content:
            return bm.group(0)  # comma-separated column list — skip
        if "/" in content:
            return f"**[{content}]**"  # severity tag like [critical/target_alignment]
        if _IDENT_RE.match(content):
            return f"**[{content}]**"  # identifier-like column/field ref like [Age]
        return bm.group(0)

    return _BRACKET_CONTENT_RE.sub(_bold_bracket, line)

# Full LLM call line in raw log files — used to enrich JSONL trace cards
_LOG_LLM_CALL_RE = re.compile(
    r"LLM call \[([^\]]+)\]\s+model=(\S+)\s+input_tokens=(\d+)\s+output_tokens=(\d+)\s+duration=(\S+)"
)


def parse_stage_log(path: "str | Path") -> dict:
    """Parse a pipeline stage log file into a structured trace dict.

    Returns:
        {
            "run_summary": {
                "log_path": str,
                "total_llm_calls": int,
                "total_tokens": int,
                "error": str | None,   # present only when file is missing/unreadable
            },
            "cards": [
                {
                    "node_name": str,
                    "status": "pass" | "warn" | "error" | "repair",
                    "summary_lines": list[str],
                    "warning_lines": list[str],
                    "raw_lines": list[str],
                }
            ],
        }
    """
    path = Path(path)
    run_summary: dict = {
        "log_path": str(path),
        "artifact_type": "stage_log",
        "total_llm_calls": 0,
        "total_tokens": 0,
    }
    cards: list[dict] = []

    if not path.is_file():
        run_summary["error"] = f"Log file not found: {path}"
        return {"run_summary": run_summary, "cards": cards}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        run_summary["error"] = f"Cannot read log file: {exc}"
        return {"run_summary": run_summary, "cards": cards}

    current_card: dict | None = None

    for line in lines:
        # Check for token usage summary fields
        m = _LLM_CALLS_RE.search(line)
        if m:
            run_summary["total_llm_calls"] = int(m.group(1))
            continue

        m = _TOTAL_TOKENS_RE.search(line)
        if m:
            run_summary["total_tokens"] = int(m.group(1))
            continue

        m = _RUN_STAGE_START_RE.search(line)
        if m:
            run_summary["stage"] = m.group(1)
            cards.append(
                _structured_event_to_card(
                    {
                        "event_type": "run_start",
                        "node": "__run__",
                        "status": "pass",
                        "stage": m.group(1),
                    }
                )
            )
            continue

        m = _RUN_STAGE_COMPLETE_RE.search(line)
        if m:
            if current_card is not None:
                _finalise_card(current_card)
                cards.append(current_card)
                current_card = None
            run_summary["stage"] = run_summary.get("stage") or m.group(1)
            cards.append(
                _structured_event_to_card(
                    {
                        "event_type": "run_complete",
                        "node": "__run__",
                        "status": "pass",
                        "stage": m.group(1),
                        "metrics": {"duration": m.group(2).strip()},
                    }
                )
            )
            continue

        # Check for new node start
        m = _NODE_START_RE.search(line)
        if m:
            if current_card is not None:
                _finalise_card(current_card)
                cards.append(current_card)
            node_name = m.group(1)
            current_card = {
                "node_name": node_name,
                "status": "pass",
                "summary_lines": [],
                "warning_lines": [],
                "raw_lines": [line],
            }
            # Nodes whose name contains "repair" are tagged as repair status
            if "repair" in node_name.lower():
                current_card["status"] = "repair"
            continue

        if current_card is not None:
            current_card["raw_lines"].append(line)
            level = _extract_level(line)
            if level == "ERROR":
                current_card["warning_lines"].append(line)
                current_card["status"] = "error"
            elif level == "WARNING":
                current_card["warning_lines"].append(line)
                if current_card["status"] not in ("error", "repair"):
                    current_card["status"] = "warn"
            else:
                msg = _extract_message(line)
                if msg:
                    current_card["summary_lines"].append(msg)

    # Flush last card
    if current_card is not None:
        _finalise_card(current_card)
        cards.append(current_card)

    return {"run_summary": run_summary, "cards": cards}


def parse_trace_artifact(path: "str | Path") -> dict:
    """Parse either a structured trace JSONL artifact or a raw stage log."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return parse_structured_trace_jsonl(path)
    return parse_stage_log(path)


def parse_live_trace_artifacts(log_path: "str | Path", trace_path: "str | Path | None") -> dict:
    """Parse a live raw log and overlay JSONL completion status when available.

    Raw stage logs are freshest during long-running nodes, but they only contain
    node-start markers. Structured JSONL is less verbose, but records true
    node-complete events. Combining them keeps the log content live while
    preventing the pipeline rail from showing an already-completed last raw node
    as still running.
    """
    raw_trace = parse_stage_log(log_path)
    if not trace_path:
        return raw_trace

    structured_trace = parse_structured_trace_jsonl(trace_path)
    completion_by_occurrence: dict[tuple[str, int], str] = {}
    structured_counts: dict[str, int] = {}
    for card in structured_trace.get("cards", []):
        name = str(card.get("node_name") or "")
        if not name or name.startswith("__") or name in _VISIBLE_LIFECYCLE_EVENTS:
            continue
        structured_counts[name] = structured_counts.get(name, 0) + 1
        completion_by_occurrence[(name, structured_counts[name])] = str(card.get("status") or "pass")

    raw_counts: dict[str, int] = {}
    for card in raw_trace.get("cards", []):
        name = str(card.get("node_name") or "")
        if not name or name.startswith("__") or name in _VISIBLE_LIFECYCLE_EVENTS:
            continue
        raw_counts[name] = raw_counts.get(name, 0) + 1
        status = completion_by_occurrence.get((name, raw_counts[name]))
        if status:
            card["status"] = status
            card["completed_by_trace"] = True

    summary = raw_trace.setdefault("run_summary", {})
    summary["live_completion_overlay"] = str(trace_path)
    return raw_trace


def parse_structured_trace_jsonl(path: "str | Path") -> dict:
    """Parse a structured trace JSONL file into the same card shape as logs."""
    path = Path(path)
    run_summary: dict[str, Any] = {
        "log_path": str(path),
        "artifact_type": "structured_trace",
        "total_llm_calls": 0,
        "total_tokens": 0,
        "total_events": 0,
    }
    cards: list[dict] = []

    if not path.is_file():
        run_summary["error"] = f"Trace file not found: {path}"
        return {"run_summary": run_summary, "cards": cards}

    try:
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        run_summary["error"] = f"Cannot read trace file: {exc}"
        return {"run_summary": run_summary, "cards": cards}

    events: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            run_summary["error"] = f"Invalid trace JSONL: {exc}"
            return {"run_summary": run_summary, "cards": []}
        if isinstance(event, dict):
            events.append(event)

    run_summary["total_events"] = len(events)

    companion_log_path: str | None = None
    for event in events:
        event_type = str(event.get("event_type") or "")
        if event.get("run_id") and not run_summary.get("run_id"):
            run_summary["run_id"] = event.get("run_id")
        if event.get("stage") and not run_summary.get("stage"):
            run_summary["stage"] = event.get("stage")

        if event_type == "run_start":
            run_summary["run_id"] = event.get("run_id", run_summary.get("run_id"))
            run_summary["stage"] = event.get("stage", run_summary.get("stage"))
            companion_log_path = (event.get("artifacts") or {}).get("log_path")
        if event_type in {"run_start", "run_complete", "run_failed", "cache_saved"} or event.get("node"):
            cards.append(_structured_event_to_card(event))

    # Enrich JSONL cards with content from the companion raw log.
    # JSONL node events have metrics:{} — all rich detail lives in the .log file.
    if companion_log_path:
        llm_calls = _parse_llm_calls_from_log(companion_log_path)
        node_content = _parse_node_content_from_log(companion_log_path)
        # Track occurrences to match the nth card to the nth LLM call for
        # nodes that run multiple times (e.g. repeated repair attempts).
        node_occurrences: dict[str, int] = {}
        total_input = total_output = 0
        for card in cards:
            node = card["node_name"]
            if node.startswith("__") or node in _VISIBLE_LIFECYCLE_EVENTS:
                continue
            calls = llm_calls.get(node, [])
            if calls:
                # LLM node: prepend call header + post-call content (hypotheses, etc.)
                idx = node_occurrences.get(node, 0)
                node_occurrences[node] = idx + 1
                call = calls[min(idx, len(calls) - 1)]
                meta = (f"model={call['model']} | input_tokens={call['input_tokens']} | "
                        f"output_tokens={call['output_tokens']} | duration={call['duration']}")
                llm_line = f"LLM call [{node}] {meta}"
                inject = [llm_line] + call.get("content_lines", [])
                for pos, l in enumerate(inject):
                    card["summary_lines"].insert(pos, l)
                total_input += int(call["input_tokens"])
                total_output += int(call["output_tokens"])
            else:
                # Non-LLM node (e.g. validate-preprocessing-output): prepend
                # general log content so violations and check results are visible.
                general = node_content.get(node, [])
                if general:
                    occ = node_occurrences.get(node, 0)
                    node_occurrences[node] = occ + 1
                    # Each run of a repeated node gets its own slice of content
                    # (content lines are not split per-occurrence, so we just
                    # show the full set for every card — repetition is harmless).
                    for pos, l in enumerate(general):
                        card["summary_lines"].insert(pos, l)
        if total_input or total_output:
            run_summary["total_tokens"] = total_input + total_output

    return {"run_summary": run_summary, "cards": cards}


def _finalise_card(card: dict) -> None:
    """No-op — summary_lines are no longer truncated here; rendering handles limits."""
    pass


def _structured_event_to_card(event: dict[str, Any]) -> dict:
    """Render one structured trace event into a card dict.

    Each section (event type, state keys, metrics, artifacts) gets its own
    header line and per-item bullets so the markdown renderer produces
    readable multi-line output instead of dense comma-separated blobs.

    State-key display strategy:
    - The metrics dict contains flattened dot-notation scalar values from the
      state update (e.g. "dataset_policy_spec.split_strategy.test_size": 0.2).
    - For each state_key, we pull matching metric values (prefix match) and
      show up to 3 representative values inline so the user can see what was
      actually written — not just the attribute name.
    - Complex objects (DataFrames, models, lists) have no scalar metrics and
      are labeled "(complex object)".
    """
    event_type = str(event.get("event_type") or "")
    raw_node = str(event.get("node") or "")
    if raw_node == "__run__" and event_type:
        node_name = event_type
    else:
        node_name = str(event.get("node") or event_type or "event")
    status = _normalize_status(str(event.get("status") or ""))
    summary_lines: list[str] = []
    warning_lines: list[str] = []

    if event_type:
        if event_type in {"run_start", "run_complete", "run_failed", "cache_saved"}:
            summary_lines.append(f"Lifecycle event: {event_type}")
        summary_lines.append(f"**Event type:** {event_type}")

    # ── State keys ────────────────────────────────────────────────────────────
    state_keys = [str(k) for k in (event.get("state_keys_written") or [])]
    metrics: dict = event.get("metrics") or {}

    # Group metrics by the state key they belong to (prefix match)
    key_metrics: dict[str, dict] = {}
    unmatched_metrics: dict = {}
    for mk, mv in metrics.items():
        matched = False
        for sk in state_keys:
            if mk == sk or mk.startswith(sk + "."):
                sub = mk[len(sk) + 1:] if mk != sk else mk
                key_metrics.setdefault(sk, {})[sub] = mv
                matched = True
                break
        if not matched:
            unmatched_metrics[mk] = mv

    if state_keys:
        summary_lines.append("**State keys written:**")
        for sk in state_keys:
            sk_metrics = key_metrics.get(sk, {})
            if sk_metrics:
                # Header bullet for the state key
                summary_lines.append(f"- `{sk}`")
                # Each scalar value on its own indented line
                preview_items = list(sk_metrics.items())[:4]
                for k, v in preview_items:
                    summary_lines.append(f"  - `{k}` = `{v}`")
                remaining = len(sk_metrics) - len(preview_items)
                if remaining > 0:
                    summary_lines.append(f"  - *+{remaining} more scalar values*")
            else:
                summary_lines.append(f"- `{sk}` — *(complex object)*")

    # ── Unmatched metrics (standalone scalars not tied to a state key) ─────────
    if unmatched_metrics:
        summary_lines.append("**Metrics:**")
        for mk, mv in list(unmatched_metrics.items())[:12]:
            summary_lines.append(f"- `{mk}` = {mv}")

    # ── Artifacts ─────────────────────────────────────────────────────────────
    artifacts: dict = event.get("artifacts") or {}
    if isinstance(artifacts, dict) and artifacts:
        summary_lines.append("**Artifacts:**")
        for ak, av in artifacts.items():
            summary_lines.append(f"- `{ak}` = `{av}`")

    warnings = event.get("warnings") or []
    warning_lines.extend(_stringify_collection(warnings))
    if event.get("error"):
        warning_lines.append(str(event.get("error")))

    raw_lines = [json.dumps(event, sort_keys=True, default=str)]
    return {
        "node_name": node_name,
        "status": status,
        "summary_lines": summary_lines,
        "warning_lines": warning_lines,
        "raw_lines": raw_lines,
    }


def _normalize_status(status: str) -> str:
    """Map structured event status values onto the markdown status palette."""
    if status in {"failed", "fail", "error"}:
        return "error"
    if status in {"warn", "warning"}:
        return "warn"
    if status == "repair":
        return "repair"
    return "pass"


def _stringify_collection(value: Any) -> list[str]:
    """Flatten warnings/errors collections into displayable strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "message" in value and isinstance(value["message"], str):
            return [value["message"]]
        return [json.dumps(value, sort_keys=True, default=str)]
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_stringify_collection(item))
        return items
    return [str(value)]


def _format_mapping_preview(value: dict[str, Any], limit: int = 4) -> str:
    """Compact dict preview for Markdown summaries."""
    parts: list[str] = []
    for idx, (key, item) in enumerate(value.items()):
        if idx >= limit:
            parts.append("...")
            break
        parts.append(f"{key}={item}")
    return ", ".join(parts)


def _extract_level(line: str) -> str | None:
    """Return log level token (DEBUG/INFO/WARNING/ERROR) from a log line, or None."""
    for level in ("ERROR", "WARNING", "INFO", "DEBUG"):
        if level in line:
            return level
    return None


def _extract_message(line: str) -> str:
    """Extract the message portion from a log line (everything after the logger name).

    Log format: "HH:MM:SS LEVEL  logger_name  message"
    Two splits (maxsplit=2) gives [timestamp+level, logger_name, message].
    maxsplit=3 would over-split messages that contain internal whitespace padding
    (e.g. "col ID                        action=drop" → "col ID" + "action=drop").
    """
    parts = re.split(r"\s{2,}", line.strip(), maxsplit=2)
    if len(parts) >= 3:
        return parts[2].strip()
    if len(parts) >= 1:
        return parts[-1].strip()
    return line.strip()


def read_log_tail(path: "str | Path", offset: int = 0) -> "tuple[str, int]":
    """Read new content from a log file starting at byte offset.

    Returns:
        (new_content: str, new_offset: int)
        new_content is the text appended since the last read.
        new_offset is the byte position after the last byte read.
        Returns ("", 0) if the file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        return "", 0
    try:
        with path.open("rb") as f:
            f.seek(offset)
            raw = f.read()
            new_offset = offset + len(raw)
        return raw.decode("utf-8", errors="replace"), new_offset
    except OSError:
        return "", offset


def summarize_log_card(lines: list[str]) -> dict:
    """Summarise a list of raw log lines into a card-like dict."""
    warnings = [l for l in lines if "WARNING" in l]
    errors = [l for l in lines if "ERROR" in l]
    status = "pass"
    if errors:
        status = "error"
    elif warnings:
        status = "warn"
    return {
        "status": status,
        "warning_lines": warnings,
        "error_lines": errors,
        "raw_lines": lines,
    }


def _format_llm_call_line(line: str) -> str:
    """Format a single LLM call header line into readable markdown.

    Input:  "LLM call [generate-eda-hypotheses] model=o4-mini input_tokens=3995 output_tokens=3452 duration=35.42s"
    Output: "**LLM call** `generate-eda-hypotheses`\\nmodel=o4-mini · input_tokens=3995 · ..."

    Prediction bullets ([tested]/[supported]/[exploratory]) arrive as separate
    summary_lines entries — they are handled by _render_summary_lines(), not here.
    """
    m = _LLM_CALL_RE.match(line.strip())
    if not m:
        return line.strip()

    node = m.group(1)
    meta = line[m.end():].strip()
    # Make metadata tokens easier to scan by separating with ·
    if "|" not in meta:
        meta = re.sub(r"\s+", "  |  ", meta, count=6)
    return f"**LLM call** `{node}`  \n{meta}"


def _render_summary_lines(summary_lines: list[str]) -> str:
    """Render a card's summary_lines into well-structured Markdown.

    Handles four patterns:
    - LLM call header lines → bold/code formatted, separated as own block
    - Section count lines ("tested predictions: N") → bold header
    - Prediction bullets ("[tested] ...", "[supported] ...", "[exploratory] ...") → bullet list
    - Everything else → plain paragraph, grouped with adjacent plain lines

    Items at the same structural level are joined with "  \\n" (Markdown soft break).
    Structurally different items are separated by "\\n\\n" (Markdown paragraph break).
    """
    _SECTION_HEADERS = ("tested predictions:", "supported conjectures:", "exploratory leads:")
    _BULLET_TAGS = ("[tested]", "[supported]", "[exploratory]")

    blocks: list[str] = []      # paragraph-level blocks
    plain_buffer: list[str] = []  # consecutive plain lines to merge
    bullet_buffer: list[str] = []  # consecutive bullet lines to merge
    llm_call_count = 0

    def flush_plain():
        if plain_buffer:
            blocks.append("  \n".join(plain_buffer))
            plain_buffer.clear()

    def flush_bullets():
        if bullet_buffer:
            blocks.append("\n".join(bullet_buffer))
            bullet_buffer.clear()

    for sl in summary_lines:
        sl = sl.strip()
        if not sl:
            continue

        if _LLM_CALL_RE.search(sl):
            flush_plain(); flush_bullets()
            llm_call_count += 1
            formatted = _format_llm_call_line(sl)
            if llm_call_count > 1:
                formatted = formatted.replace("**LLM call**", "**LLM call** _(retry)_", 1)
            blocks.append(formatted)

        elif sl.startswith("**") and sl.endswith("**") and len(sl) > 4:
            # Pre-formatted bold header (e.g. from structured event cards)
            flush_plain(); flush_bullets()
            blocks.append(sl)

        elif sl.startswith("**") and ":**" in sl:
            # Bold header with trailing content, e.g. "**Event type:** node_complete"
            flush_plain(); flush_bullets()
            blocks.append(sl)

        elif sl.startswith("- ") or sl.startswith("  - "):
            # Pre-formatted bullet or nested bullet (e.g. from structured event cards)
            flush_plain()
            bullet_buffer.append(sl)

        elif any(sl.startswith(h) for h in _SECTION_HEADERS):
            flush_plain(); flush_bullets()
            blocks.append(f"**{sl}**")

        elif any(sl.startswith(tag) for tag in _BULLET_TAGS):
            flush_plain()
            # Bold the tier tag prefix while keeping the rest of the line
            for tag in _BULLET_TAGS:
                if sl.startswith(tag):
                    sl = f"**{tag}**" + sl[len(tag):]
                    break
            bullet_buffer.append(f"- {sl}")

        else:
            flush_bullets()
            plain_buffer.append(_apply_inline_bold(sl))

    flush_plain()
    flush_bullets()
    return "\n\n".join(blocks)


def build_pipeline_html(trace: dict) -> str:
    """Build an HTML node-edge pipeline diagram from a trace dict.

    Actual trace events are rendered in execution order. This keeps conditional
    repair loops beside the stage that triggered them instead of appending
    unknown retry nodes at the bottom of the diagram. Pending nodes are limited
    to the remaining main-path stages.
    """
    cards = trace.get("cards", [])
    items = _build_pipeline_items(cards)

    boxes: list[str] = []
    for item in items:
        stage = item["node_name"]
        status = item["status"]
        bg, fg = _NODE_COLORS.get(status, _NODE_COLORS["pending"])
        icon = {"pass": "✅", "warn": "⚠️", "error": "❌", "repair": "🔧",
                "running": "⟳", "pending": "○"}.get(status, "•")
        attempt = item.get("attempt")
        suffix = f" #{attempt}" if isinstance(attempt, int) and attempt > 1 else ""
        label = f"{stage}{suffix}".replace("_", "-").replace("-", "‑")  # non-breaking hyphen for compact labels
        safe_label = html.escape(label)
        safe_title = html.escape(stage)

        pulse = ""
        if status == "running":
            pulse = " class=\"pulsing\""

        boxes.append(
            f'<div{pulse} title="{safe_title}" style="'
            f'background:{bg};color:{fg};'
            f'border-radius:9px;padding:10px 12px;margin:4px 0;'
            f'min-height:30px;display:flex;align-items:center;gap:7px;'
            f'font-size:12px;line-height:1.25;font-family:monospace;font-weight:650;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'max-width:240px;box-sizing:border-box;'
            f'">{icon} <span style="overflow:hidden;text-overflow:ellipsis;">{safe_label}</span></div>'
        )
        boxes.append(
            '<div style="text-align:center;color:#bdc3c7;font-size:16px;line-height:24px;height:24px;">│</div>'
        )

    # Remove trailing connector
    if boxes and boxes[-1].startswith('<div style="text-align'):
        boxes.pop()

    css = """
<style>
@keyframes pulse-border {
  0%,100% { opacity:1; }
  50%      { opacity:0.6; }
}
.pulsing { animation: pulse-border 1.2s ease-in-out infinite; }
</style>"""

    return css + '<div style="padding:8px 4px;">' + "\n".join(boxes) + "</div>"


def _build_pipeline_items(cards: list[dict]) -> list[dict]:
    """Return sidebar items in real execution order plus main-path pending nodes."""
    items: list[dict] = []
    occurrence_counts: dict[str, int] = {}
    terminal_seen = False
    last_main_idx = -1
    last_actual: dict | None = None

    for card in cards:
        name = str(card.get("node_name") or "")
        if name.startswith("__"):
            continue

        if name in _VISIBLE_LIFECYCLE_EVENTS:
            terminal_seen = name in _TERMINAL_LIFECYCLE_EVENTS or terminal_seen
        elif name in _PIPELINE_STAGE_INDEX:
            last_main_idx = max(last_main_idx, _PIPELINE_STAGE_INDEX[name])

        occurrence_counts[name] = occurrence_counts.get(name, 0) + 1
        item = {
            "node_name": name,
            "status": card.get("status", "pending"),
            "attempt": occurrence_counts[name],
            "completed_by_trace": bool(card.get("completed_by_trace")),
        }
        items.append(item)
        last_actual = item

    if not terminal_seen:
        if (
            last_actual
            and last_actual.get("status") == "pass"
            and last_actual.get("node_name") != "run_start"
            and not last_actual.get("completed_by_trace")
        ):
            last_actual["status"] = "running"
        for pending_stage in _pending_stages_after(last_actual, last_main_idx):
            items.append({"node_name": pending_stage, "status": "pending"})

    if not items:
        items = [{"node_name": stage, "status": "pending"} for stage in _PIPELINE_STAGES]

    return items


def _pending_stages_after(last_actual: dict | None, last_main_idx: int) -> list[str]:
    """Compute pending main-path stages after the latest real event."""
    if not last_actual:
        return list(_PIPELINE_STAGES)

    last_name = str(last_actual.get("node_name") or "")
    last_status = str(last_actual.get("status") or "")

    if last_name == "run_start":
        return list(_PIPELINE_STAGES)

    if last_status == "error" and last_name in _CONDITIONAL_NEXT_STAGE:
        return [_CONDITIONAL_NEXT_STAGE[last_name]]

    if last_name in {"repair-preprocessing-code", "repair-feature-engineering-code"}:
        return [_CONDITIONAL_NEXT_STAGE[last_name]]

    if last_main_idx < 0:
        return []

    return _PIPELINE_STAGES[last_main_idx + 1:]


def _parse_node_content_from_log(log_path: "str | Path") -> dict[str, list[str]]:
    """Extract all logged content lines per node from a companion raw stage log.

    Unlike _parse_llm_calls_from_log (which is scoped to LLM call output),
    this captures every log line for every node — including WARNING lines for
    validation failures, role violations, invariant checks, etc.  Used to
    enrich JSONL cards for nodes that make no LLM call (e.g. validate-preprocessing-output).

    Returns {node_name: [message_line, ...]}, capped at 30 lines per node.
    """
    path = Path(log_path)
    if not path.is_file():
        return {}
    result: dict[str, list[str]] = {}
    current_node: str | None = None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _NODE_START_RE.search(line)
            if m:
                current_node = m.group(1)
                continue
            if current_node is None:
                continue
            bucket = result.setdefault(current_node, [])
            if len(bucket) >= 30:
                continue
            level = _extract_level(line)
            msg = _extract_message(line)
            if not msg:
                continue
            # Prefix WARNING/ERROR lines so the renderer can highlight them
            if level in ("ERROR", "WARNING"):
                bucket.append(f"⚠ {msg}")
            else:
                bucket.append(msg)
    except OSError:
        pass
    return result


def _parse_llm_calls_from_log(log_path: "str | Path") -> dict[str, list[dict]]:
    """Extract per-node LLM call metadata AND content lines from a companion raw log.

    For each LLM call, captures:
      - model, input_tokens, output_tokens, duration
      - content_lines: all log message lines that follow the LLM call line until
        the next node start (>>>) or next LLM call — includes hypothesis bullets,
        section counts, and any other structured output the node logged.

    Returns {node_name: [{model, ..., content_lines: [str, ...]}, ...]}.
    Multiple calls per node (e.g. escalated repairs) are preserved in order.
    """
    path = Path(log_path)
    if not path.is_file():
        return {}
    result: dict[str, list[dict]] = {}
    current_call_node: str | None = None
    current_call_entry: dict | None = None

    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            # New node start resets content collection
            if _NODE_START_RE.search(line):
                current_call_node = None
                current_call_entry = None
                continue

            m = _LOG_LLM_CALL_RE.search(line)
            if m:
                node = m.group(1)
                entry: dict = {
                    "model": m.group(2),
                    "input_tokens": m.group(3),
                    "output_tokens": m.group(4),
                    "duration": m.group(5),
                    "content_lines": [],
                }
                result.setdefault(node, []).append(entry)
                current_call_node = node
                current_call_entry = entry
                continue

            # Collect content lines that follow the LLM call (hypothesis bullets, etc.)
            if current_call_entry is not None and len(current_call_entry["content_lines"]) < 40:
                msg = _extract_message(line)
                if msg:
                    current_call_entry["content_lines"].append(msg)
    except OSError:
        pass
    return result


def list_available_logs(log_dir: "str | Path") -> list[str]:
    """Return one preferred artifact per run for historical inspection.

    When both a structured JSONL trace and a raw stage log exist for the same
    run, prefer the JSONL trace. The returned list is ordered by actual file
    recency across artifact types so recent raw logs are not buried under older
    JSONLs.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    jsonls = list(log_dir.glob("trace_events_*.jsonl"))
    logs = list(log_dir.glob("stage_full_*.log"))

    traced_run_ids = {p.stem.replace("trace_events_", "", 1) for p in jsonls}
    preferred = list(jsonls) + [
        log for log in logs
        if log.stem.replace("stage_full_", "", 1) not in traced_run_ids
    ]
    preferred.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in preferred[:100]]


def build_trace_markdown(trace: dict) -> str:
    """Render a parse_stage_log/parse_structured_trace result as Markdown.

    Each top-level item is separated by a blank line (\\n\\n) so Markdown
    renders them as distinct paragraphs rather than collapsing into one line.
    """
    rs = trace.get("run_summary", {})
    log_path = rs.get("log_path", "unknown")
    artifact_type = rs.get("artifact_type", "stage_log")

    # ── Header block ──────────────────────────────────────────────────────────
    header_lines: list[str] = [f"### Developer Trace — `{Path(log_path).name}`"]
    if artifact_type == "structured_trace":
        meta_parts = []
        if rs.get("run_id"):
            meta_parts.append(f"**Run ID:** `{rs['run_id']}`")
        if rs.get("stage"):
            meta_parts.append(f"**Stage:** `{rs['stage']}`")
        if rs.get("total_events") is not None:
            meta_parts.append(f"**Events:** {rs['total_events']}")
        meta_parts.append("**Source:** Structured trace JSONL")
        header_lines.append("  \n".join(meta_parts))

    if rs.get("error"):
        header_lines.append(f"> ⚠ {rs['error']}")
        return "\n\n".join(header_lines)

    llm_calls = rs.get("total_llm_calls", 0)
    total_tokens = rs.get("total_tokens", 0)
    if llm_calls or total_tokens:
        header_lines.append(f"**LLM calls:** {llm_calls}  ·  **Total tokens:** {total_tokens:,}")

    # ── Cards ─────────────────────────────────────────────────────────────────
    cards = trace.get("cards", [])
    if not cards:
        header_lines.append("_No node events found in log._")
        return "\n\n".join(header_lines)

    status_icons = {"pass": "✅", "warn": "⚠️", "error": "❌", "repair": "🔧"}

    card_sections: list[str] = []
    for card in cards:
        icon = status_icons.get(card["status"], "•")
        card_blocks: list[str] = [f"#### {icon} `{card['node_name']}`"]

        if card["warning_lines"]:
            warn_lines = ["**Warnings / Errors:**"]
            for wl in card["warning_lines"][:5]:
                if isinstance(wl, str) and wl.startswith("{"):
                    warn_lines.append(f"- `{wl}`")
                else:
                    msg = _extract_message(wl)
                    warn_lines.append(f"- {msg}")
            card_blocks.append("\n".join(warn_lines))

        if card["summary_lines"]:
            # Injected log lines (LLM call header + hypothesis bullets, or validation
            # log lines) are prepended at the front.  Structured event metadata
            # (Event type, State keys written) follows.  Split them so structured
            # metadata always shows in full while injected content is capped.
            _VERBOSE_NODES = {"exploratory-data-analysis", "generate-eda-hypotheses"}
            injected: list[str] = []
            structured: list[str] = []
            in_structured = False
            for sl in card["summary_lines"]:
                if not in_structured and (
                    sl.startswith("Lifecycle event:")
                    or sl.startswith("**Event type:")
                    or sl.startswith("**State keys")
                    or sl.startswith("**Metrics:")
                    or sl.startswith("**Artifacts:")
                ):
                    in_structured = True
                (structured if in_structured else injected).append(sl)

            parts: list[str] = []
            if injected:
                cap = 10 if card["node_name"] in _VERBOSE_NODES else 6
                rendered_inj = _render_summary_lines(injected[:cap])
                if rendered_inj:
                    extra = len(injected) - cap
                    if extra > 0:
                        rendered_inj += f"\n\n_+{extra} more_"
                    parts.append(rendered_inj)
            if structured:
                rendered_str = _render_summary_lines(structured)
                if rendered_str:
                    parts.append(rendered_str)
            if parts:
                card_blocks.append("\n\n".join(parts))

        card_sections.append("\n\n".join(card_blocks))

    sections = ["\n\n".join(header_lines)] + card_sections
    return "\n\n---\n\n".join(sections)
