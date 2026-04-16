"""Trace artifact parsers and log-tail helpers for the Developer Trace tab.

All functions are pure (no Gradio dependencies) so they can be unit-tested
without a running Gradio server.
"""

import json
import re
from pathlib import Path
from typing import Any

# Regex patterns for log line parsing
_NODE_START_RE = re.compile(r">>> ([\w-]+)")
_LLM_CALLS_RE = re.compile(r"Total LLM calls:\s*(\d+)")
_TOTAL_TOKENS_RE = re.compile(r"Total tokens:\s*(\d+)")
_TOKEN_SUMMARY_START = "--- Token usage summary ---"

# Canonical pipeline stage order — used for the node-edge visualization.
# Names must match the node names emitted by >>> markers in the log.
_PIPELINE_STAGES = [
    "dataset-policy-spec",
    "exploratory-data-analysis",
    "generate-eda-hypotheses",
    "column-transform-spec",
    "generate-preprocessing-code",
    "repair-preprocessing-code",
    "preprocess",
    "generate-feature-engineering-code",
    "repair-feature-engineering-code",
    "feature-engineering",
    "train-models",
    "evaluate-models",
    "training-diagnostics",
    "select-model",
    "global-xai",
    "local-xai",
    "package-analysis-bundle",
    "run-inference",
    "explain-risk",
    "recommend-action",
]

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

    for event in events:
        event_type = str(event.get("event_type") or "")
        if event.get("run_id") and not run_summary.get("run_id"):
            run_summary["run_id"] = event.get("run_id")
        if event.get("stage") and not run_summary.get("stage"):
            run_summary["stage"] = event.get("stage")

        if event_type == "run_start":
            run_summary["run_id"] = event.get("run_id", run_summary.get("run_id"))
            run_summary["stage"] = event.get("stage", run_summary.get("stage"))
        if event_type in {"run_start", "run_complete", "run_failed", "cache_saved"} or event.get("node"):
            cards.append(_structured_event_to_card(event))

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
    node_name = str(event.get("node") or event.get("event_type") or "event")
    status = _normalize_status(str(event.get("status") or ""))
    summary_lines: list[str] = []
    warning_lines: list[str] = []

    event_type = event.get("event_type")
    if event_type:
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
    """Extract the message portion from a log line (everything after the logger name)."""
    # Log format: "HH:MM:SS LEVEL logger_name  message"
    # We split on two or more spaces following the logger name part
    parts = re.split(r"\s{2,}", line.strip(), maxsplit=3)
    if len(parts) >= 4:
        return parts[3].strip()
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
    meta = re.sub(r"\s+", " · ", meta, count=6)
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
            blocks.append(_format_llm_call_line(sl))

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
            bullet_buffer.append(f"- {sl}")

        else:
            flush_bullets()
            plain_buffer.append(sl)

    flush_plain()
    flush_bullets()
    return "\n\n".join(blocks)


def build_pipeline_html(trace: dict) -> str:
    """Build an HTML node-edge pipeline diagram from a trace dict.

    Completed nodes are coloured by status. The last completed node is
    marked as "running" (blue) if the run is still live. Nodes not yet
    seen are shown as grey pending boxes.
    """
    cards = trace.get("cards", [])

    # Map node_name → status for nodes seen in the trace
    seen: dict[str, str] = {}
    for card in cards:
        name = card["node_name"]
        status = card["status"]
        # Keep worst status if node appears multiple times (e.g. repair loops)
        priority = {"error": 4, "warn": 3, "repair": 2, "pass": 1, "pending": 0, "running": 0}
        if name not in seen or priority.get(status, 0) > priority.get(seen[name], 0):
            seen[name] = status

    # Build full ordered stage list — canonical stages first, then any extra nodes from log
    canonical = list(_PIPELINE_STAGES)
    extras = [c["node_name"] for c in cards if c["node_name"] not in canonical
              and not c["node_name"].startswith("__")]
    all_stages = canonical + [e for e in extras if e not in canonical]

    # Determine "running" node: last seen node that completed (blue highlight)
    completed_in_order = [s for s in all_stages if s in seen]
    if completed_in_order:
        last_completed = completed_in_order[-1]
        # Only mark as running if it passed (error/warn are terminal colours)
        if seen.get(last_completed) == "pass":
            seen[last_completed] = "running"

    boxes: list[str] = []
    for stage in all_stages:
        status = seen.get(stage, "pending")
        bg, fg = _NODE_COLORS.get(status, _NODE_COLORS["pending"])
        icon = {"pass": "✅", "warn": "⚠️", "error": "❌", "repair": "🔧",
                "running": "⟳", "pending": "○"}.get(status, "•")
        label = stage.replace("-", "‑")  # non-breaking hyphen so long names don't wrap badly

        pulse = ""
        if status == "running":
            pulse = " class=\"pulsing\""

        boxes.append(
            f'<div{pulse} style="'
            f'background:{bg};color:{fg};'
            f'border-radius:6px;padding:5px 8px;margin:2px 0;'
            f'font-size:11px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'max-width:180px;'
            f'">{icon} {label}</div>'
        )
        boxes.append(
            '<div style="text-align:center;color:#bdc3c7;font-size:10px;line-height:1.2;">│</div>'
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


def list_available_logs(log_dir: "str | Path") -> list[str]:
    """Return log file names available for historical inspection, newest first.

    Includes structured JSONL traces (preferred) and raw stage logs.
    Filenames only — the caller resolves full paths.
    """
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    jsonls = sorted(log_dir.glob("trace_events_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    logs = sorted(log_dir.glob("stage_full_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    # JSONL first (richer), then raw logs; cap at 30 total
    all_files = list(jsonls) + [l for l in logs if l.stem not in {j.stem.replace("trace_events_", "stage_full_") for j in jsonls}]
    return [p.name for p in all_files[:30]]


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
            rendered = _render_summary_lines(card["summary_lines"][:10])
            if rendered:
                card_blocks.append(rendered)

        card_sections.append("\n\n".join(card_blocks))

    sections = ["\n\n".join(header_lines)] + card_sections
    return "\n\n---\n\n".join(sections)
