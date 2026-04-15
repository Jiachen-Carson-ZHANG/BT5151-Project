"""Raw pipeline log parser and log-tail helpers for the Developer Trace tab.

All functions are pure (no Gradio dependencies) so they can be unit-tested
without a running Gradio server.
"""

import re
from pathlib import Path

# Regex patterns for log line parsing
_NODE_START_RE = re.compile(r">>> ([\w-]+)")
_LLM_CALLS_RE = re.compile(r"Total LLM calls:\s*(\d+)")
_TOTAL_TOKENS_RE = re.compile(r"Total tokens:\s*(\d+)")
_TOKEN_SUMMARY_START = "--- Token usage summary ---"


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


def _finalise_card(card: dict) -> None:
    """Trim summary_lines to 10 most informative entries."""
    card["summary_lines"] = card["summary_lines"][:10]


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


def build_trace_markdown(trace: dict) -> str:
    """Render a parse_stage_log result dict as a Markdown string.

    Suitable for display in a gr.Markdown component.
    """
    parts: list[str] = []
    rs = trace.get("run_summary", {})

    # Header
    log_path = rs.get("log_path", "unknown")
    parts.append(f"### Developer Trace — `{Path(log_path).name}`\n")

    if rs.get("error"):
        parts.append(f"> ⚠ {rs['error']}\n")
        return "\n".join(parts)

    llm_calls = rs.get("total_llm_calls", 0)
    total_tokens = rs.get("total_tokens", 0)
    parts.append(f"**LLM calls:** {llm_calls}  |  **Total tokens:** {total_tokens:,}\n")

    cards = trace.get("cards", [])
    if not cards:
        parts.append("_No node events found in log._\n")
        return "\n".join(parts)

    status_icons = {
        "pass": "✅",
        "warn": "⚠️",
        "error": "❌",
        "repair": "🔧",
    }

    for card in cards:
        icon = status_icons.get(card["status"], "•")
        parts.append(f"\n#### {icon} `{card['node_name']}`")

        if card["warning_lines"]:
            parts.append("\n**Warnings / Errors:**")
            for wl in card["warning_lines"][:5]:
                msg = _extract_message(wl)
                parts.append(f"- {msg}")

        if card["summary_lines"]:
            for sl in card["summary_lines"][:5]:
                parts.append(f"  {sl}")

    parts.append("")
    return "\n".join(parts)
