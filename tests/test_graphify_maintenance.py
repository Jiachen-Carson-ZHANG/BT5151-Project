"""Tests for the shared Graphify maintenance helper."""

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parent.parent / "scripts" / "graphify_maintenance.py"
    spec = importlib.util.spec_from_file_location("graphify_maintenance", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_paths(monkeypatch, module, tmp_path):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "STATE_DIR", tmp_path / ".graphify-state")
    monkeypatch.setattr(module, "CHANGED_FILES", tmp_path / ".graphify-state" / "changed-files.txt")
    monkeypatch.setattr(module, "GRAPH_DIR", tmp_path / "graphify-out")
    monkeypatch.setattr(module, "SEMANTIC_MARKER", tmp_path / "graphify-out" / ".semantic_update_needed")
    monkeypatch.setattr(module, "CLEAN_REBUILD_MARKER", tmp_path / "graphify-out" / ".needs_clean_rebuild")


def test_code_only_flush_runs_ast_update(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')\n")
    calls = []
    monkeypatch.setattr(gm.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    gm.record_paths(["src/app.py"])
    result = gm.flush()

    assert result.code_paths == ("src/app.py",)
    assert calls
    assert calls[0][0][0] == ["graphify", "update", "."]
    assert not gm.SEMANTIC_MARKER.exists()
    assert not gm.CLEAN_REBUILD_MARKER.exists()


def test_semantic_change_marks_stale_without_running_graphify(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    (tmp_path / "docs" / "architecture").mkdir(parents=True)
    (tmp_path / "docs" / "architecture" / "current-state.md").write_text("# Current\n")
    calls = []
    monkeypatch.setattr(gm.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    gm.record_paths(["docs/architecture/current-state.md"])
    result = gm.flush()

    assert result.semantic_paths == ("docs/architecture/current-state.md",)
    assert gm.SEMANTIC_MARKER.exists()
    assert "docs/architecture/current-state.md" in gm.SEMANTIC_MARKER.read_text()
    assert calls == []


def test_implementation_log_only_does_nothing(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    (tmp_path / "docs" / "changes").mkdir(parents=True)
    (tmp_path / "docs" / "changes" / "implementation-log.md").write_text("# Log\n")
    calls = []
    monkeypatch.setattr(gm.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    gm.record_paths(["docs/changes/implementation-log.md"])
    result = gm.flush()

    assert result.ignored_paths == ("docs/changes/implementation-log.md",)
    assert not gm.SEMANTIC_MARKER.exists()
    assert not gm.CLEAN_REBUILD_MARKER.exists()
    assert calls == []


def test_graphifyignore_change_marks_clean_rebuild(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    (tmp_path / ".graphifyignore").write_text("tests/\n")
    gm.SEMANTIC_MARKER.parent.mkdir(parents=True)
    gm.SEMANTIC_MARKER.write_text("stale semantic marker\n")
    calls = []
    monkeypatch.setattr(gm.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    gm.record_paths([".graphifyignore"])
    result = gm.flush()

    assert result.clean_rebuild_paths == (".graphifyignore",)
    assert gm.CLEAN_REBUILD_MARKER.exists()
    assert not gm.SEMANTIC_MARKER.exists()
    assert ".graphifyignore" in gm.CLEAN_REBUILD_MARKER.read_text()
    assert calls == []


def test_hook_json_file_path_is_parsed(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "graph.py").write_text("VALUE = 1\n")
    payload = {"tool_input": {"file_path": str(tmp_path / "src" / "graph.py")}}

    paths = gm.paths_from_hook_stdin(json.dumps(payload))

    assert paths == ["src/graph.py"]


def test_missing_recorded_file_marks_clean_rebuild(tmp_path, monkeypatch):
    gm = _load_module()
    _patch_paths(monkeypatch, gm, tmp_path)
    calls = []
    monkeypatch.setattr(gm.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    gm.record_paths(["src/deleted.py"])
    result = gm.flush()

    assert result.clean_rebuild_paths == ("src/deleted.py",)
    assert result.missing_paths == ("src/deleted.py",)
    assert gm.CLEAN_REBUILD_MARKER.exists()
    assert calls == []
