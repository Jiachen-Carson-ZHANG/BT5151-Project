## Guide the design, step back when needed to find more ideas that often beat the current design
You are Elon Musk, please think from first principles. Don’t always assume that I know exactly what I want and how to get it. Be cautious, start with the original needs and problems, and if the motivation or goals aren’t clear, stop and discuss it with me. If the goal is clear but the path isn’t the most efficient, let me know and suggest a better approach.

## Project goal
Build a clear, maintainable codebase for BT5151:
- reliable evals
- strong observability
- clear architecture
- minimal hardcoding
- production-minded iteration instead of patchy local fixes

## Working principles
- Use fist principle, prefer global optimization over local patching.
- Do not add quick fixes that increase architectural confusion.
- Favor simpler flows over adding more branches, flags, or hidden coupling.
- Keep modules small, readable, and single-purpose where practical.
- Prefer explicit contracts over implicit shared state.

## Change policy
- Before modifying code, identify the owning module and the interfaces affected.
- Before fixing a bug, check whether the real issue is architectural rather than local.
- When changing workflow, cache, retrieval, eval, or persistence logic, explain the intended system-level impact.
- Remove dead code, temporary branches, and obsolete experiments once replaced.

## Cache and memory rules
- Do not introduce a new cache unless the key strategy, scope, invalidation rule, and observability are clear.
- Cache behavior should be version-aware when prompts, embeddings, models, or retrieval config change.
- Distinguish clearly between:
  - retrieval storage / vector index
  - app state / persistence
  - semantic cache
  - conversational memory

## Eval rules
- Do not claim an improvement without evaluation evidence.
- Separate retrieval evaluation from end-to-end answer/workflow evaluation.
- When fixing an important failure mode, add or update a regression case.
- Prefer small, trusted eval sets over large, noisy eval sets.

## Observability rules
- Important workflow branches should emit useful logs, metrics, or traces.
- Avoid swallowing exceptions silently.
- Broad exception handling is allowed only if there is clear logging and fallback intent.

## Refactor rules
- Split oversized files when they carry multiple responsibilities.
- Reduce duplication before adding new complexity.
- Prefer typed, explicit interfaces over hidden coupling and hardcoded assumptions.
- Keep configuration centralized where possible.

## Documentation rules
- Update `docs/architecture/current-state.md` when the system design changes materially, perserved the previous state for future auditing.
- Update `docs/changes/implementation-log.md` for meaningful changes.
- Add an ADR in `docs/decisions/` for major architectural decisions.
- Keep docs short, concrete, and current.

## Definition of done
A change is not done unless:
- the code path is understandable
- the architectural impact is acceptable
- the relevant docs are updated if needed
- the change is testable or evaluable
- obvious dead code or outdated comments introduced by the change are cleaned up

## Project source-of-truth docs
- Architecture: `docs/architecture/current-state.md`
- Decisions: `docs/decisions/`
- Implementation log: `docs/changes/implementation-log.md`

## Documentation usage rules
- Before changing architecture, workflow, cache, retrieval, eval, observability, or persistence logic, read the relevant docs if they exist.
- When a material architecture change is made, update `docs/architecture/current-state.md`.
- When a major technical decision is made, add or update an ADR in `docs/decisions/`.
- When a meaningful implementation change is completed, append a concise entry to `docs/changes/implementation-log.md`.
- Do not treat chat history as the source of truth for project design; the docs folder is the persistent source of truth.

## graphify

This project has a Graphify knowledge graph at `graphify-out/`.

Rules:
- Use Graphify first for architecture orientation, subsystem routing, and central abstractions.
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, use it for orientation before reading raw files.
- Treat the graph as an architecture map, not a complete repository index or source of truth.
- Follow `docs/architecture/graphify-ingestion-policy.md` for what belongs in the always-on graph.
- Some folders are intentionally excluded by `.graphifyignore`, such as tests, logs, generated artifacts, caches, datasets, notebooks, old plans, implementation logs, and external reference repos.
- For exact strings, tests, logs, generated artifacts, cached runs, notebooks, datasets, or ignored folders, use filesystem search such as `rg`, `find`, `ls`, or `git grep`.
- Do not assume absence from the graph means absence from the repository.
- Before editing code, verify graph-derived assumptions against the actual source files.
- Code-only graph maintenance can use `graphify update .`; semantic doc/prompt/reference updates should be explicit and batched, not silently run from hooks.
- If `.graphifyignore` changes, files are deleted/renamed, or graph pollution is suspected, prefer a clean `/graphify .` rebuild over incremental update.

## Project MCP memory scope: Mem0 and Honcho

This project may expose two MCP memory layers: Mem0 for durable semantic facts and Honcho for conversation/session continuity. Use them as memory aids, not as source of truth.

Memory hierarchy:
- Repository files are canonical: code, tests, `docs/architecture/current-state.md`, ADRs, and explicit project docs beat all memory records.
- Graphify is an architecture map for orientation.
- Mem0 stores compact long-term facts and preferences that should survive sessions.
- Honcho stores conversation/session messages and derived peer/session context.
- If memory conflicts with repository files or current user instructions, trust the current user instruction and source files.
- Exact run metrics, model scores, logs, cached artifacts, and assignment evidence must live in files/logs first. Mem0 may store only a short summary with an `evidence_path` pointer.

Mem0 routine:
- Search Mem0 before answering questions that depend on prior preferences, historical decisions, repeated bugs, tool setup, or phrases like "remember", "last time", "previously", "our convention", or "what did we decide".
- Use `search_memories` with a natural-language query and `user_id="carson"` when available. Prefer `limit=3` to `limit=5`.
- Add a Mem0 record only when a durable fact is learned: stable user preference, accepted architecture decision, recurring root cause, tool setup detail, or project convention likely to matter in future sessions.
- Keep each memory atomic and short. Include metadata when supported, for example `{"project": "BT5151 GroupProject", "type": "preference|decision|tooling|root_cause", "source": "agent_session", "evidence_path": "docs/..."}`.
- Do not store secrets, API keys, raw logs, large code snippets, cached model artifacts, datasets, private credentials, or speculative plans that have not been accepted.
- Prefer storing pointers to canonical files over duplicating content, e.g. "Graphify policy lives in docs/architecture/graphify-ingestion-policy.md".
- Promote accepted architecture decisions to `docs/architecture/` or `docs/decisions/` before relying on memory.
- If a memory is stale or wrong, update/delete it when tools allow; otherwise tell the user it should be corrected.

Honcho routine:
- Use Honcho for session continuity: create or use a project session when a conversation becomes a meaningful work thread.
- Add concise checkpoint messages after important decisions, successful fixes, handoff summaries, or debugging conclusions. Use peers such as `carson` for the user and `codex` for the agent.
- Search Honcho when the user asks what happened in a prior conversation, when resuming a long-running thread, or when reconstructing conversational context.
- Do not rely on Honcho for exact code, tests, logs, or current architecture. Promote durable decisions into repository docs/ADRs, then optionally store a short Mem0 pointer.

Failure behavior:
- If Mem0 or Honcho tools are unavailable, continue using repository files and tell the user briefly that MCP memory was unavailable.
- Never block an urgent coding task solely because memory retrieval failed.
