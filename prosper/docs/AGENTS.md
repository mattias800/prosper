# Technical documentation

This folder maps Prosper's architecture, subsystem contracts, title investigations and verification
workflows. Dated measurements and falsifications explain the evidence behind a decision; retain
that context when refreshing current-state sections.

The repository-root charter governs work here. Game trackers hold the live milestone and blocker
index, while `GAME_COMPAT_ORCHESTRATION.md` maps shared ownership and experiment scheduling.
User-facing compatibility rows and the screenshot blog live at the repository root. Source-level
API details belong beside their implementation; reproducible tools and input routes live in their
own tool or script folders.

## Layout

Top level holds only this file and `ROADMAP.md`. Everything else lives in a topic folder; put a new
doc in the folder whose question it answers, not beside whatever you last opened.

| Folder | What belongs there |
| --- | --- |
| `architecture/` | How the pieces fit: components, the host/platform seam, the target source tree, refactor plans |
| `gpu/` | AGC/PM4, the Vulkan executor, resource binding, tiling, recompiler, renderer design notes (dated `_YYYY_MM` design/measurement docs included) |
| `performance/` | Profiling method, performance roadmaps and the dated measurement passes |
| `subsystems/` | One reimplemented library or service per doc: audio, video decode, save data, input, the frontend app |
| `engines/` | Engine-wide bring-up shared across titles (Unreal, Unity) |
| `platforms/` | Windows / Linux / macOS / Android porting and release |
| `process/` | How we work: verification, debugging workflows, diagnostics policy, bug-hunt backlog, multi-lane orchestration, session start |
| `games/` | Per-title `*_STATUS.md` (each carries a `## Ruled out`), plus title handoffs and surveys |
| `games/messenger/` | The Messenger bring-up history (reconnaissance, cutscene, deserialisation, black-render investigations) |
| `evidence/`, `screenshots/` | Raw measurement output and committed captures cited by the docs above |

Cite docs by full path (`prosper/docs/gpu/GRAPHICS.md`); file names are unique repo-wide, so a bare
name is still greppable. `tools/docs/reorganize_docs.py` is the one-shot that performed the 2026-10
move; do not re-run it.
