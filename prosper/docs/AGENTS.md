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
| `spec/` | The binding architecture rules, each with an ID, a status and the instrument that enforces it. Rules only; no measurements or narrative |
| `adr/` | Architecture decision records: why each rule exists and who accepted it. Accepted ADRs are frozen and superseded, never edited |
| `architecture/` | How the pieces fit: components, the host/platform seam, the target source tree, refactor plans, and `DESIGN_DOC_TEMPLATE.md`, the section shape for a new subsystem design doc |
| `gpu/` | AGC/PM4, the Vulkan executor, resource binding, tiling, recompiler, renderer design notes (dated `_YYYY_MM` design/measurement docs included) |
| `performance/` | Profiling method, performance roadmaps and the dated measurement passes |
| `subsystems/` | One reimplemented library or service per doc: audio, video decode, save data, input, the frontend app |
| `engines/` | Engine-wide bring-up shared across titles (Unreal, Unity) |
| `platforms/` | Windows / Linux / macOS / Android porting and release |
| `process/` | How we work: verification, debugging workflows, diagnostics policy, bug-hunt backlog, multi-lane orchestration, session start |
| `games/` | Per-title `*_STATUS.md` (each carries a `## Ruled out`), plus title handoffs and surveys |
| `games/messenger/` | The Messenger bring-up history (reconnaissance, cutscene, deserialisation, black-render investigations) |
| `archive/` | Superseded or fully resolved docs kept for the evidence trail. Each carries a banner; never start work from one |
| `evidence/`, `screenshots/` | Raw measurement output and committed captures cited by the docs above |

## Document kinds

Every document here except `AGENTS.md` opens with a frontmatter block naming its kind and status
(`spec/` and `adr/` carry their own richer one), checked by `tools/docs/check_doc_meta.py`:

| kind | what it is | read it as |
| --- | --- | --- |
| `design` | how a subsystem works today and where it is going | current unless its status says otherwise |
| `reference` | facts about the PS5 platform (packet sizes, tiling, save-data layout) | stable; verified against captures |
| `investigation` | a dated measurement pass or bring-up log | true when measured; re-measure before quoting |
| `status` | one title's current rung, route, blockers and `## Ruled out` | the authoritative per-title page |
| `process` | how we work: verification, debugging, releases, orchestration | binding procedure |
| `plan` | roadmaps and planned moves | intent, not fact |
| `archive` | superseded or resolved records, in `archive/` only | evidence; never start work from one |

`status` is `current`, `historical` or `superseded` (with `superseded-by: <path>`). A new subsystem
design doc may declare `template: design` to have its sections checked.

Cite docs by full path (`prosper/docs/gpu/GRAPHICS.md`); file names are unique repo-wide, so a bare
name is still greppable. `tools/docs/check_arch_docs.py` gates `spec/` and `adr/` in CI.
`tools/docs/reorganize_docs.py` is the one-shot that performed the 2026-10
move; do not re-run it.
