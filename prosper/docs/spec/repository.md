---
kind: spec
status: draft
owner: area:infra
last-verified: 2026-10-06 c2ee413a
---

# Repository layout

Rules about where non-code artefacts live in the repository, so that a folder answers "where is
it?" without a search. Source layout is `layers.md`; documentation kinds are `docs/AGENTS.md`.

### ASSET-1 -- screenshots live at a dated path under their title

A committed screenshot is `assets/screenshots/<GROUP>/<YYYY-MM-DD>-<what>[-i<issue>].webp`, where
`GROUP` is `<TITLE_ID>-<slug>` (one folder per title id), `cross-title` or `app`, and `what` is
lowercase kebab-case. Files committed before the rule are grandfathered by
`tools/screenshots/legacy_screenshot_paths.txt`, a list that only shrinks.
Status: proposed (adr:0026)
Enforcement: ci:screenshot-paths, adr:0026
