# `compat_site` — generated compatibility page

Static, offline generator for the public per-title compatibility page (GitHub Pages).
It answers: "how far does title X get, and can I search/filter the list?" —
the shadPS4/AnyPS5-roadmap shape: headline stats, a filterable main table, and a
no-JS fallback.

- `gen_compat_site.py` joins the two committed sources — `PROGRESS_TRACKER.md`
  (rung, ladder, guard, tracker links; itself generated from the `tracker:game`
  issues) and `COMPATIBILITY.md` (summary table for engine/milestone prose, first
  `<img>` per `##` title section for the cover) — and emits `compat.json`,
  `index.html`, `classic.html`, `llms.txt` plus `covers/` into an output directory.
  It reads no network: the Pages workflow runs it against the checkout
  on every `main` push, so the page is exactly as current as the committed files.
- The trackers stay authoritative. A row here is a projection of `PROGRESS_TRACKER.md`
  plus hand-written milestone prose; when they disagree with a tracker, the tracker wins.
- Statuses are rung buckets (Guarded 6 … Boots 0), never the ✅/🚧/🔬 markers:
  the charter records those markers are not a rung scale. There are no OS/region/version
  columns — prosper tracks one dump revision per title, mostly on Linux, so per-OS
  breakdowns would be invented data. Titles with no `##` section yet (recent
  additions) render a `—` cover cell rather than a wrong image.
- `test_gen_compat_site.py` pins the parsers with hand-built fixtures plus a real-tree
  run over the committed files. ctest runs it as `compat_site_generator`.
