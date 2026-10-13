# `compat_site` — generated compatibility page

Static, offline generator for the public prosper website (GitHub Pages): a landing page
that explains the project and hosts the searchable per-title compatibility list,
plus a no-JS static table.

- `gen_compat_site.py` joins the two committed sources — `PROGRESS_TRACKER.md`
  (rung, ladder, guard, tracker links; itself generated from the `tracker:game`
  issues) and `COMPATIBILITY.md` (summary table for engine/milestone prose, first
  `<img>` per `##` title section for the cover) — and emits `compat.json`,
  `index.html`, `classic.html`, `llms.txt` plus `covers/` into an output directory.
  It reads no network: the Pages workflow runs it against the checkout
  on every `main` push, so the page is exactly as current as the committed files.
- `site_page.py` is the presentation half: the prosper landing page (nav, hero with live
  stats and screenshot mosaic, how-it-works, the ladder with live counts, a card-grid game
  browser, scope, footer). Its copy only restates README.md and `docs/ROADMAP.md`; it makes
  no claim about a single title. The template is a plain string with `@@TOKEN@@`
  placeholders, so CSS and the page script keep their braces; a test fails on any leftover token.
- The trackers stay authoritative. A row here is a projection of `PROGRESS_TRACKER.md`
  plus hand-written milestone prose; when they disagree with a tracker, the tracker wins.
- Statuses are plain-language rung buckets (Complete 6 … Boots 0, with the exact rung
  as a hover tooltip), never the ✅/🚧/🔬 markers:
  the charter records those markers are not a rung scale. There are no OS/region/version
  columns — prosper tracks one dump revision per title, mostly on Linux, so per-OS
  breakdowns would be invented data. Titles with no `##` section yet (recent
  additions) render a `—` cover cell rather than a wrong image.
- `test_gen_compat_site.py` pins the parsers with hand-built fixtures plus a real-tree
  run over the committed files. It runs under pytest; ctest runs only the `--selftest`
  arm as `compat_site_generator`, mirroring the tracker generator's registration.
