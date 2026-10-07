---
kind: adr
status: proposed
date: 2026-10-06
---

# ADR 0026: One screenshot root, one folder per title, dated file names

## Context

Committed screenshots are the project's progress record: the charter requires every PR that
advances a title to attach captures and a `BLOG.md` entry, and `tools/screenshots/shrink.py` keeps
each one a 1920-wide WebP. Their *paths* have no convention. On `main` at the time of writing:

- `assets/screenshots/` is one flat folder of 261 files, and `prosper/docs/screenshots/` a second
  root of 53;
- names follow whatever each author chose: an issue number first (`3407-default-control-gta.webp`),
  a title spelled two ways (`alex-kidd.webp`, `alexkidd-title-wave32-proved.webp`), a bare title
  (`dead-cells.webp`), and never a date;
- about 600 references point into the two roots: 258 Markdown links and 350 HTML `src=`
  attributes at `c2ee413a`.

So "every capture of one title", "the latest capture of it", or "what it looked like before a given
fix" cannot be answered by listing a folder; it takes a grep over names that share no convention.
#4567 found the cost of drift directly: `shrink.py` re-encoded PNGs to WebP, and links kept pointing
at the old names.

## Decision

Spec rule `ASSET-1`:

    assets/screenshots/<GROUP>/<YYYY-MM-DD>-<what>[-i<issue>].webp

1. **One root**, `assets/screenshots/`. `prosper/docs/screenshots/` is retired; its files move.
2. **`GROUP`** is `<TITLE_ID>-<slug>` for one title (`PPSA04263-gta5`): the title id makes the
   folder unambiguous, the slug makes it readable. Exactly one folder per title id. Captures that
   are not one title go in `cross-title/` (a comparison across titles or a shared fix) or `app/`
   (the frontend's own UI).
3. **File name**: the capture date first, so a folder lists in time order; then what the picture is
   evidence of, in lowercase kebab-case; then optionally `-i<number>` for the issue or PR it belongs
   to. Format WebP, as `shrink.py` produces.
4. **Enforcement**: `tools/screenshots/check_screenshot_paths.py`, in the `Docs` CI job, reports any
   image under either root that neither conforms nor is listed in
   `legacy_screenshot_paths.txt`. While this ADR is proposed it runs `--report-only` (warnings,
   step passes); accepting the ADR removes the flag, the same PR regenerating the list against the
   merge result so no capture merged in between is reported as new. The legacy list grandfathers the 321 files committed before this
   rule; it may only shrink, and a stale line fails.
5. **Migration** of the existing files is one move-only PR after this ADR is accepted: `git mv`
   into the convention, every Markdown reference rewritten by script (the `check_doc_meta.py` link
   check proves none was missed), and the legacy list deleted line by line. Capture dates come from
   each file's first commit date, since the originals record none.

## Consequences

A title's visual history is one folder in time order, a capture's age is in its name, and a new
capture cannot arrive outside the scheme. The migration touches every document that cites a
screenshot, including status pages other lanes edit, so it lands in a quiet window as a single
mechanical PR. If this ADR is accepted, new captures follow the convention from then on and old ones stay where they
are until that migration.

## Alternatives considered

- Folders by title name instead of id: names are spelled several ways today; ids are not.
- Date subfolders (`<title>/<YYYY-MM>/`): a deeper tree for the same ordering the name prefix gives.
- Leave paths free and add an index file: an index drifts; a path is checked.

## Approval

Requires the project owner's acceptance. Until then the checker only reports: a capture that lands
outside the convention, including one merged by another lane between two CI runs, prints a warning
and fails nothing. The charter's screenshot instructions should name the path in the same change
that makes the check binding, so lanes learn the rule from the charter rather than from a red
check.
