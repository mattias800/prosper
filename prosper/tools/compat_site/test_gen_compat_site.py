"""Tests for the compatibility-site generator.

Hand-built tracker/compat fixtures pin what the parsers accept and reject; the last
case runs the real committed files, which must parse and join without error. The
XSS arms use a milestone the page must escape, so a template that interpolates raw
data fails here instead of shipping script injection on a public page.

Runs under pytest, or directly (`python3 test_gen_compat_site.py`) as ctest
`compat_site_generator` does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_compat_site import (  # noqa: E402
    PAGE_CSS,
    ParseError,
    bucket_of,
    build,
    counts,
    ladder_rows,
    links_cell,
    parse_compat_covers,
    parse_compat_table,
    parse_tracker_table,
    render_classic,
    render_index,
    strip_md,
    vendor_covers,
)

TRACKER = (
    "| Title | Title ID | Rung | Ladder | FPS | Guard | Oracle | Open blockers | Tracker | Status doc |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    "| Demo Game | `PPSA00001` | 3 | `123---` | - | `demo-gameplay` | none | - | "
    "[#11](https://example.test/11) | - |\n"
)

COMPAT = (
    "| Title | Title ID | Engine | User-visible milestone | Tracker |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| *Demo Game* | `PPSA00001` | Unity | ✅ First [room](https://example.test/x) | "
    "[#11](https://example.test/11) |\n"
)


def test_join_uses_tracker_rung_and_compat_prose():
    """Rung/bucket come from the tracker row; engine/milestone from COMPATIBILITY.md."""
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    assert len(titles) == 1
    row = titles[0]
    assert (row["rung"], row["status"]) == (3, "Playable")
    assert row["hint"].startswith("Rung 3 of 6")
    assert row["engine"] == "Unity"
    assert row["milestone"] == "✅ First room"
    assert row["guard"] == ["demo-gameplay"]
    assert row["tracker"] == {"number": "11", "url": "https://example.test/11"}


def test_compat_only_title_is_untracked_not_rung_zero():
    """A title with no tracker row must not read as measured-and-failing."""
    compat = parse_compat_table(COMPAT)
    titles = build([], compat)
    assert titles[0]["rung"] is None
    assert titles[0]["status"] == "Untracked"


def test_markers_are_not_statuses():
    """Every rung maps to its plain-language bucket label; an unknown rung fails loudly."""
    assert bucket_of(6)[0] == "Complete"
    assert bucket_of(0)[0] == "Boots"
    assert "rung" not in bucket_of(3)[0].lower() and "guard" not in bucket_of(6)[0].lower()
    try:
        bucket_of(9)
    except ParseError:
        pass
    else:
        raise AssertionError("bucket_of(9) parsed without error")


def test_milestone_html_is_escaped():
    """Milestone prose reaches the templates escaped, in both pages."""
    compat = parse_compat_table(COMPAT)
    compat["PPSA00001"]["milestone"] = "<script>alert(1)</script>"
    titles = build(parse_tracker_table(TRACKER), compat)
    index_html = render_index(titles, counts(titles), "2026-01-01")
    assert "e(t.milestone" in index_html and "<script>alert" not in index_html
    assert "??" not in index_html.split("<script>")[1]
    assert index_html.count("`") % 2 == 0
    classic = render_classic(titles, "2026-01-01")
    assert "&lt;script&gt;" in classic and "<script>alert" not in classic
    assert '<td class="links"><a href="https://example.test/11">Tracker</a></td>' in classic


def test_covers_first_repo_image_wins():
    """First image per `##` section is the cover; later and external ones are ignored."""
    sections = (
        "## Demo Game — `PPSA00001`\n\n"
        '<p align="center"><img src="assets/screenshots/demo-title.webp" alt="t"></p>\n'
        '<p align="center"><img src="assets/screenshots/demo-second.webp" alt="t"></p>\n\n'
        "## Next — `PPSA00002`\n\n"
        '<p align="center"><img src="https://example.test/ext.png" alt="t"></p>\n'
    )
    covers = parse_compat_covers(sections)
    assert covers == {"PPSA00001": "assets/screenshots/demo-title.webp"}, covers
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT), covers)
    assert titles[0]["cover"] == "assets/screenshots/demo-title.webp", titles


def test_vendor_covers_missing_source_aborts(tmp_path):
    """A cover outside the checkout fails the build instead of publishing a broken tile."""
    try:
        vendor_covers(tmp_path, tmp_path / "out", {"PPSA00001": "assets/screenshots/nope.webp"})
    except ParseError:
        pass
    else:
        raise AssertionError("missing cover source did not fail")


def test_site_has_every_section_and_no_leftover_tokens():
    """The landing page is a website, not a table: nav, hero, story, ladder, games, scope."""
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    page = render_index(titles, counts(titles), "2026-01-01", "prosper-logo.png")
    for anchor in ('id="top"', 'id="how"', 'id="progress"', 'id="games"', 'id="scope"'):
        assert anchor in page, anchor
    assert "@@" not in page, "an unreplaced template token reached the page"
    assert 'name="viewport"' in page and "@media (max-width:640px)" in page
    assert page.count("<script>") == 1 and page.count("</script>") == 1


def test_cursors_match_what_is_clickable():
    """Hand on links/buttons/chips, text cursor on search, icon overlay never eats the pointer."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    assert "a,button,select,summary,.btn,.chip,.ms,label[for]{cursor:pointer}" in page
    assert "input[type=search],input[type=text]{cursor:text}" in page
    assert "opacity:.6;pointer-events:none}" in page


def test_ladder_shows_all_seven_rungs_with_live_counts():
    """Empty rungs still render (count 0) so the scale is always the whole scale."""
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    rows = ladder_rows(counts(titles))
    assert [r["rung"] for r in rows] == [6, 5, 4, 3, 2, 1, 0]
    assert {r["rung"]: r["count"] for r in rows} == {6: 0, 5: 0, 4: 0, 3: 1, 2: 0, 1: 0, 0: 0}
    page = render_index(titles, counts(titles), "2026-01-01")
    assert page.count('class="rung"') == 7


def test_links_and_pills_stay_on_one_line():
    """Links render inline (`Tracker · Doc`, never stacked); pills cannot wrap."""
    assert "white-space:nowrap" in PAGE_CSS
    assert links_cell(
        {"number": "11", "url": "https://example.test/11"},
        "https://example.test/doc.md",
    ) == (
        '<td class="links"><a href="https://example.test/11">Tracker</a> · '
        '<a href="https://example.test/doc.md">Doc</a></td>'
    )
    assert links_cell({"number": "11", "url": "https://example.test/11"}, "") == (
        '<td class="links"><a href="https://example.test/11">Tracker</a></td>'
    )


def test_strip_md_keeps_link_labels():
    """`[room](url)` renders as `room`, not as a dropped cell or a raw URL."""
    assert strip_md("✅ First [room](https://example.test/x)") == "✅ First room"
    assert strip_md("New Joe &amp; Mac") == "New Joe & Mac"


def test_bad_rows_fail_loudly():
    """Four malformed inputs, each of which must abort rather than emit a wrong page."""
    bad = [
        (parse_tracker_table, TRACKER.replace("`PPSA00001`", "`BOGUS`")),
        (parse_tracker_table, TRACKER.replace("| 3 |", "| 7 |")),
        (parse_compat_table, COMPAT.replace("`PPSA00001`", "`PPSA1`")),
        (parse_compat_table, COMPAT.replace("[#11](https://example.test/11) |", "- |")),
    ]
    for fn, text in bad:
        try:
            fn(text)
        except ParseError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} parsed without error: {text[:60]!r}")


def test_real_tree_parses_and_joins():
    """The committed files parse with unique ids and a sane bucket split."""
    root = Path(__file__).resolve().parents[3]
    rows = parse_tracker_table((root / "PROGRESS_TRACKER.md").read_text(encoding="utf-8"))
    compat = parse_compat_table((root / "COMPATIBILITY.md").read_text(encoding="utf-8"))
    titles = build(rows, compat)
    assert len(titles) == len({t["id"] for t in titles}) > 50
    assert sum(s["count"] for s in counts(titles)) == len(titles)


if __name__ == "__main__":
    import tempfile

    test_join_uses_tracker_rung_and_compat_prose()
    test_compat_only_title_is_untracked_not_rung_zero()
    test_markers_are_not_statuses()
    test_milestone_html_is_escaped()
    test_site_has_every_section_and_no_leftover_tokens()
    test_cursors_match_what_is_clickable()
    test_ladder_shows_all_seven_rungs_with_live_counts()
    test_links_and_pills_stay_on_one_line()
    test_covers_first_repo_image_wins()
    with tempfile.TemporaryDirectory() as tmp:
        test_vendor_covers_missing_source_aborts(Path(tmp))
    test_strip_md_keeps_link_labels()
    test_bad_rows_fail_loudly()
    test_real_tree_parses_and_joins()
    print("compat_site tests ok")
