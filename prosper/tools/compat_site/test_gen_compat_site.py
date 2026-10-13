"""Tests for the compatibility-site generator.

Hand-built tracker/compat fixtures pin what the parsers accept and reject; the last
case runs the real committed files, which must parse and join without error. The
XSS arms use a milestone the page must escape, so a template that interpolates raw
data fails here instead of shipping script injection on a public page.

Runs under pytest; ctest `compat_site_generator` runs the generator's separate
stdlib selftest. Browser rendering checks require Node.js.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_compat_site import (  # noqa: E402
    PAGE_CSS,
    ParseError,
    bucket_of,
    build,
    counts,
    engine_family,
    ladder_rows,
    links_cell,
    parse_compat_covers,
    parse_compat_table,
    parse_tracker_table,
    render_classic,
    render_index,
    strip_md,
    vendor_covers,
    vendor_logo,
)
from site_page import PAGE_SCRIPT  # noqa: E402

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


def test_vendor_covers_refuses_escape(tmp_path):
    """An existing ../outside file, an absolute path and an escaping symlink are all refused."""
    inside = tmp_path / "root"
    (inside / "assets").mkdir(parents=True)
    (inside / "assets" / "ok.webp").write_bytes(b"cover")
    (tmp_path / "outside.webp").write_bytes(b"secret")
    assert vendor_covers(inside, tmp_path / "out", {"PPSA00001": "assets/ok.webp"}) == {
        "PPSA00001": "covers/ok.webp"
    }
    for evil in ("../outside.webp", str(tmp_path / "outside.webp")):
        try:
            vendor_covers(inside, tmp_path / "out", {"PPSA00001": evil})
        except ParseError:
            pass
        else:
            raise AssertionError(f"escaping cover {evil!r} was vendored")
    link = inside / "assets" / "link.webp"
    try:
        link.symlink_to(tmp_path / "outside.webp")
    except (OSError, NotImplementedError):
        return  # symlink creation needs privilege here; the resolve check above is the control
    try:
        vendor_covers(inside, tmp_path / "out", {"PPSA00001": "assets/link.webp"})
    except ParseError:
        pass
    else:
        raise AssertionError("escaping symlink was vendored")


def test_hostile_url_and_filename_are_escaped():
    """Quote-breaking tracker URLs and cover names reach attributes only through the escaper."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    for point in ("e(t.tracker.url)", "e(t.doc_url)", "e(t.cover)", "e(t.colour)", "e(t.hint)"):
        assert point in page, point
    for raw in ("t.tracker.url}", "t.doc_url}`", 't.cover}"', 't.colour}"'):
        assert raw not in page, raw


def test_vendor_logo_refuses_escaping_symlink(tmp_path):
    """The fixed logo path must obey the same resolved-checkout boundary as covers."""
    root = tmp_path / "root"
    (root / "assets").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    logo = root / "assets" / "prosper-logo.png"
    logo.write_bytes(b"inside logo")
    assert vendor_logo(root, out) == "prosper-logo.png"
    assert (out / "prosper-logo.png").read_bytes() == b"inside logo"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"synthetic outside data")
    logo.unlink()
    try:
        logo.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create the symlink refusal fixture")
    with pytest.raises(ParseError, match="escapes the checkout"):
        vendor_logo(root, out)
    assert (out / "prosper-logo.png").read_bytes() == b"inside logo"


def test_browser_rendering_escapes_hostile_attributes_and_text(tmp_path):
    """Run the actual client renderer, then parse its HTML to detect extra attributes/tags."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the page's browser renderer")
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    title = titles[0]
    title.update(
        title='Demo <script>bad</script> " &',
        cover='covers/demo" onerror="bad.webp',
        doc_url='https://example.test/doc" onclick="bad',
        milestone='<img src=x onerror="bad"> & milestone',
        hint='hint" onclick="bad',
    )
    title["tracker"]["url"] = 'https://example.test/" onclick="bad'
    harness = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const nodes = {};
function element() {
  return {innerHTML:'', textContent:'', hidden:true, children:[],
    classList:{add(){}, remove(){}, toggle(){}, contains(){return false;}},
    querySelector(){return element();}, addEventListener(){}, setAttribute(){}};
}
// This test exercises rendering, not fetch scheduling. Keep loading pending, then
// supply the decoded records to the actual renderer without cross-realm Promise jobs.
const pending = {then(){return this;}, catch(){return this;}};
const context = {
  document:{getElementById(id){return nodes[id] || (nodes[id] = element());},
    addEventListener(){}, querySelectorAll(){return [];}},
  fetch(){return pending;}
};
vm.runInNewContext(input.script, context);
context.T = input.payload.titles;
context.B = input.payload.buckets;
context.go();
if (!nodes.grid || !nodes.grid.innerHTML) { console.error('grid not rendered'); process.exit(2); }
process.stdout.write(nodes.grid.innerHTML);
"""
    # A file avoids making the browser-escaping assertion depend on Node's stdin
    # pipe handling. CI timed out before the old asynchronous harness returned.
    fixture = tmp_path / "page.json"
    fixture.write_text(
        json.dumps(
            {"script": PAGE_SCRIPT, "payload": {"titles": titles, "buckets": counts(titles)}}
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, "-e", harness, str(fixture)],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert result.stderr == "", result.stderr

    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            self.tags.append((tag, dict(attrs)))

        def handle_data(self, data):
            self.text.append(data)

    parsed = Tags()
    parsed.feed(result.stdout)
    assert any(tag == "article" for tag, _ in parsed.tags), result.stdout
    assert not any(tag == "script" for tag, _ in parsed.tags), result.stdout
    assert not any(key.startswith("on") for _, attrs in parsed.tags for key in attrs), result.stdout
    images = [attrs for tag, attrs in parsed.tags if tag == "img"]
    assert len(images) == 1 and images[0]["src"] == title["cover"]
    assert images[0]["alt"] == title["title"]
    links = [attrs["href"] for tag, attrs in parsed.tags if tag == "a"]
    assert links == [title["tracker"]["url"], title["doc_url"]]
    assert title["title"] in parsed.text and title["milestone"] in parsed.text


def test_site_has_every_section_and_no_leftover_tokens():
    """The landing page is a website, not a table: nav, hero, story, ladder, games, scope."""
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    page = render_index(titles, counts(titles), "2026-01-01", "prosper-logo.png")
    for anchor in ('id="top"', 'id="how"', 'id="progress"', 'id="games"', 'id="scope"'):
        assert anchor in page, anchor
    assert "@@" not in page, "an unreplaced template token reached the page"
    assert 'name="viewport"' in page and "@media (max-width:640px)" in page
    assert page.count("<script>") == 1 and page.count("</script>") == 1


def test_engine_variants_collapse_into_families():
    """Five Unity strings are one filter choice; unknowns are grouped, not dropped."""
    unity = [
        "Unity",
        "Unity / IL2CPP",
        "Unity 2019.4 / IL2CPP",
        "Unity 2020.3.1 / IL2CPP",
        "Unity 6",
        "Unity 6 / IL2CPP",
    ]
    assert {engine_family(e) for e in unity} == {"Unity"}
    assert {
        engine_family(e) for e in ("Unreal Engine", "Unreal Engine 4", "Unreal Engine (inferred)")
    } == {"Unreal Engine"}
    assert engine_family("Hedgehog Engine 2 (Needle)") == "Hedgehog Engine"
    assert engine_family("Custom (Saber)") == "Custom"
    assert engine_family("RE Engine") == "RE Engine"
    assert engine_family("Anvil (inferred)") == "Anvil"
    assert engine_family("-") == engine_family("—") == "Unknown"
    titles = build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT))
    assert titles[0]["family"] == "Unity" and titles[0]["engine"] == "Unity"


def test_dropdowns_are_custom_components_and_filter_by_family():
    """No native <select> (its popup cannot be themed); the filter compares families."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    assert "<select" not in page and "<option" not in page
    assert page.count('class="dd-btn"') == 2 and page.count('class="dd-list"') == 2
    assert 'id="eng"' in page and 'id="sort"' in page
    assert ".dd-list{position:absolute" in page and "border-radius:16px" in page
    assert "padding:12px 14px 12px 16px" in page  # room for the chevron on the right
    assert 'aria-haspopup="listbox"' in page and "Escape" in page and "ArrowDown" in page
    assert "t.family===eng" in page and "t.engine===eng" not in page


def test_game_list_is_paginated_and_filters_reset_to_page_one():
    """12 per page with a pager; every filter path resets the page; pager never shows <2 pages."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    assert 'id="pager"' in page and "PER=12" in page
    assert "R.slice(from,from+PER)" in page
    assert 'aria-label="Previous page"' in page and 'aria-label="Next page"' in page
    # search, status chip, engine and sort must all return to page 1
    assert "q=ev.target.value.toLowerCase();page=1;go()" in page
    assert 'on=b.getAttribute("data-s");page=1;' in page
    assert "eng=v;page=1;go()" in page and "sort=v;page=1;go()" in page
    assert 'if(n<=1){$("pager").innerHTML="";return;}' in page


def test_nav_links_have_hit_area_hover_and_scrollspy():
    """Menu links are pills (padding = hit area) with hover/active state and a scrollspy."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    assert ".navlinks a:not(.btn){padding:10px 14px;border-radius:999px;cursor:pointer" in page
    assert ".navlinks a:not(.btn):hover,.navlinks a.active{" in page
    assert 'typeof IntersectionObserver!=="undefined"' in page  # safe where it is missing
    for target in ("how", "progress", "games", "scope"):
        assert f'href="#{target}"' in page and f'id="{target}"' in page


def test_cursors_match_what_is_clickable():
    """Hand on links/buttons/chips, text cursor on search, icon overlay never eats the pointer."""
    page = render_index(
        build(parse_tracker_table(TRACKER), parse_compat_table(COMPAT)), [], "2026-01-01"
    )
    assert (
        "a,button,summary,.btn,.chip,.ms,.dd-btn,.dd-list li,.pg,label[for]{cursor:pointer}" in page
    )
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
    """Malformed inputs, each of which must abort rather than emit a wrong page."""
    tracker_dup = TRACKER + TRACKER.splitlines()[-1] + "\n"
    compat_dup = COMPAT + COMPAT.splitlines()[-1] + "\n"
    tracker_nosep = TRACKER.replace("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n", "")
    bad = [
        (parse_tracker_table, TRACKER.replace("`PPSA00001`", "`BOGUS`")),
        (parse_tracker_table, TRACKER.replace("| 3 |", "| 7 |")),
        (parse_tracker_table, tracker_dup),  # duplicate id drops a title silently
        (parse_tracker_table, tracker_nosep),  # first data row eaten as separator
        (
            parse_tracker_table,
            TRACKER.replace("https://example.test/11", 'javascript:alert("x")'),
        ),  # hostile scheme in an attribute-bound URL
        (parse_compat_table, COMPAT.replace("`PPSA00001`", "`PPSA1`")),
        (parse_compat_table, COMPAT.replace("[#11](https://example.test/11) |", "- |")),
        (parse_compat_table, compat_dup),  # duplicate id publishes the wrong milestone
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
    test_engine_variants_collapse_into_families()
    test_dropdowns_are_custom_components_and_filter_by_family()
    test_game_list_is_paginated_and_filters_reset_to_page_one()
    test_nav_links_have_hit_area_hover_and_scrollspy()
    test_cursors_match_what_is_clickable()
    test_ladder_shows_all_seven_rungs_with_live_counts()
    test_links_and_pills_stay_on_one_line()
    test_hostile_url_and_filename_are_escaped()
    test_covers_first_repo_image_wins()
    with tempfile.TemporaryDirectory() as tmp:
        test_vendor_covers_missing_source_aborts(Path(tmp))
        test_vendor_covers_refuses_escape(Path(tmp))
    test_strip_md_keeps_link_labels()
    test_bad_rows_fail_loudly()
    test_real_tree_parses_and_joins()
    print("compat_site tests ok")
