#!/usr/bin/env python3
"""Generate the static compatibility page from the committed tracker projection.

The page joins two files that already exist in the checkout — `PROGRESS_TRACKER.md`
(rung, ladder, guard, tracker links; generated from the `tracker:game` issues) and
`COMPATIBILITY.md`'s summary table (engine, milestone prose) — and emits a static
site (`compat.json`, `index.html`, `classic.html`, `llms.txt`) with no build step
beyond this script and no network at all. The GitHub Pages workflow runs it against
the checkout on every `main` push, so the page is exactly as current as the files
it was built from, and the trackers stay authoritative over every row.

Statuses are rung buckets (Guarded 6 down to Boots 0), never the ✅/🚧/🔬 markers,
which the charter records are not a rung scale. There are no OS/region/version
columns: prosper tracks one dump revision per title, so per-OS breakdowns would be
invented data.

USAGE
    gen_compat_site.py OUTDIR [--root DIR]   # build the site into OUTDIR
    gen_compat_site.py --selftest            # offline: prove the parsers still discriminate
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
from pathlib import Path

from site_page import render_site

TOOL_DIR = Path(__file__).resolve().parent
PROSPER = TOOL_DIR.parent.parent
REPO_ROOT = PROSPER.parent

TRACKER_MD = REPO_ROOT / "PROGRESS_TRACKER.md"
COMPAT_MD = REPO_ROOT / "COMPATIBILITY.md"

ID_RE = re.compile(r"PPSA\d{5}")
TRACKER_LINK_RE = re.compile(r"\[#(\d+)\]\(([^)]+)\)")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")
COVER_IMG_RE = re.compile(r'<img\s+src="([^"]+)"')
COVER_HEADING_RE = re.compile(r"^##\s+.+`?(PPSA\d{5})`?\s*$")

# Highest rung first: (rung, plain-language label, bar colour, tooltip). None covers
# titles present only in COMPATIBILITY.md with no tracker row at all. Labels stay short
# and jargon-free (no "rung N" / "guard" terms); the tooltip carries the exact rung.
BUCKETS: list[tuple[int | None, str, str, str]] = [
    (6, "Complete", "#2ea043", "Rung 6 of 6: finished, guarded by a reviewed automatic check"),
    (5, "PS5 match", "#39c5cf", "Rung 5 of 6: compared against a real PS5"),
    (4, "Verified", "#7ee787", "Rung 4 of 6: a human confirmed it looks right"),
    (3, "Playable", "#4493f8", "Rung 3 of 6: gameplay renders"),
    (2, "Menus", "#ab7df8", "Rung 2 of 6: title screen or menus"),
    (1, "Graphics", "#d29922", "Rung 1 of 6: first graphics, no menus yet"),
    (0, "Boots", "#f85149", "Rung 0 of 6: starts but shows no game picture yet"),
    (None, "Untracked", "#6e7681", "No measurement yet"),
]


class ParseError(Exception):
    """A required table shape is missing or malformed. Names the file; aborts the run."""


def split_row(line: str) -> list[str]:
    """Split a Markdown table row on unescaped pipes; `\\|` inside a cell is data."""
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.replace("\\|", "|").strip() for c in cells]


def strip_md(text: str) -> str:
    """Reduce inline Markdown to plain text: links keep their label, emphasis goes."""
    text = html.unescape(text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.strip().strip("*").strip("`").strip()
    return re.sub(r"\s+", " ", text)


def tracker_links(cell: str) -> list[dict[str, str]]:
    """Every `[#NNNN](url)` in a cell, in order."""
    return [{"number": m.group(1), "url": m.group(2)} for m in TRACKER_LINK_RE.finditer(cell)]


def find_table(text: str, source: str, want: list[str]) -> list[str]:
    """Body rows of the table under `want`'s header. Other tables in the file are ignored."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("|") and [c.strip("* ").strip() for c in split_row(ln)] == want:
            body = []
            for row in lines[i + 1 :]:
                if not row.startswith("|"):
                    break
                body.append(row)
            if len(body) < 2:
                raise ParseError(f"{source}: table under {want[0]!r} has no rows")
            return body[1:]  # skip the separator row
    raise ParseError(f"{source}: table with header {want[0]!r} not found")


def parse_tracker_table(text: str) -> list[dict]:
    """Rows of PROGRESS_TRACKER.md's per-title table. Hard-fails on anything unparseable."""
    want = [
        "Title",
        "Title ID",
        "Rung",
        "Ladder",
        "FPS",
        "Guard",
        "Oracle",
        "Open blockers",
        "Tracker",
        "Status doc",
    ]
    rows = []
    for ln in find_table(text, "PROGRESS_TRACKER.md", want):
        cells = split_row(ln)
        if len(cells) != len(want):
            raise ParseError(
                f"PROGRESS_TRACKER.md: {len(cells)} cells, want {len(want)}: {ln[:80]}"
            )
        title, tid, rung, ladder = cells[0], cells[1].strip("`"), cells[2], cells[3].strip("`")
        if not ID_RE.fullmatch(tid):
            raise ParseError(f"PROGRESS_TRACKER.md: bad title id {tid!r}")
        if not rung.isdigit() or not 0 <= int(rung) <= 6:
            raise ParseError(f"PROGRESS_TRACKER.md: bad rung {rung!r} for {tid}")
        if not re.fullmatch(r"[0-6-]{6}", ladder):
            raise ParseError(f"PROGRESS_TRACKER.md: bad ladder {ladder!r} for {tid}")
        trackers = tracker_links(cells[8])
        if not trackers:
            raise ParseError(f"PROGRESS_TRACKER.md: no tracker link for {tid}")
        guards = [g.strip("` ") for g in split_row_fallback(cells[5]) if g.strip("` ") != "-"]
        rows.append(
            {
                "title": title,
                "id": tid,
                "rung": int(rung),
                "ladder": ladder,
                "fps": None if cells[4] == "-" else strip_md(cells[4]),
                "guard": guards,
                "blockers": tracker_links(cells[7]),
                "tracker": trackers[0],
                "status_doc": cells[9],
            }
        )
    if not rows:
        raise ParseError("PROGRESS_TRACKER.md: table has no rows")
    return rows


def split_row_fallback(cell: str) -> list[str]:
    """Comma-separated backticked guard lists; kept separate so the failure mode is visible."""
    return [g.strip() for g in cell.split(",")]


def parse_compat_table(text: str) -> dict[str, dict]:
    """COMPATIBILITY.md's summary table: id -> engine, milestone prose, tracker number."""
    want = ["Title", "Title ID", "Engine", "User-visible milestone", "Tracker"]
    out: dict[str, dict] = {}
    for ln in find_table(text, "COMPATIBILITY.md", want):
        cells = split_row(ln)
        if len(cells) != 5:
            raise ParseError(f"COMPATIBILITY.md: {len(cells)} cells, want 5: {ln[:80]}")
        ids = ID_RE.findall(cells[1])
        if len(ids) != 1:
            raise ParseError(f"COMPATIBILITY.md: bad title id cell {cells[1]!r}")
        milestone = strip_md(cells[3])
        trackers = tracker_links(cells[4])
        if not trackers:
            raise ParseError(f"COMPATIBILITY.md: no tracker link for {ids[0]}")
        out[ids[0]] = {
            "title": strip_md(cells[0]),
            "engine": strip_md(cells[2]) or "-",
            "milestone": milestone,
            "tracker": trackers[0],
        }
    if not out:
        raise ParseError("COMPATIBILITY.md: summary table has no rows")
    return out


def parse_compat_covers(text: str) -> dict[str, str]:
    """First repo-relative `<img src>` under each `## Title PPSAxxxxx` section.

    Only the first image per section: it is the title screen where one exists, and one
    cover per row keeps the Pages artifact small. External URLs are skipped, never vendored.
    """
    covers: dict[str, str] = {}
    current: str | None = None
    for ln in text.splitlines():
        if ln.startswith("## "):
            heading = COVER_HEADING_RE.match(ln)
            current = heading.group(1) if heading else None
            continue
        if current and current not in covers:
            m = COVER_IMG_RE.search(ln)
            if m and not m.group(1).startswith(("http://", "https://")):
                covers[current] = m.group(1)
    return covers


def engine_family(engine: str) -> str:
    """Collapse the many recorded engine strings into a short, filterable family name.

    The detailed string stays on each card; only the filter uses the family, so
    `Unity 6 / IL2CPP`, `Unity 2019.4 / IL2CPP` and `Unity` are one choice, not five.
    """
    name = re.sub(r"\s*\(.*?\)", "", engine).strip()
    low = name.lower()
    if low in {"", "-", "—", "unknown"}:
        return "Unknown"
    for prefix, family in (
        ("unity", "Unity"),
        ("unreal", "Unreal Engine"),
        ("hedgehog", "Hedgehog Engine"),
        ("custom", "Custom"),
        ("ryu ga gotoku", "Ryu Ga Gotoku"),
    ):
        if low.startswith(prefix):
            return family
    return name


def bucket_of(rung: int | None) -> tuple[str, str, str]:
    """(label, colour, tooltip) for a rung; unknown rungs are a hard error, never a grey row."""
    for r, label, colour, hint in BUCKETS:
        if r == rung:
            return label, colour, hint
    raise ParseError(f"no status bucket for rung {rung!r}")


def build(
    tracker_rows: list[dict], compat: dict[str, dict], covers: dict[str, str] | None = None
) -> list[dict]:
    """Union by title id; tracker rung wins, COMPATIBILITY.md supplies engine/milestone/cover."""
    covers = covers or {}
    titles: dict[str, dict] = {}
    for row in tracker_rows:
        comp = compat.get(row["id"], {})
        label, colour, hint = bucket_of(row["rung"])
        titles[row["id"]] = {
            "title": row["title"],
            "id": row["id"],
            "engine": comp.get("engine", "-"),
            "family": engine_family(comp.get("engine", "-")),
            "milestone": comp.get("milestone", "-"),
            "cover": covers.get(row["id"], ""),
            "rung": row["rung"],
            "ladder": row["ladder"],
            "status": label,
            "colour": colour,
            "hint": hint,
            "guard": row["guard"],
            "fps": row["fps"],
            "blockers": row["blockers"],
            "tracker": row["tracker"],
            "status_doc": row["status_doc"],
        }
    for tid, comp in compat.items():
        if tid not in titles:
            label, colour, hint = bucket_of(None)
            titles[tid] = {
                "title": comp["title"],
                "id": tid,
                "engine": comp["engine"],
                "family": engine_family(comp["engine"]),
                "milestone": comp["milestone"],
                "cover": covers.get(tid, ""),
                "rung": None,
                "ladder": "------",
                "status": label,
                "colour": colour,
                "hint": hint,
                "guard": [],
                "fps": None,
                "blockers": [],
                "tracker": comp["tracker"],
                "status_doc": "-",
            }
    return [titles[k] for k in sorted(titles, key=lambda i: titles[i]["title"].lower())]


def vendor_covers(root: Path, out: Path, covers: dict[str, str]) -> dict[str, str]:
    """Copy each referenced cover into `out/covers/`; returns id -> site-relative path.

    A missing source file aborts the build: a row pointing at an image outside the
    checkout would publish a broken tile, and silently skipping it would look complete.
    """
    dest = out / "covers"
    dest.mkdir(parents=True, exist_ok=True)
    placed: dict[str, str] = {}
    taken: set[str] = set()
    for tid, src in covers.items():
        source = root / src
        if not source.is_file():
            raise ParseError(f"cover for {tid} not found in checkout: {src}")
        name = source.name
        if name in taken:
            name = f"{tid}_{name}"
        taken.add(name)
        (dest / name).write_bytes(source.read_bytes())
        placed[tid] = f"covers/{name}"
    return placed


def counts(titles: list[dict]) -> list[dict]:
    """Per-bucket counts in BUCKETS order, skipping empty buckets."""
    order = [label for _, label, _, _ in BUCKETS]
    hits: dict[str, int] = {label: 0 for label in order}
    for t in titles:
        hits[t["status"]] += 1
    total = len(titles) or 1
    out = []
    for rung, label, colour, _ in BUCKETS:
        if hits[label]:
            out.append(
                {
                    "rung": rung,
                    "label": label,
                    "colour": colour,
                    "count": hits[label],
                    "percent": round(100 * hits[label] / total, 1),
                }
            )
    return out


def esc(text: str) -> str:
    """HTML-escape data before it touches a template; titles carry `&` and quotes."""
    return html.escape(text, quote=True)


PAGE_CSS = """
body{font-family:sans-serif;background:#0d1117;color:#e6edf3;max-width:1100px;margin:auto;padding:16px}
a{color:#4493f8}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:16px 0}
.bar{height:8px;border-radius:4px;background:#30363d;margin:4px 0 12px}.bar>div{height:8px;border-radius:4px}
.row{display:flex;justify-content:space-between;gap:12px;font-size:14px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}
input,select{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font-size:14px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid #30363d;padding:8px;text-align:left;vertical-align:top}
td.milestone{min-width:220px}
img.thumb{width:96px;border-radius:4px;display:block}
img.logo{height:56px;vertical-align:middle;margin-right:12px}
.brand{display:flex;align-items:center;margin:16px 0 0}
.brand h1{margin:0;font-size:32px}
.pill{display:inline-block;padding:2px 12px;border-radius:12px;font-size:12px;color:#0d1117;font-weight:bold;white-space:nowrap}
.links{white-space:nowrap}
.mut{color:#8b949e;font-size:12px}
@media (max-width:720px){
 body{padding:10px}
 .brand h1{font-size:24px}
 img.logo{height:40px;margin-right:8px}
 .card{padding:12px}
 .filters{flex-direction:column}
 .filters input,.filters select{width:100%;box-sizing:border-box}
 img.thumb{width:64px}
 table{font-size:13px}
 th,td{padding:6px}
}
"""


def links_cell(tracker: dict[str, str], doc_url: str) -> str:
    """One line, never stacked: `Tracker · Doc`. Doc only when a status doc exists."""
    cell = f'<td class="links"><a href="{esc(tracker["url"])}">Tracker</a>'
    if doc_url:
        cell += f' · <a href="{esc(doc_url)}">Doc</a>'
    return cell + "</td>"


def ladder_rows(stats: list[dict]) -> list[dict]:
    """All seven real rungs, highest first, with live counts (zero when a bucket is empty)."""
    by_label = {s["label"]: s["count"] for s in stats}
    return [
        {"rung": r, "label": label, "colour": colour, "hint": hint, "count": by_label.get(label, 0)}
        for r, label, colour, hint in BUCKETS
        if r is not None
    ]


def render_index(titles: list[dict], stats: list[dict], generated: str, logo: str = "") -> str:
    """The prosper landing page; the game list loads from compat.json (classic.html is no-JS)."""
    return render_site(titles, stats, ladder_rows(stats), generated, logo)


def doc_url(cell: str) -> str:
    """First prosper/docs/*.md path in a status-doc cell, else empty."""
    m = re.search(r"prosper/docs/[^\s)]+\.md", cell)
    return "https://github.com/mattias800/prosper/blob/main/" + m.group(0) if m else ""


def render_classic(titles: list[dict], generated: str, logo: str = "") -> str:
    """Static table for no-JS readers: the same rows, no search."""

    def cover_cell(t: dict) -> str:
        if not t["cover"]:
            return "<td>—</td>"
        return (
            f'<td><img class="thumb" loading="lazy" src="{esc(t["cover"])}" '
            f'alt="{esc(t["title"])}"></td>'
        )

    def status_cell(t: dict) -> str:
        return (
            f'<td><span class="pill" style="background:{t["colour"]}" '
            f'title="{esc(t["hint"])}">{esc(t["status"])}</span></td>'
        )

    rows = "\n".join(
        f"<tr>{cover_cell(t)}<td>{esc(t['title'])}</td><td><code>{t['id']}</code></td>"
        f"<td>{esc(t['engine'])}</td>{status_cell(t)}"
        f'<td class="milestone">{esc(t["milestone"])}</td>'
        f"{links_cell(t['tracker'], t.get('doc_url', ''))}</tr>"
        for t in titles
    )
    brand = (
        f'<div class="brand"><img class="logo" src="{esc(logo)}" alt="prosper logo">'
        "<h1>Compatibility (static)</h1></div>"
        if logo
        else "<h1>Compatibility (static)</h1>"
    )
    favicon = f'<link rel="icon" href="{esc(logo)}">' if logo else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>prosper compatibility (static)</title>{favicon}<style>{PAGE_CSS}</style></head><body>
{brand}
<p>{len(titles)} titles. Generated {esc(generated)}. Trackers authoritative.</p>
<div class="tablewrap"><table><thead><tr><th>Cover</th><th>Title</th><th>ID</th><th>Engine</th><th>Status</th>
<th>Milestone</th><th>Links</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>
"""


def render_llms(titles: list[dict], generated: str) -> str:
    """One line per title for LLM consumers."""
    lines = [
        f"# prosper compatibility ({len(titles)} titles, generated {generated})",
        "# trackers authoritative; statuses are rung buckets 0-6.",
        "# title | id | engine | status | rung | guard | milestone | tracker-url",
    ]
    for t in titles:
        guard = ",".join(t["guard"]) if t["guard"] else "-"
        rung = str(t["rung"]) if t["rung"] is not None else "-"
        milestone = re.sub(r"\s+", " ", t["milestone"])[:300]
        lines.append(
            f"{t['title']} | {t['id']} | {t['engine']} | {t['status']} | "
            f"{rung} | {guard} | {milestone} | {t['tracker']['url']}"
        )
    return "\n".join(lines) + "\n"


LOGO_SRC = "assets/prosper-logo.png"


def vendor_logo(root: Path, out: Path) -> str:
    """Copy the README logo into the site root; returns its site-relative path.

    Missing logo aborts the build like a missing cover: a header pointing at an image
    outside the checkout would publish a broken brand mark.
    """
    source = root / LOGO_SRC
    if not source.is_file():
        raise ParseError(f"site logo not found in checkout: {LOGO_SRC}")
    (out / source.name).write_bytes(source.read_bytes())
    return source.name


def write_site(root: Path, out: Path) -> dict:
    """Parse the committed files and write the site artifacts. Returns the JSON payload."""
    tracker_rows = parse_tracker_table((root / "PROGRESS_TRACKER.md").read_text(encoding="utf-8"))
    compat_text = (root / "COMPATIBILITY.md").read_text(encoding="utf-8")
    compat = parse_compat_table(compat_text)
    out.mkdir(parents=True, exist_ok=True)
    vendored = vendor_covers(root, out, parse_compat_covers(compat_text))
    logo = vendor_logo(root, out)
    titles = build(tracker_rows, compat, vendored)
    for t in titles:
        t["doc_url"] = doc_url(t.pop("status_doc"))
        t["blockers"] = [b["number"] for b in t["blockers"]]
    generated = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    payload = {
        "generated": generated,
        "count": len(titles),
        "buckets": counts(titles),
        "titles": titles,
    }
    (out / "compat.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    (out / "index.html").write_text(
        render_index(titles, payload["buckets"], generated, logo), encoding="utf-8"
    )
    (out / "classic.html").write_text(render_classic(titles, generated, logo), encoding="utf-8")
    (out / "llms.txt").write_text(render_llms(titles, generated), encoding="utf-8")
    return payload


def selftest() -> None:
    """Compliant trackers/compat rows parse; seven violations fail loudly. No network."""
    good_tracker = (
        "| Title | Title ID | Rung | Ladder | FPS | Guard | Oracle | Open blockers | Tracker | Status doc |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| Demo <Game> | `PPSA00001` | 3 | `123---` | - | `demo-gameplay` | none | - | "
        "[#11](https://example.test/11) | - |\n"
    )
    good_compat = (
        "| Title | Title ID | Engine | User-visible milestone | Tracker |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| *Demo <Game>* | `PPSA00001` | Unity | ✅ First [room](https://example.test/x) | "
        "[#11](https://example.test/11) |\n"
    )
    good_sections = (
        "## Demo Game — `PPSA00001`\n\n"
        '<p align="center"><img src="assets/screenshots/demo-title.webp" alt="t"></p>\n'
        '<p align="center"><img src="assets/screenshots/demo-second.webp" alt="t"></p>\n\n'
        "## Next — `PPSA00002`\n\n"
        '<p align="center"><img src="https://example.test/ext.png" alt="t"></p>\n'
    )
    rows = parse_tracker_table(good_tracker)
    comp = parse_compat_table(good_compat)
    assert rows[0]["guard"] == ["demo-gameplay"], rows
    assert comp["PPSA00001"]["engine"] == "Unity", comp
    assert comp["PPSA00001"]["milestone"] == "✅ First room", comp
    covers = parse_compat_covers(good_sections)
    assert covers == {"PPSA00001": "assets/screenshots/demo-title.webp"}, covers
    titles = build(rows, comp, {"PPSA00001": "covers/demo-title.webp"})
    assert titles[0]["status"] == "Playable" and titles[0]["rung"] == 3, titles
    assert titles[0]["hint"].startswith("Rung 3 of 6"), titles
    assert titles[0]["cover"] == "covers/demo-title.webp", titles
    classic = render_classic(titles, "2026-01-01")
    assert "Demo &lt;Game&gt;" in classic and "Demo <Game>" not in classic, classic
    evil = build(
        rows, {"PPSA00001": {**comp["PPSA00001"], "milestone": "<script>alert(1)</script>"}}
    )
    # index.html carries data via compat.json and renders client-side: pin that the
    # milestone goes through the escaper, and that the static page escapes server-side.
    index_html = render_index(evil, counts(evil), "2026-01-01", "prosper-logo.png")
    assert "e(t.milestone" in index_html and "<script>alert" not in index_html
    assert 'class="logo"' in index_html and 'rel="icon"' in index_html
    # `??` next to `||` is a SyntaxError that kills the whole page script (empty table),
    # so the template must not contain it; backticks must balance for the same reason.
    assert "??" not in index_html.split("<script>")[1]
    assert index_html.count("`") % 2 == 0
    assert "&lt;script&gt;" in render_classic(evil, "2026-01-01")

    bad_trackers = [
        good_tracker.replace("`PPSA00001`", "`BOGUS`"),  # bad title id
        good_tracker.replace("| 3 |", "| 7 |"),  # rung out of range
        good_tracker.replace("`123---`", "`12x---`"),  # bad ladder
        good_tracker.replace("[#11](https://example.test/11) | - |", "- | - |"),  # no tracker link
    ]
    bad_header = good_compat.replace(
        "| Title | Title ID | Engine | User-visible milestone | Tracker |",
        "| Title | Title ID | Engine | Milestone | Tracker |",
    )
    bad_compats = [
        good_compat.replace("`PPSA00001`", "`PPSA1`"),  # bad id cell
        good_compat.replace("[#11](https://example.test/11) |", "- |"),  # no tracker link
        bad_header,  # wrong header
    ]
    for i, bad in enumerate(bad_trackers):
        try:
            parse_tracker_table(bad)
        except ParseError:
            pass
        else:
            raise AssertionError(f"bad tracker {i} parsed without error")
    for i, bad in enumerate(bad_compats):
        try:
            parse_compat_table(bad)
        except ParseError:
            pass
        else:
            raise AssertionError(f"bad compat {i} parsed without error")
    try:
        bucket_of(9)
    except ParseError:
        pass
    else:
        raise AssertionError("bucket_of(9) did not fail")
    # The real committed files must parse: the selftest pins the live contract.
    live_rows = parse_tracker_table(TRACKER_MD.read_text(encoding="utf-8"))
    live_compat_text = COMPAT_MD.read_text(encoding="utf-8")
    live_comp = parse_compat_table(live_compat_text)
    live_covers = parse_compat_covers(live_compat_text)
    live = build(live_rows, live_comp, live_covers)
    assert len(live) == len({t["id"] for t in live}) and len(live) > 50, len(live)
    covered = sum(1 for t in live if t["cover"])
    assert covered > 50, covered
    print(
        f"selftest ok: {len(live)} live titles, {covered} covers, "
        f"{len(live_rows)} tracker rows, {len(live_comp)} compat rows"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: build the site into OUTDIR, or run --selftest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if not args.outdir:
        parser.error("the output directory is required")
    payload = write_site(args.root, args.outdir)
    print(f"compat site: {payload['count']} titles -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
