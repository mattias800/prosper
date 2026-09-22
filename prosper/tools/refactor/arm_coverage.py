"""arm_coverage.py -- say which inactive `#if` arms a refactor check never saw (#3520).

map_symbols.py builds its region and reference data from ONE libclang parse, and a parse sees one
arm of every `#if`. So every reference-based verdict the refactor tools print -- split_file.py's
"every cross-part reference resolves", promote_internal.py's dependency closure -- is a statement
about the parsed arms only. On `hle_kernel_mem.cpp` that is 48% of the file; the other 52% is the
Windows/macOS code nobody examined, and a plain `[ok]` read as covering it is how #3519 round 1
compiled on Linux with 0 errors and on MinGW with 208.

This module does not parse the other arms (that is #3520's option 3, a separate design). It makes
the limit impossible to miss: map_symbols records the preprocessor's skipped spans in the map
(`unparsed_spans`), and every tool that prints a reference verdict prints, beside it, exactly which
line ranges went unchecked and where they end up. A map that predates the field, or one whose
skipped ranges could not be measured, is reported as UNKNOWN coverage -- never as complete.

Pure, no libclang: the report is arithmetic over the map, so it is pinned by --selftest in CI.
"""

from __future__ import annotations


def region_at(regions: list[dict], line: int) -> dict | None:
    for r in regions:
        if r["start"] <= line <= r["end"]:
            return r
    return None


def unchecked_arms(map_data: dict, placement: dict[int, str] | None = None) -> list[dict]:
    """One entry per skipped span: its lines, the regions it touches, and where those regions go.

    `placement` maps a body region index to the output it is written to; replicated regions
    (preamble / namespace open / close) go to every output and are reported as `*`.
    """
    spans = map_data.get("unparsed_spans") or []
    regions = map_data.get("regions", [])
    out = []
    for s, e in spans:
        touched = [r for r in regions if r["start"] <= e and s <= r["end"]]
        dests: set[str] = set()
        for r in touched:
            if r.get("role") != "body":
                dests.add("*")
            elif placement is not None and r["index"] in placement:
                dests.add(placement[r["index"]])
            elif placement is not None:
                dests.add("(stays)")
        out.append({"start": s, "end": e, "lines": e - s + 1,
                    "regions": [r["index"] for r in touched], "outputs": sorted(dests)})
    return out


def coverage_report(map_data: dict, placement: dict[int, str] | None = None,
                    limit: int = 12) -> tuple[bool, list[str]]:
    """(complete, lines to print). `complete` is True ONLY when the map says no arm was skipped."""
    if "unparsed_spans" not in map_data:
        return False, ["  [UNCHECKED] this map predates skipped-arm recording, so which #if arms the "
                       "reference check covered is UNKNOWN -- re-run map_symbols.py"]
    spans = map_data["unparsed_spans"]
    if spans is None:
        why = map_data.get("unparsed_error") or "the skipped ranges could not be measured"
        return False, [f"  [UNCHECKED] coverage UNKNOWN: {why}"]
    if not spans:
        return True, []
    arms = unchecked_arms(map_data, placement)
    total = sum(a["lines"] for a in arms)
    file_lines = map_data.get("total_lines") or 0
    share = f" ({100.0 * total / file_lines:.1f}% of the file)" if file_lines else ""
    msg = [f"  [UNCHECKED] {len(arms)} inactive #if arm(s), {total} line(s){share}, were never "
           f"parsed -- references inside them were NOT checked:"]
    for a in arms[:limit]:
        where = f" -> {', '.join(a['outputs'])}" if a["outputs"] else ""
        msg.append(f"                lines {a['start']}-{a['end']} ({a['lines']}L), "
                   f"region(s) {a['regions']}{where}")
    if len(arms) > limit:
        msg.append(f"                ... and {len(arms) - limit} more")
    msg.append("                Build every platform whose arm is listed before trusting this "
               "result (e.g. the MinGW cross build for an `#ifdef _WIN32` arm).")
    return False, msg


def selftest() -> int:
    bad = 0

    def check(cond: bool, label: str) -> None:
        nonlocal bad
        print(f"  [{'ok' if cond else 'FAIL'}]   {label}")
        bad += 0 if cond else 1

    regions = [
        {"index": 0, "start": 1, "end": 3, "role": "preamble"},
        {"index": 1, "start": 4, "end": 10, "role": "body"},
        {"index": 2, "start": 11, "end": 20, "role": "body"},
        {"index": 3, "start": 21, "end": 22, "role": "close"},
    ]
    base = {"regions": regions, "total_lines": 22}

    ok, lines = coverage_report(base)
    check(not ok and "UNKNOWN" in lines[0], "a map with no unparsed_spans field is UNKNOWN, not complete")
    ok, lines = coverage_report({**base, "unparsed_spans": None, "unparsed_error": "no libclang"})
    check(not ok and "no libclang" in lines[0], "an unmeasurable map is UNKNOWN and says why")
    ok, lines = coverage_report({**base, "unparsed_spans": []})
    check(ok and lines == [], "only a map that measured zero skipped lines reports complete")

    m = {**base, "unparsed_spans": [[12, 15], [2, 5]]}
    ok, lines = coverage_report(m, {1: "a.cpp", 2: "b.cpp"})
    arms = unchecked_arms(m, {1: "a.cpp", 2: "b.cpp"})
    check(not ok, "any skipped span makes coverage incomplete")
    check(arms[0]["outputs"] == ["b.cpp"] and arms[0]["regions"] == [2],
          "an arm inside a body region names that region and the output it goes to")
    check(arms[1]["outputs"] == ["*", "a.cpp"],
          "an arm straddling the preamble and a body names every output (*) and the body's output")
    check("8 line(s)" in lines[0] and "36.4%" in lines[0], "the report states lines and share exactly")
    arms = unchecked_arms(m, {2: "b.cpp"})
    check(arms[1]["outputs"] == ["(stays)", "*"] or arms[1]["outputs"] == ["*", "(stays)"],
          "a body region the plan does not move is reported as staying")
    print("== PASS ==" if not bad else f"== FAIL: {bad} ==")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
