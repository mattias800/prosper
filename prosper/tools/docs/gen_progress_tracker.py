#!/usr/bin/env python3
"""Generate PROGRESS_TRACKER.md from the `tracker:game` GitHub issues.

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND
---------------------------------------------
The project holds three commitments that pull against each other:

  1. the GitHub game trackers are the source of truth for per-title progress,
  2. COMPATIBILITY.md is a user-facing overview and NOT a status store,
  3. progress must be answerable from a file checkout, without the network.

A hand-maintained third copy satisfies (3) by breaking (1): it drifts, and it drifts
silently, because nothing can tell a stale row from a current one. That is exactly the
defect #2730 documents -- an audit that had to scan 6,224 issue comments to answer
"does a hardware-oracle record exist for this title", and found nine trackers asserting
a verification that had never happened. So this file is DERIVED. The trackers stay
authoritative; the checkout gets a greppable projection of them; CI keeps the two equal.

FAIL LOUDLY, NEVER SILENTLY OMIT
--------------------------------
A generator that skips what it cannot parse produces a file that LOOKS complete and is
not -- the same shape of defect, one level further out, and harder to see because the
output is machine-made and therefore trusted. So every required field is an error when
absent, the error names the tracker, and NO FILE IS WRITTEN on any failure. The run
prints "N trackers, N parsed" so a truncated fetch is visible rather than plausible.

The distinction the tool draws, and it is deliberate:

  * REQUIRED fields are structural contracts every tracker is expected to hold. A miss
    is a hard error: the title/TITLE_ID heading, the six-rung ladder, the machine-readable
    `Oracle record:` line, and a `## Current blocker(s)` section.
  * OPTIONAL fields are genuinely absent for some titles, and an explicit "-" is the
    honest value: a title with no status document has no status document. These never
    fail the run, and the header of the generated file says which is which so a reader
    cannot mistake "-" for "the generator gave up".

`FPS record:` is optional but STRICTLY VALIDATED when it is there, and that combination is
deliberate. Making it required would break all 39 trackers at once on the day it landed, and
the tool writes no file at all on any parse failure -- so a newly required field takes the
whole projection down until every tracker is edited. But a framerate carried as free text
degrades into a bare number within a release or two, and a bare framerate is not a
measurement: it is meaningless without the resolution, what was on screen, which frontend
measured it and when. So the line is optional, `none` is an explicit absence, and anything
else must be the full form. A malformed one is a hard error that names the tracker.

NOT DERIVED, AND WHY -- read before adding a column
---------------------------------------------------
"Engine" and "latest verified master" were both considered and both deliberately LEFT
OUT. No tracker carries either in a fixed field, so both would have to be recovered by
grepping prose, and a prose grep here does not fail loudly -- it succeeds wrongly. Two
measurements, taken 2026-08-19 over the live bodies rather than assumed:

  * a `master <sha>` grep matches in only 10 of 39 trackers, and in #1892 the first
    match is prose about a past investigation rather than the verified head. A column
    filled that way would be right for some rows, silently stale for others, and
    indistinguishable between the two.
  * engine appears as free text ("Unity/IL2CPP", "UE4") in some bodies and nowhere in
    others.

If either is wanted, the fix is to add a fixed line to the tracker bodies the way
`Oracle record:` was added -- then it becomes parseable and this tool can carry it.
Inventing a heuristic instead is how the column would start lying.

USAGE
    gen_progress_tracker.py                 # fetch from GitHub, write PROGRESS_TRACKER.md
    gen_progress_tracker.py --check         # fetch, compare, exit 1 if the file is stale
    gen_progress_tracker.py --dump-json F   # fetch and save the raw input (for offline work)
    gen_progress_tracker.py --from-json F   # generate from a saved input, no network
    gen_progress_tracker.py --selftest      # offline: prove the checker still discriminates

`--selftest` runs first in CI and separately from the live check, because this tool's own
failure mode is silence: a parser that stops recognising the ladder would report every
tracker as unparseable (loud, fine) OR, if a regex quietly widened, would accept anything
and generate a plausible wrong file (silent, fatal). The selftest holds hand-written
COMPLIANT and VIOLATING inputs and asserts both verdicts, so a gate that has stopped
discriminating fails there rather than passing everything downstream.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "mattias800/prosper"
TRACKER_LABEL = "tracker:game"

# Repo-relative, resolved from this file so the tool works from any cwd.
TOOLS_DOCS = Path(__file__).resolve().parent
PROSPER = TOOLS_DOCS.parent.parent
REPO_ROOT = PROSPER.parent
SNAPSHOTS_JSON = PROSPER / "tools" / "snapshot" / "snapshots.json"
OUTPUT = REPO_ROOT / "PROGRESS_TRACKER.md"

TITLE_RE = re.compile(r"^\[Game tracker\]\s+(?P<name>.+?)\s+\((?P<tid>PPSA\d{5})\)\s*$")
LADDER_HEADING_RE = re.compile(r"^#{2,3}\s+Progress ladder\s*$", re.M)
RUNG_RE = re.compile(r"^-\s+\[(?P<mark>[ xX])\]\s+Rung\s+(?P<n>\d+)\b", re.M)
ORACLE_RE = re.compile(r"^Oracle record:\s*(?P<value>\S.*?)\s*$", re.M)
FPS_RE = re.compile(r"^FPS record:\s*(?P<value>\S.*?)\s*$", re.M)

# The one accepted shape of a non-`none` FPS record. Semicolons separate the fields because a
# GFM table cell splits on `|` and a comma is too likely to appear inside a scene description.
#
#   FPS record: 3.4 distinct / 59.8 presented at 3840x2160; gameplay; screenshot; 2026-08-21
#
# `distinct` counts guest frames whose CONTENT changed; `presented` counts publications. Both are
# required and the record is rejected if distinct exceeds presented, because that ordering is the
# entire point: prosper re-publishes its retained frame when a submit produces no present source, so
# a presented rate reads full speed for a frozen title (prosper/src/gpu/present/present_frame_rate.hpp,
# instrument trap 90, #2783). A record carrying only one number could not show that, and a reader
# would have no way to tell which number they had been given.
FPS_RECORD_RE = re.compile(
    r"^(?P<distinct>\d+(?:\.\d+)?)\s+distinct\s*/\s*(?P<presented>\d+(?:\.\d+)?)\s+presented"
    r"\s+at\s+(?P<width>\d+)x(?P<height>\d+)"
    r"\s*;\s*(?P<scene>[^;]+?)"
    r"\s*;\s*(?P<frontend>[^;]+?)"
    r"\s*;\s*(?P<date>\d{4}-\d{2}-\d{2})$"
)

FPS_FORM = ("FPS record: <distinct> distinct / <presented> presented at <W>x<H>; "
            "<what was running>; <frontend>; <YYYY-MM-DD>")

# Common resolutions get their usual short name so the column stays narrow; anything else is
# printed verbatim rather than rounded to the nearest familiar label.
RESOLUTION_NAMES = {
    (3840, 2160): "4K",
    (2560, 1440): "1440p",
    (1920, 1080): "1080p",
    (1600, 900): "900p",
    (1280, 720): "720p",
}
BLOCKER_HEADING_RE = re.compile(r"^#{2,3}\s+Current blockers?\s*$", re.M | re.I)
ANY_HEADING_RE = re.compile(r"^#{1,6}\s+", re.M)
ISSUE_REF_RE = re.compile(r"#(\d{2,6})\b")
DOCS_PATH_RE = re.compile(r"prosper/docs/[A-Za-z0-9_.\-]+\.md")

EXPECTED_RUNGS = 6

RUNG_NAMES = {
    1: "any real graphics",
    2: "title screen",
    3: "gameplay with real GPU draws",
    4: "manual visual verification",
    5: "PS5 hardware-oracle comparison",
    6: "reviewed automatic gameplay snapshot guard",
}


class ParseError(Exception):
    """A required field is missing or malformed. Names the tracker; aborts the run."""


# --------------------------------------------------------------------------------------
# Input acquisition
# --------------------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "error: command failed: %s\n  exit %d\n  stderr: %s"
            % (" ".join(cmd), proc.returncode, proc.stderr.strip())
        )
    return proc.stdout


def _require_gh() -> None:
    """Refuse to run networkless rather than degrading to a partial or empty result.

    A generator that shrugs at a missing `gh` writes a shorter file and exits 0, which is
    the exact failure this tool exists to prevent -- so the absence of the fetch mechanism
    is a hard error, never an empty table.
    """
    if shutil.which("gh") is None:
        raise SystemExit(
            "error: `gh` is not on PATH, and this tool reads the trackers through it.\n"
            "       Install the GitHub CLI and authenticate (`gh auth login`, or set\n"
            "       GH_TOKEN), or run offline against a saved input with --from-json."
        )


def fetch_input() -> dict:
    """Fetch every tracker plus the open/closed state of every issue they cite.

    Two network calls total, regardless of tracker count: one `gh issue list` for the
    bodies, and one batched GraphQL query for the referenced issue states. Kept small
    deliberately -- this runs in CI on every push.
    """
    _require_gh()
    raw = _run(
        [
            "gh", "issue", "list",
            "--repo", REPO,
            "--label", TRACKER_LABEL,
            "--state", "all",
            "--limit", "500",
            "--json", "number,title,body,url,state",
        ]
    )
    trackers = json.loads(raw)
    if not trackers:
        raise SystemExit(
            "error: the tracker query returned 0 issues. That is never correct for this\n"
            "       repository, so it is reported as a failure rather than written out as\n"
            "       an empty table (a truncated fetch must not look like a finished one)."
        )

    referenced: set[int] = set()
    for t in trackers:
        for num in _blocker_refs(t["body"], t["number"]):
            referenced.add(num)

    return {
        "trackers": sorted(trackers, key=lambda t: t["number"]),
        "ref_states": resolve_ref_states(sorted(referenced)),
    }


def resolve_ref_states(numbers: list[int]) -> dict[str, str]:
    """Map issue/PR number -> "OPEN"/"CLOSED"/"MERGED", batched.

    `issueOrPullRequest` is used rather than `issue` because a blocker line may legitimately
    cite a PR, and a lookup that only knows about issues would report a real PR as missing.
    A number that resolves to nothing is an error, not a blank -- a citation to an issue
    that does not exist is a defect in the tracker, and this is the only place it is visible.
    """
    states: dict[str, str] = {}
    for chunk_start in range(0, len(numbers), 50):
        chunk = numbers[chunk_start : chunk_start + 50]
        fields = "\n".join(
            'n%d: issueOrPullRequest(number: %d) { '
            "... on Issue { state } ... on PullRequest { state } }" % (n, n)
            for n in chunk
        )
        query = 'query { repository(owner: "%s", name: "%s") { %s } }' % (
            REPO.split("/")[0],
            REPO.split("/")[1],
            fields,
        )
        out = _run(["gh", "api", "graphql", "-f", "query=" + query])
        data = json.loads(out)
        repo = (data.get("data") or {}).get("repository") or {}
        for n in chunk:
            node = repo.get("n%d" % n)
            if not node or "state" not in node:
                raise SystemExit(
                    "error: a tracker cites #%d, which does not resolve to any issue or\n"
                    "       pull request in %s. Fix the citation in the tracker body."
                    % (n, REPO)
                )
            states[str(n)] = node["state"]
    return states


# --------------------------------------------------------------------------------------
# Parsing -- every failure here names the tracker and aborts
# --------------------------------------------------------------------------------------


def _section(body: str, heading_re: re.Pattern) -> str | None:
    """Return the text under the first heading matching `heading_re`, up to the next heading."""
    m = heading_re.search(body)
    if not m:
        return None
    rest = body[m.end() :]
    nxt = ANY_HEADING_RE.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _blocker_refs(body: str, self_number: int) -> list[int]:
    """Issue/PR numbers cited in the tracker's blockers section, in first-seen order.

    Deliberately scoped to that one section rather than the whole body: trackers cite
    issues throughout (route notes, ruled-out history, PR links), and a whole-body scan
    would report resolved history as a live blocker.
    """
    section = _section(body, BLOCKER_HEADING_RE)
    if section is None:
        return []
    seen: list[int] = []
    for m in ISSUE_REF_RE.finditer(section):
        n = int(m.group(1))
        if n != self_number and n not in seen:
            seen.append(n)
    return seen


def parse_fps_record(body: str, where: str, name: str) -> dict | None:
    """Parse `FPS record:` -- absent, `none`, or the full form. Anything else is an error.

    Returns None for "no line at all" and {"none": True} for an explicit `none`. The two are
    rendered differently on purpose: `-` means nobody has written the line, `none` means somebody
    looked and there is no measurement. That distinction is exactly what `Oracle record:` exists to
    make for hardware comparisons (#2730), and it is worth as much here -- "we have never measured
    this title" and "this title has no framerate worth recording" are different states, and a single
    blank cell would collapse them.
    """
    lines = FPS_RE.findall(body)
    if not lines:
        return None
    if len(lines) > 1:
        raise ParseError(
            "%s (%s): found %d 'FPS record:' lines, expected at most 1.\n"
            "        Two records cannot both be current, and this tool will not guess which."
            % (where, name, len(lines))
        )
    value = lines[0].strip()
    if value == "none":
        return {"none": True}

    m = FPS_RECORD_RE.match(value)
    if not m:
        raise ParseError(
            "%s (%s): the 'FPS record:' line does not match the required form.\n"
            "        Found: %r\n"
            "        Wanted: %s\n"
            "        Or 'FPS record: none' if this title has no measurement. A bare number is\n"
            "        rejected deliberately: a framerate without its resolution, its scene, the\n"
            "        frontend that measured it and the date is not a measurement anybody can use."
            % (where, name, value, FPS_FORM)
        )
    distinct = float(m.group("distinct"))
    presented = float(m.group("presented"))
    if distinct > presented:
        raise ParseError(
            "%s (%s): the 'FPS record:' line reports %.1f distinct frames per second against\n"
            "        %.1f presented. Distinct publications are a SUBSET of publications, so this\n"
            "        is either a transposed pair or two numbers from different runs."
            % (where, name, distinct, presented)
        )
    return {
        "none": False,
        "distinct": distinct,
        "presented": presented,
        "width": int(m.group("width")),
        "height": int(m.group("height")),
        "scene": m.group("scene").strip(),
        "frontend": m.group("frontend").strip(),
        "date": m.group("date"),
    }


def parse_tracker(issue: dict, guards_by_title: dict[str, list[str]]) -> dict:
    number = issue["number"]
    where = "tracker #%d" % number

    tm = TITLE_RE.match(issue["title"].strip())
    if not tm:
        raise ParseError(
            "%s: the issue title %r does not match the required form\n"
            "        '[Game tracker] <Name> (PPSAxxxxx)'. Every tracker's title and title\n"
            "        ID are read from it, so a free-form title has no parse."
            % (where, issue["title"])
        )
    name, title_id = tm.group("name"), tm.group("tid")

    body = issue["body"] or ""

    # --- required: the six-rung ladder ------------------------------------------------
    ladder_section = _section(body, LADDER_HEADING_RE)
    if ladder_section is None:
        raise ParseError(
            "%s (%s): no '## Progress ladder' heading. The rung is the single most\n"
            "        load-bearing field in this table; a tracker without a ladder has no\n"
            "        rung, and a blank cell would read as rung 0." % (where, name)
        )
    rungs = RUNG_RE.findall(ladder_section)
    if len(rungs) != EXPECTED_RUNGS:
        raise ParseError(
            "%s (%s): the ladder has %d rung lines, expected exactly %d.\n"
            "        Found: %s"
            % (where, name, len(rungs), EXPECTED_RUNGS, ", ".join(n for _, n in rungs) or "none")
        )
    # The order check below SUBSUMES the count check above -- a wrong count can never equal
    # 1..6 -- and that is stated rather than left to be rediscovered, because it means a
    # mutation of the count guard alone does not move the selftest. The count guard is kept
    # only for its error message, which names the actual number found; the ORDER guard is the
    # load-bearing one, and it is the one the selftest's mutation arm exercises.
    numbers = [int(n) for _, n in rungs]
    if numbers != list(range(1, EXPECTED_RUNGS + 1)):
        raise ParseError(
            "%s (%s): the ladder rungs are %s, expected 1..%d in order."
            % (where, name, numbers, EXPECTED_RUNGS)
        )
    ticked = [mark.lower() == "x" for mark, _ in rungs]

    # --- required: the machine-readable oracle field ----------------------------------
    oracles = ORACLE_RE.findall(body)
    if len(oracles) != 1:
        raise ParseError(
            "%s (%s): found %d 'Oracle record:' lines, expected exactly 1.\n"
            "        This field is the whole point of the #2730 convention: it makes a null\n"
            "        checkable in one command instead of 6,224 issue-comment scans. Add\n"
            "        'Oracle record: none' if there genuinely is no hardware comparison."
            % (where, name, len(oracles))
        )
    oracle = oracles[0].strip()

    # --- optional, but strictly validated when present: the framerate record --------------
    fps = parse_fps_record(body, where, name)

    # --- required: a blockers section (its CONTENT may legitimately be empty) ----------
    if _section(body, BLOCKER_HEADING_RE) is None:
        raise ParseError(
            "%s (%s): no '## Current blocker' / '## Current blockers' heading. A tracker\n"
            "        with no blockers section is indistinguishable from one whose blockers\n"
            "        were dropped, so the heading is required even when it says 'none'."
            % (where, name)
        )

    # --- optional: status doc ---------------------------------------------------------
    docs = DOCS_PATH_RE.findall(body)
    status_docs = [d for d in docs if d.endswith("_STATUS.md")]
    status_doc = (status_docs or docs or [None])[0]

    return {
        "number": number,
        "name": name,
        "title_id": title_id,
        "url": issue["url"],
        "issue_state": issue["state"],
        "ticked": ticked,
        "rung": max([i + 1 for i, t in enumerate(ticked) if t], default=0),
        "ladder": "".join(str(i + 1) if t else "-" for i, t in enumerate(ticked)),
        "oracle": oracle,
        "fps": fps,
        "blocker_refs": _blocker_refs(body, number),
        "guards": guards_by_title.get(title_id, []),
        "status_doc": status_doc,
    }


def load_guards(snapshots_path: Path) -> dict[str, list[str]]:
    """Map PPSA title ID -> guard names, read from the snapshot registry.

    File-derived rather than tracker-derived on purpose: `snapshots.json` is the thing that
    actually decides whether a guard exists, so a tracker's prose claim about one cannot
    make this column disagree with reality.
    """
    if not snapshots_path.is_file():
        raise SystemExit("error: snapshot registry not found at %s" % snapshots_path)
    data = json.loads(snapshots_path.read_text())
    out: dict[str, list[str]] = {}
    for entry in data["snapshots"]:
        dump = entry.get("dump", "")
        m = re.search(r"(PPSA\d{5})", dump)
        if not m:
            raise SystemExit(
                "error: snapshot %r has no PPSA title id in its 'dump' field (%r), so its\n"
                "       guard cannot be attributed to a title."
                % (entry.get("name", "?"), dump)
            )
        out.setdefault(m.group(1), []).append(entry["name"])
    return {k: sorted(v) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

HEADER = """# Progress tracker

<!--
    GENERATED FILE -- DO NOT EDIT BY HAND.
    Produced by prosper/tools/docs/gen_progress_tracker.py from the `tracker:game` GitHub
    issues. Regenerate with:  python3 prosper/tools/docs/gen_progress_tracker.py
    CI regenerates this file and fails if it differs from the committed copy.
-->

**This file is generated. The GitHub game trackers are authoritative.**
It is produced by [`prosper/tools/docs/gen_progress_tracker.py`](prosper/tools/docs/gen_progress_tracker.py)
from the `tracker:game` issues, and exists so per-title progress is answerable from a
checkout -- with `grep`, offline, in one command -- rather than only through the GitHub API.
Editing a row here changes nothing: **edit the tracker issue**, then regenerate.

```bash
python3 prosper/tools/docs/gen_progress_tracker.py          # rewrite this file
python3 prosper/tools/docs/gen_progress_tracker.py --check   # fail if it is stale
```

It is a projection, not a second source of truth, and it is deliberately narrow: it carries
only fields the trackers hold in a **fixed, parseable form**. `COMPATIBILITY.md` remains the
user-facing overview, and its markers are a chart, not a rung scale.

> **What this file can be wrong about, and it is not a defect in the generator.**
> It reads each tracker's **body**. The charter records that a tracker body is routinely days
> behind its own comment thread, and that a fresh `updated_at` does not mean the body was
> re-read -- the timestamp moves on every comment. So a row here is exactly as current as the
> body it came from, and no more. This file makes that staleness **visible and greppable**
> instead of leaving it to be rediscovered per reader, but it cannot repair it: the repair is
> editing the tracker body. **Before quoting a rung from this table, read the tracker's last
> comments**, and prefer the title's status doc when the two disagree.
>
> A dated instance, found the day this caveat was written. #2736 merged at `22:58:42Z` on
> 2026-08-19, putting *Asterix & Obelix: Babylon Mission* gameplay on master. Tracker #1884's
> `updatedAt` then moved to `23:16:55Z` -- **after** the merge -- while its ladder still read
> rung 2, because what changed was a comment and nobody had re-read the body. It was ticked to
> rung 3 at `23:29:19Z`, once a human opened the screenshot and confirmed the scene. For those
> 31 minutes the tracker was demonstrably stale **and** demonstrably fresh-looking, and every
> timestamp-based freshness audit would have passed it. That is the whole reason the rule above
> is "read the comments", not "check the timestamp".

## How to read the columns

| Column | Derivation |
| --- | --- |
| **Rung** | The highest **ticked** rung, 0 if none. |
| **Ladder** | Every ticked rung, `-` for unticked. **The ladder is legitimately non-contiguous** on some titles -- PR #1696 and #1676 deliberately took titles from rung 3/4 to rung 6 without rung 5, because a reviewed gameplay guard is evidenced by its own route and thresholds and never depended on a hardware oracle. `1234-6` is a real state, not an editing slip. |
| **Guard** | From `prosper/tools/snapshot/snapshots.json`, matched on title ID -- not from the tracker's prose, so it cannot disagree with the registry. |
| **FPS** | The tracker's `FPS record:` line. `-` means **no tracker line exists**; `none` means somebody looked and there is no measurement. See below. |
| **Oracle** | The tracker's `Oracle record:` line, verbatim. `none` means **no PS5 hardware comparison is on record** (see #2730). Not to be confused with `snapshots.json`'s `structural_references`, which are luminance signatures generated from prosper's own runs -- a *regression* reference, not a hardware oracle. |
| **Open blockers** | Issues/PRs cited in the tracker's `## Current blocker(s)` section that are still open. Cited-and-closed entries are omitted; a tracker citing nothing shows `-`. |
| **Status doc** | First `prosper/docs/*_STATUS.md` referenced by the tracker, else its first `prosper/docs/*.md`. `-` means the tracker references neither. |

A `-` is an **explicit absence**, never a parse failure: the generator aborts on anything it
cannot parse and writes no file at all, so a row that is present is a row that was read.

### The FPS column: two numbers, and the first one is the honest one

`**3.4** / 59.8 fps` reads **3.4 distinct frames per second, 59.8 publications per second**, and the
gap between them is information rather than noise. prosper re-publishes the frame it retained
whenever a submit produces no usable present source, and that re-serve goes through the ordinary
publish path -- so a title whose picture is completely frozen keeps publishing at the display's
rate. A framerate counted from publications reads **full speed for a frozen title**; that is
instrument trap 90, and it is exactly the R-Type Delta regression #2783, which hid for nine days
behind a healthy-looking present rate. **Quote the first number. When the two are far apart, the
title has a defect and neither number is its framerate.**
(`prosper/src/gpu/present/present_frame_rate.hpp` carries the argument in full.)

The rest of the cell is not decoration: a framerate means nothing without its conditions. Resolution,
what was on screen, and which frontend measured it all move the number by more than the differences
anybody is trying to see, and a date is what stops a figure from a fixed-since regression being read
as current.

To record one, add exactly one line anywhere in the tracker body:

```
FPS record: 3.4 distinct / 59.8 presented at 3840x2160; gameplay; screenshot; 2026-08-21
FPS record: none
```

The line is **optional** -- a tracker without one renders `-` and parses fine -- but it is **strictly
validated when present**, and a malformed one fails the whole run and names the tracker. That is on
purpose: a required field would take the entire projection down the day it landed, while a loosely
parsed one would decay into bare numbers, and a bare framerate is not a measurement. Get both rates
from `tools/screenshot`'s summary line or the `distinct_fps` / `presented_fps` fields of its manifest,
or from `prosper-app --fps`.

"""

FOOTER_TEMPLATE = """
## Counts

| Highest rung reached | Titles |
| --- | --- |
{rung_rows}

**{n_oracle} of {n_total}** trackers record a PS5 hardware-oracle comparison; the rest carry
`Oracle record: none`. That ratio is the reason this column exists -- before #2730 it took a
scan of 6,224 issue comments to establish, and it was wrong by nine titles.
"""


def _cell(text: str) -> str:
    """Escape a value for a GFM table cell.

    GFM splits a row on `|` BEFORE parsing inline content, so an unescaped pipe -- including
    one inside a code span -- silently discards the excess cells on the rendered page. The
    repo's own docs gate exists because six rows were truncated that way for months.
    """
    return text.replace("|", "\\|")


def _fps_cell(fps: dict | None) -> str:
    """Render one framerate record for the table.

    BOTH rates are shown, distinct first and bold. Showing only the distinct rate would be tidier
    and would throw away the diagnostic: a large gap between the two means prosper was re-publishing
    a retained frame rather than the title running fast, which is a defect the table can surface for
    free. No threshold is applied here on purpose -- a judgment call encoded in the generator would
    be a second, drifting copy of the one in
    prosper/src/gpu/present/present_frame_rate.hpp. The reader compares two numbers.
    """
    if fps is None:
        return "-"
    if fps["none"]:
        return "none"
    resolution = RESOLUTION_NAMES.get(
        (fps["width"], fps["height"]), "%dx%d" % (fps["width"], fps["height"])
    )
    return "**%.1f** / %.1f fps · %s · %s · %s · %s" % (
        fps["distinct"], fps["presented"], resolution, fps["scene"], fps["frontend"], fps["date"]
    )


def _issue_link(n: int) -> str:
    return "[#%d](https://github.com/%s/issues/%d)" % (n, REPO, n)


def render(records: list[dict], ref_states: dict[str, str]) -> str:
    rows = []
    for r in sorted(records, key=lambda r: (-r["rung"], r["name"].lower())):
        open_blockers = [
            n for n in r["blocker_refs"] if ref_states.get(str(n), "CLOSED") == "OPEN"
        ]
        guards = ", ".join("`%s`" % g for g in r["guards"]) or "-"

        oracle = r["oracle"]
        if oracle == "none":
            oracle_cell = "none"
        elif oracle.startswith("http"):
            oracle_cell = "[comment](%s)" % oracle
        else:
            oracle_cell = "[`%s`](%s)" % (Path(oracle).name, oracle)

        status = (
            "[`%s`](%s)" % (Path(r["status_doc"]).name, r["status_doc"])
            if r["status_doc"]
            else "-"
        )

        rows.append(
            "| %s | `%s` | %d | `%s` | %s | %s | %s | %s | [#%d](%s) | %s |"
            % (
                _cell(r["name"]),
                r["title_id"],
                r["rung"],
                r["ladder"],
                _cell(_fps_cell(r["fps"])),
                _cell(guards),
                _cell(oracle_cell),
                ", ".join(_issue_link(n) for n in open_blockers) or "-",
                r["number"],
                r["url"],
                _cell(status),
            )
        )

    table = (
        "| Title | Title ID | Rung | Ladder | FPS | Guard | Oracle | Open blockers | Tracker | Status doc |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"
    )

    by_rung: dict[int, int] = {}
    for r in records:
        by_rung[r["rung"]] = by_rung.get(r["rung"], 0) + 1
    rung_rows = "\n".join(
        "| %d -- %s | %d |" % (k, RUNG_NAMES.get(k, "not started"), by_rung[k])
        for k in sorted(by_rung, reverse=True)
    )

    footer = FOOTER_TEMPLATE.format(
        rung_rows=rung_rows,
        n_oracle=sum(1 for r in records if r["oracle"] != "none"),
        n_total=len(records),
    )
    return HEADER + table + footer


def build(payload: dict) -> tuple[str, int, int]:
    guards = load_guards(SNAPSHOTS_JSON)
    trackers = payload["trackers"]
    records, errors = [], []
    for issue in trackers:
        try:
            records.append(parse_tracker(issue, guards))
        except ParseError as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit(
            "error: %d of %d trackers could not be parsed. NO FILE WAS WRITTEN -- a\n"
            "       generated file that silently omits a tracker is worse than no file,\n"
            "       because it looks complete.\n\n%s"
            % (len(errors), len(trackers), "\n\n".join("  " + e for e in errors))
        )
    return render(records, payload["ref_states"]), len(trackers), len(records)


# --------------------------------------------------------------------------------------
# Selftest -- compliant AND violating inputs, so a gate that stopped firing fails here
# --------------------------------------------------------------------------------------

_GOOD_BODY = """## Current milestone

**Rung 6.**

## Current blockers

- [#4242](https://github.com/mattias800/prosper/issues/4242) - an open thing.
- [#4243](https://github.com/mattias800/prosper/issues/4243) - a closed thing.

Technical record: prosper/docs/EXAMPLE_STATUS.md

## Progress ladder

- [x] Rung 1 - any real graphics
- [x] Rung 2 - title screen
- [x] Rung 3 - gameplay with real GPU draws
- [x] Rung 4 - manual visual verification
- [ ] Rung 5 - PS5 hardware-oracle comparison
- [x] Rung 6 - reviewed automatic gameplay snapshot guard

Oracle record: none
"""


def _issue(number=9001, title="[Game tracker] Example (PPSA99999)", body=_GOOD_BODY):
    return {
        "number": number,
        "title": title,
        "body": body,
        "url": "https://github.com/%s/issues/%d" % (REPO, number),
        "state": "OPEN",
    }


# Column ordinal of the FPS cell in a rendered data row, counting the empty field before the leading
# `|`. Kept beside render() so the two move together.
_FPS_COLUMN = 5


def _rendered_fps_cell(record: dict) -> str:
    """The FPS cell of `record`'s row, extracted BY POSITION.

    This is positional rather than a substring search over `render()`'s output, and that is not
    fastidiousness -- the substring version was written first and was VOID. `render()` returns
    HEADER + table + footer, and the header documents this column with a worked example, so
    `"**3.4** / 59.8 fps" in render(...)` matched the prose no matter what the row contained: it
    passed unchanged under a mutation that made the cell show only the presented rate. The same
    applies to `"| - |"` and `"| none |"`, which the Guard, Oracle and Open-blockers columns produce
    for the very fixture this arm uses.
    """
    rendered = render([record], {"4242": "OPEN", "4243": "CLOSED"})
    for line in rendered.splitlines():
        if line.startswith("| ") and record["title_id"] in line:
            return line.split("|")[_FPS_COLUMN].strip()
    return "<no row rendered>"


def _parses(issue) -> tuple[bool, str]:
    try:
        return True, "" if parse_tracker(issue, {}) else ""
    except ParseError as exc:
        return False, str(exc)


def selftest() -> int:
    """Assert the parser ACCEPTS a compliant tracker and REJECTS each violation.

    The accept arm is not decoration. A checker whose regexes have quietly widened accepts
    everything and reports a clean tree forever; a checker whose regexes have quietly
    narrowed rejects everything, which at least is loud. Only running both directions
    separates "still discriminating" from "still running".
    """
    failures: list[str] = []

    ok, err = _parses(_issue())
    if not ok:
        failures.append("compliant tracker was REJECTED: %s" % err)

    # A tracker with NO `FPS record:` line must still parse. This arm is the one that stops the
    # field being quietly promoted to required: 39 trackers have no line, and the tool writes no file
    # at all on any parse failure, so a required field takes the whole projection down on day one.
    ok, err = _parses(_issue(body=_GOOD_BODY))
    if not ok:
        failures.append("a tracker with no FPS record was REJECTED: %s" % err)
    if parse_tracker(_issue(), {})["fps"] is not None:
        failures.append("a tracker with no FPS record did not parse as absent")

    fps_line = ("FPS record: 3.4 distinct / 59.8 presented at 3840x2160; gameplay; "
                "screenshot; 2026-08-21\n")
    recorded = parse_tracker(_issue(body=_GOOD_BODY + fps_line), {})["fps"]
    for field, got, want in [
        ("fps.distinct", recorded["distinct"], 3.4),
        ("fps.presented", recorded["presented"], 59.8),
        ("fps.width", recorded["width"], 3840),
        ("fps.height", recorded["height"], 2160),
        ("fps.scene", recorded["scene"], "gameplay"),
        ("fps.frontend", recorded["frontend"], "screenshot"),
        ("fps.date", recorded["date"], "2026-08-21"),
    ]:
        if got != want:
            failures.append("parsed %s = %r, expected %r" % (field, got, want))

    none_record = parse_tracker(_issue(body=_GOOD_BODY + "FPS record: none\n"), {})["fps"]
    if none_record != {"none": True}:
        failures.append("'FPS record: none' did not parse as an explicit absence")

    # `-` and `none` must RENDER differently, or the distinction the field exists to make is lost at
    # the last step. This is the arm a "simplification" that collapsed them would fail.
    absent_cell = _rendered_fps_cell(parse_tracker(_issue(), {}))
    none_cell = _rendered_fps_cell(
        parse_tracker(_issue(body=_GOOD_BODY + "FPS record: none\n"), {}))
    if absent_cell != "-":
        failures.append("a tracker with no FPS record rendered as %r, expected '-'" % absent_cell)
    if none_cell != "none":
        failures.append("'FPS record: none' rendered as %r, expected 'none'" % none_cell)

    # Both numbers, distinct first and bold. A cell carrying only the presented rate would report a
    # frozen title as fast, which is the whole reason this column stores two numbers (#2783).
    recorded_cell = _rendered_fps_cell(parse_tracker(_issue(body=_GOOD_BODY + fps_line), {}))
    if not recorded_cell.startswith("**3.4** / 59.8 fps"):
        failures.append("the FPS cell is %r; it must LEAD with the distinct rate" % recorded_cell)
    for condition in ("4K", "gameplay", "screenshot", "2026-08-21"):
        if condition not in recorded_cell:
            failures.append("the FPS cell dropped %r -- the number is meaningless without it"
                            % condition)

    violations = [
        (
            "free-form issue title",
            _issue(title="Blue Prince progress"),
        ),
        (
            "a bare FPS number with no conditions",
            _issue(body=_GOOD_BODY + "FPS record: 3.4\n"),
        ),
        (
            "an FPS record missing its date",
            _issue(body=_GOOD_BODY +
                   "FPS record: 3.4 distinct / 59.8 presented at 3840x2160; gameplay; screenshot\n"),
        ),
        (
            "an FPS record with only one rate",
            _issue(body=_GOOD_BODY +
                   "FPS record: 3.4 fps at 3840x2160; gameplay; screenshot; 2026-08-21\n"),
        ),
        (
            "more distinct frames than published ones",
            _issue(body=_GOOD_BODY +
                   "FPS record: 59.8 distinct / 3.4 presented at 3840x2160; gameplay; "
                   "screenshot; 2026-08-21\n"),
        ),
        (
            "two FPS records",
            _issue(body=_GOOD_BODY + fps_line + fps_line),
        ),
        (
            "no ladder heading",
            _issue(body=_GOOD_BODY.replace("## Progress ladder", "## Ladder of progress")),
        ),
        (
            "a rung line deleted",
            _issue(body=_GOOD_BODY.replace("- [x] Rung 3 - gameplay with real GPU draws\n", "")),
        ),
        (
            "rungs out of order",
            _issue(body=_GOOD_BODY.replace("Rung 3 - gameplay", "Rung 7 - gameplay")),
        ),
        (
            "no Oracle record line",
            _issue(body=_GOOD_BODY.replace("Oracle record: none\n", "")),
        ),
        (
            "two Oracle record lines",
            _issue(body=_GOOD_BODY + "\nOracle record: none\n"),
        ),
        (
            "no blockers heading",
            _issue(body=_GOOD_BODY.replace("## Current blockers", "## Things in the way")),
        ),
    ]
    for label, issue in violations:
        ok, _ = _parses(issue)
        if ok:
            failures.append("violation was ACCEPTED (%s) -- the gate is not discriminating" % label)

    # The parse must also read the ladder CORRECTLY, not merely accept it. A parser that
    # accepts every input and returns rung 6 for all of them passes every arm above.
    rec = parse_tracker(_issue(), {"PPSA99999": ["example-guard"]})
    checks = [
        ("rung", rec["rung"], 6),
        ("ladder", rec["ladder"], "1234-6"),
        ("title_id", rec["title_id"], "PPSA99999"),
        ("name", rec["name"], "Example"),
        ("oracle", rec["oracle"], "none"),
        ("fps", rec["fps"], None),
        ("blocker_refs", rec["blocker_refs"], [4242, 4243]),
        ("guards", rec["guards"], ["example-guard"]),
        ("status_doc", rec["status_doc"], "prosper/docs/EXAMPLE_STATUS.md"),
    ]
    for field, got, want in checks:
        if got != want:
            failures.append("parsed %s = %r, expected %r" % (field, got, want))

    # Closed citations must be dropped from the rendered row, or the column reports
    # resolved work as a live blocker.
    out = render([rec], {"4242": "OPEN", "4243": "CLOSED"})
    if "/issues/4242)" not in out:
        failures.append("an OPEN blocker was dropped from the rendered table")
    if "/issues/4243)" in out:
        failures.append("a CLOSED blocker was rendered as an open blocker")

    if failures:
        print("selftest FAILED:", file=sys.stderr)
        for f in failures:
            print("  - %s" % f, file=sys.stderr)
        return 1
    print("selftest passed: 1 compliant input accepted, %d violations rejected, "
          "%d parsed fields correct" % (len(violations), len(checks)))
    return 0


# --------------------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare against the committed file; exit 1 if stale")
    ap.add_argument("--selftest", action="store_true",
                    help="offline: assert the parser still accepts valid and rejects invalid input")
    ap.add_argument("--from-json", metavar="FILE",
                    help="generate from a saved input payload instead of the network")
    ap.add_argument("--dump-json", metavar="FILE",
                    help="save the fetched input payload (for offline reruns)")
    ap.add_argument("--output", metavar="FILE", default=str(OUTPUT),
                    help="where to write (default: PROGRESS_TRACKER.md at the repo root)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    payload = json.loads(Path(args.from_json).read_text()) if args.from_json else fetch_input()
    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    text, n_trackers, n_parsed = build(payload)
    print("%d trackers, %d parsed" % (n_trackers, n_parsed))

    out = Path(args.output)
    if args.check:
        if not out.is_file():
            print("error: %s does not exist. Run the generator and commit it." % out.name,
                  file=sys.stderr)
            return 1
        if out.read_text() != text:
            print(
                "error: %s is STALE -- it does not match the trackers it is generated from.\n"
                "       The trackers are authoritative, so the file is what is wrong.\n"
                "       Fix: python3 prosper/tools/docs/gen_progress_tracker.py\n"
                "       then commit the result." % out.name,
                file=sys.stderr,
            )
            return 1
        print("%s is up to date with the trackers." % out.name)
        return 0

    out.write_text(text)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
