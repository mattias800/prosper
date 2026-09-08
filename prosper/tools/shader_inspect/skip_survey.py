#!/usr/bin/env python3
"""Boot each title briefly and count the fragment shaders the renderer refuses.

W7 on #3464. The wave64 loss was found twice by eye and once by a census; nobody knows how many
titles it affects, and every priority on that issue is a guess without the list. This is the cheapest
possible instrument for it: the renderer ALREADY prints one line per distinct refused shader, with
the reason mask, so a boot and a grep answers for a whole title in about two minutes. No capture, no
GPU replay, no new emulator code.

    python skip_survey.py --app <prosper-app> --dumps <DUMP_ROOT> [--seconds 120]

WHAT THIS COUNTS, and the trap it is built around. The renderer's skip message sits INSIDE a dedupe
guard keyed on shader identity, while the `continue` that drops the draw sits OUTSIDE it. So this
counts SHADERS, not draws -- a census that divided one by the other reported "1.5% of draws" and
had to be withdrawn (render_runner.h:6940-6944).

They are UPPER BOUNDS on distinct shaders, not exact counts, and are in different units from
`capture_wave_census.py`. The `fs=` field is the renderer's compile-instance key, not a content
hash, so one program compiled twice counts twice here while the capture census keys on content.
The two tools' numbers therefore must not be differenced -- which is also true for a larger
reason already recorded there: this samples a boot, that samples one frame.

WHAT A ZERO MEANS, and this is the thing most likely to be misread. **A zero is never a clean bill
of health**, for TWO independent reasons, and an early version of this file published only the first.

**(1) Coverage.** A title refuses nothing until it renders the thing that would have been refused, so
a zero means only "none in the surveyed window". Measured on this tool's first run: GTA V rendered
5160 frames and reported 0 refused, because 150 s reaches its menus and not its world.

**(2) Admission.** On a title covered by the native-wave32 allowlist, a fragment shader whose reason
set is exactly `wave-any` is ADMITTED rather than skipped -- `render_runner.h:7140-7155` sets
`fragment_subgroup_skip = false` and logs `[render] GTA V native-width fragment vote:` INSTEAD of the
skip line. So on `PPSA04263` that class never appears in the REFUSED column at any route or window,
and wave-any is exactly the class every hit in the corpus belongs to. It is not invisible: it is
counted separately, as `admitted`. An earlier version of this file said "invisible at any route",
which was true before the admit line was counted and false afterwards -- printed, at that point,
immediately beside the column counting it.

(2) is the dangerous one, because (1) alone invites the obvious repair -- "run the Story/Performance
route and look again" -- which cannot surface a single wave-any shader on that title. So this tool
also counts the ADMIT line, reported as `admitted` beside `refused`. That number is what the
allowlist is buying, i.e. what #3464's W5 would extend to other titles.

A zero is therefore printed as `none in window`, never `clean`; the window, frame count and admitted
count print beside every row; and a title on the allowlist says so inline.

Reason masks come from the emitter and are printed by the renderer, not decoded here: a second
implementation of the bit names is exactly the drift this project has been bitten by.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Titles whose real rendering is behind a menu route, so a default boot surveys menus and a zero says
# nothing about the content. A courtesy, not a safety net: an unlisted title can be just as
# route-gated, which is why no zero is ever called clean.
ROUTE_GATED = {
    "PPSA04263": "world is behind the Story/Performance menu route; a default boot surveys menus",
}

# Titles on the renderer's native-wave32 allowlist (live_renderer.cpp:1216 -> render_runner.h:7140).
# On these, a fragment shader whose reason set is exactly wave-any is ADMITTED and logs the admit
# line rather than the skip line, so it never appears in the REFUSED column here -- not with a
# longer window and not with a route. It is counted, in the ADMITTED column. Kept in step with
# capture_wave_census.py's NATIVE_VOTE_ALLOWLIST and pinned by test_skip_survey.py.
NATIVE_VOTE_ALLOWLIST = ("PPSA04263",)

# Both patterns are pinned to the emitter's format strings by test_skip_survey.py, which greps the
# producing source. A regex that silently stops matching would make every title report zero -- the
# worst failure this tool has, and one that looks exactly like good news (instrument trap 272).
#
# "[render] skip draw=N fs=HASH: fragment shader requires subgroup size 64 (... why=0x2 wave-any)"
SKIP = re.compile(r"\[render\] skip draw=\d+ fs=([0-9a-f]+): fragment shader requires subgroup "
                  r"size (\d+) .*?why=(\S+)([^)]*)\)")
# "[render] GTA V native-width fragment vote: subgroup 64 -> 32 (why=0x2)" -- the ADMIT line. It is
# emitted INSTEAD of a skip line, so without counting it an allowlisted title reports a zero that no
# route or window can correct.
ADMIT = re.compile(r"\[render\] GTA V native-width fragment vote: subgroup (\d+) -> (\d+) "
                   r"\(why=(\S+)\)")
FPS = re.compile(r"\[app\] [\d.]+ fps \((\d+) frames")
# The counts are host-dependent in the extreme: on an AMD host wave64 is native, nothing is refused,
# and every count is legitimately 0. A baseline recorded on one GPU and compared against another
# would report every shader as fixed and go green, so the device is recorded and checked.
DEVICE = re.compile(r"\[render\] Vulkan device: (.+)")


def survey(app, dump, seconds, out_dir):
    title = Path(dump).name.replace("-app0", "")
    log = Path(out_dir) / ("skip_%s.log" % title)
    env_line = {
        "PROSPER_RENDER": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
    }
    import os
    env = dict(os.environ)
    env.update(env_line)
    started = time.time()
    with open(log, "wb") as fh:
        proc = subprocess.Popen([app, "--volume", "0", dump],
                                stdout=subprocess.DEVNULL, stderr=fh, env=env)
        deadline = started + seconds
        while time.time() < deadline and proc.poll() is None:
            time.sleep(2)
        exited_early = proc.poll() is not None
        if not exited_early:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                # kill() only sends the signal; without this the file may still be open when the
                # log is read below. Guarded because on Windows kill() is
                # TerminateProcess and this second wait can still time out -- letting it
                # raise would abort the whole corpus run over one stuck title, losing
                # every result already gathered.
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
    elapsed = time.time() - started
    text = log.read_text(encoding="utf-8", errors="replace")

    shaders = {}
    for fs_hash, size, mask, names in SKIP.findall(text):
        shaders[fs_hash] = (int(size), mask, names.strip())
    admitted = ADMIT.findall(text)
    # NOT a "did it draw" test. The fps line prints once per 60 presented frames (main.cpp), so this
    # is 0 for any run that presented fewer than 60 -- which is emphatically not the same as never
    # drawing, and skip lines are emitted from the submit path independently of presentation. It is
    # reported for context and never used to suppress a count.
    frames = max([int(m) for m in FPS.findall(text)] or [0])
    device = DEVICE.search(text)
    return {
        "title": title,
        "device": device.group(1).strip() if device else "",
        "frames": frames,
        "elapsed": round(elapsed, 1),
        "exited_early": exited_early,
        "returncode": proc.returncode,
        "refused_shaders": len(shaders),
        "admitted_shaders": len(admitted),
        "reasons": collections.Counter("%s %s" % (m, n) for _, m, n in shaders.values()),
        "widths": sorted({s for s, _, _ in shaders.values()}),
        "log": str(log),
    }


def unjudgeable(row):
    """Why this row cannot support a verdict, or "" if it can.

    ONE definition, used by both the record and compare sides. It was written twice and drifted
    within a single commit: the record side omitted the unrecorded-device case, so a run mixing a
    device-less row with a good one recorded that row as 0 and then refused to compare against the
    file it had just written. A predicate that decides what counts as evidence belongs in one place.

    ADMISSION RULE, because this body answers two different questions and they only happen to agree.
    The record side asks "can this row DEFINE truth?"; the compare side asks "can it be MEASURED
    AGAINST truth?". All three conditions below make a row unusable for BOTH, which is what makes one
    predicate correct today. A condition that disqualifies a row for only one of them does NOT belong
    here -- it belongs at that caller, or the two sides silently inherit each other's standards.

    Two contract facts both callers rely on: the returned reason is never falsy when there is one
    (they test `if why`), and it reads as a fragment following a title id (they format `"%s %s"`).
    """
    if row["exited_early"]:
        return "exited early after %.0fs (rc=%s)" % (row["elapsed"], row["returncode"])
    if not row["frames"]:
        return "presented no frames"
    if not row["device"]:
        return "the renderer announced no Vulkan device, so its counts cannot be attributed to one"
    return ""


def baseline_from(rows):
    """The baseline a run would record, or (None, why) if the run cannot support one.

    Applies the same cannot-be-judged rule the comparison does. Without it a title that presented no
    frames was recorded as `0`, quietly baking "this title refuses nothing" into the file every later
    run is measured against -- which would then redden honestly on the next good run and be read as a
    regression. The failure direction is benign; the confusion is not.
    """
    unjudged = ["%s %s" % (r["title"], why)
                for r in rows for why in [unjudgeable(r)] if why]
    if unjudged:
        return None, ("cannot record a baseline: %s. Recording 0 for a title that could not be "
                      "judged would bake a false clean state into the file every later run is "
                      "measured against." % "; ".join(sorted(unjudged)))
    devices = {r["device"] for r in rows}
    if len(devices) != 1:
        return None, ("cannot record a baseline: this run reports %d distinct Vulkan devices, so "
                      "nothing later compared against it could be trusted." % len(devices))
    return {
        "device": sorted(devices)[0],
        "titles": {r["title"]: r["refused_shaders"] for r in rows},
    }, ""


def write_baseline(rows, path):
    """Validate FIRST, then write. Returns (data, why).

    The previous version wrote the file and then reported that it had refused, so a bad run
    clobbered the committed baseline while printing "REFUSING to write".
    """
    data, why = baseline_from(rows)
    if data is None:
        return None, why
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data, ""


def compare_to_baseline(rows, baseline):
    """(ok, message). A guard that cannot compare returns False.

    That stance is the opposite of the survey's own reporting, and deliberately. A survey may say "I
    could not tell"; a GUARD that cannot tell has not established the thing it exists to establish,
    so passing would let a title that stopped booting read as a title with no dropped shaders. Every
    non-comparable case below therefore FAILS:

      - the run presented no frames, or the process exited before its window closed
      - a title in the baseline was not surveyed at all
      - the GPU differs from the one the baseline was recorded on, or is unknown

    Only two outcomes pass: every title matched its recorded count, or some went DOWN. A decrease
    passes and says so, because the repair is to re-record the baseline rather than to loosen it.
    """
    expected = baseline.get("titles", {})
    want_device = baseline.get("device", "")

    # A baseline with no titles compared clean against every possible run, reporting
    # "unchanged: 0 title(s)". Vacuous truth is the purest form of this guard's only real failure --
    # passing while establishing nothing -- so an empty baseline is unusable, not satisfied.
    if not expected:
        return False, ("the baseline lists no titles, so it cannot establish anything -- every run "
                       "would match it. Re-record it with --write-baseline.")

    seen_devices = {r["device"] for r in rows}
    if not want_device:
        return False, ("the baseline records no device, so nothing can be compared against it -- "
                       "re-record it with --write-baseline")
    if seen_devices != {want_device}:
        got = ", ".join(sorted(d or "(unrecorded)" for d in seen_devices))
        return False, ("REFUSING to compare: baseline recorded on %r, this run reports %s. These "
                       "counts are host-dependent -- on a host with native wave64 nothing is "
                       "refused and every count is legitimately 0, which would read as every "
                       "shader being fixed." % (want_device, got))

    by_title = {r["title"]: r for r in rows}
    missing = sorted(set(expected) - set(by_title))
    if missing:
        return False, ("could not be judged: %d baseline title(s) were not surveyed (%s). A title "
                       "that did not run is not a title with no dropped shaders."
                       % (len(missing), ", ".join(missing)))

    uncomparable = ["%s %s" % (title, why)
                    for title in sorted(expected)
                    for why in [unjudgeable(by_title[title])] if why]
    if uncomparable:
        return False, ("could not be judged: %s. A run that never got going has established "
                       "nothing, so this is a FAILURE and not an improvement."
                       % "; ".join(uncomparable))

    worse, better = [], []
    for title in sorted(expected):
        got, want = by_title[title]["refused_shaders"], expected[title]
        if got > want:
            worse.append("%s refuses %d, baseline %d" % (title, got, want))
        elif got < want:
            better.append("%s refuses %d, baseline %d" % (title, got, want))

    # A surveyed title the baseline has never heard of is not a regression -- there is nothing to
    # regress from -- but dropping it silently let a title refusing 99 shaders read as
    # "unchanged: 3 title(s)". It is reported on every verdict, with its count, so the summary can
    # never be more reassuring than the run.
    unknown = sorted(set(by_title) - set(expected))
    extra = ""
    if unknown:
        extra = (" NOT IN BASELINE (no verdict possible, add with --write-baseline): %s."
                 % "; ".join("%s refuses %d" % (u, by_title[u]["refused_shaders"])
                             for u in unknown))

    if worse:
        return False, ("REGRESSION: %s.%s%s" % ("; ".join(worse),
                       " (also improved: %s)" % "; ".join(better) if better else "", extra))
    if better:
        return True, ("improved: %s -- re-record with --write-baseline so the gain is locked in.%s"
                      % ("; ".join(better), extra))
    return True, ("unchanged: %d title(s) match the baseline exactly.%s" % (len(expected), extra))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--dumps", required=True, help="directory holding <TITLE>-app0 dumps")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", default=".", help="where per-title logs are written")
    ap.add_argument("--only", help="comma-separated title ids to survey")
    ap.add_argument("--json", help="also write the results here")
    ap.add_argument("--baseline",
                    help="compare against this baseline; exit 1 on a regression, on a run "
                         "that cannot be judged, or on a different GPU")
    ap.add_argument("--write-baseline", help="record this run as the baseline")
    args = ap.parse_args()

    app = str(Path(args.app).resolve())
    root = Path(args.dumps).resolve()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dumps = sorted(d for d in root.glob("*-app0") if d.is_dir())
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        dumps = [d for d in dumps if d.name.replace("-app0", "") in want]
    if not dumps:
        print("no dumps found under %s" % root, file=sys.stderr)
        return 2

    results = []
    for dump in dumps:
        print("... %s (%ds)" % (dump.name, args.seconds), flush=True)
        results.append(survey(app, str(dump), args.seconds, args.out))

    print("\n%-12s %7s %6s %8s %8s  %s"
          % ("title", "sec", "frames", "refused", "admitted", "reasons (distinct shaders)"))
    print("-" * 96)
    for r in results:
        notes = []
        if r["refused_shaders"]:
            notes.append(", ".join("%s x%d" % (k, v) for k, v in r["reasons"].most_common()))
        else:
            # Never "clean". A zero is scoped to the window AND, on an allowlisted title, to a class
            # this tool structurally cannot see.
            notes.append("none in window")
        if r["title"] in NATIVE_VOTE_ALLOWLIST:
            notes.append("ALLOWLISTED: wave-any is ADMITTED here, not skipped -- it is in the "
                         "admitted column, never the refused one, at any route")
        if r["title"] in ROUTE_GATED:
            notes.append(ROUTE_GATED[r["title"]])
        if r["exited_early"]:
            notes.append("EXITED EARLY after %.0fs (rc=%s) -- window not completed"
                         % (r["elapsed"], r["returncode"]))
        print("%-12s %7.0f %6d %8d %8d  %s"
              % (r["title"], r["elapsed"], r["frames"], r["refused_shaders"],
                 r["admitted_shaders"], "; ".join(notes)))

    affected = [r for r in results if r["refused_shaders"]]
    unknown = [r for r in results if not r["frames"]]
    print("\n%d of %d titles refused at least one fragment shader in a %ds window; %d rendered"
          % (len(affected), len(results), args.seconds, len(unknown)))
    print("fewer than 60 frames, which is NOT the same as never drawing.")
    print("")
    print("A zero is NOT a clean bill of health, for two reasons. (1) A title refuses nothing until")
    print("it renders the thing that would have been refused -- GTA V reported 0 over 5160 frames")
    print("while its world refuses 21 (#3464), because the window reached only its menus. (2) On an")
    print("ALLOWLISTED title a wave-any shader is admitted and logs the admit line instead of a skip")
    print("line, so it is counted in the ADMITTED column and never the refused one, at any route")
    print("and any window -- and wave-any is the class every hit in this corpus belongs to.")
    print("Counts are SHADERS, never draws -- the renderer's message is inside a "
          "shader-identity")
    print("dedupe guard and the draw drop is outside it. They are UPPER BOUNDS on distinct "
          "shaders:")
    print("fs= is a compile-instance key, not a content hash, so one program compiled twice "
          "counts")
    print("twice -- and they are in different units from capture_wave_census.py, which keys on content.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: (dict(v) if isinstance(v, collections.Counter) else v)
              for k, v in r.items()} for r in results], indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)
    if args.write_baseline:
        data, why = write_baseline(results, args.write_baseline)
        if data is None:
            print("REFUSING to write a baseline: %s" % why, file=sys.stderr)
            return 2
        print("wrote baseline %s (device %s, %d titles)"
              % (args.write_baseline, data["device"], len(data["titles"])))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        ok, why = compare_to_baseline(results, baseline)
        print("BASELINE %s: %s" % ("OK" if ok else "FAILED", why))
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
