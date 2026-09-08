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
    return {
        "title": title,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--dumps", required=True, help="directory holding <TITLE>-app0 dumps")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", default=".", help="where per-title logs are written")
    ap.add_argument("--only", help="comma-separated title ids to survey")
    ap.add_argument("--json", help="also write the results here")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
