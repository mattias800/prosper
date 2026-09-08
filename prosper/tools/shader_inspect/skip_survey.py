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
counts DISTINCT SHADERS, and it is NOT a draw count -- a census that divided one by the other
reported "1.5% of draws" and had to be withdrawn (render_runner.h:6940-6944). Every number here is
shaders.

WHAT A ZERO MEANS, and this is the thing most likely to be misread. **A zero is never a clean bill
of health.** A title refuses nothing until it renders the thing that would have been refused, so a
zero means only "none in the surveyed window".

Measured on this tool's own first run: **GTA V rendered 5160 frames and reported 0 refused**, because
150 s reaches its menus and not its world. Its world refuses 21 (#3464). A frame count is therefore
NOT a proxy for "did this title render its content" -- the first version of this guard used exactly
that reasoning and would have published "GTA V: clean".

So: a zero is reported as `none in window`, never `clean`; the window and frame count print beside
every row; and a title that needs a route to reach its real rendering is named in ROUTE_GATED below
so its zero carries the warning inline.

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

# Titles whose real rendering is behind a menu route, so a default boot surveys menus and
# its zero says nothing about the content. GTA V is the measured case: 5160 frames, 0
# refused, against 21 in its world. This list is a courtesy, not a safety net -- an
# unlisted title can be just as route-gated, which is why no zero is called clean.
ROUTE_GATED = {
    "PPSA04263": "world is behind the Story/Performance menu route; a default boot surveys menus",
}

# "[render] skip draw=N fs=HASH: fragment shader requires subgroup size 64 (... why=0x2 wave-any)"
SKIP = re.compile(r"\[render\] skip draw=\d+ fs=([0-9a-f]+): fragment shader requires subgroup "
                  r"size (\d+) .*?why=(\S+)([^)]*)\)")
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
    with open(log, "wb") as fh:
        proc = subprocess.Popen([app, "--volume", "0", dump],
                                stdout=subprocess.DEVNULL, stderr=fh, env=env)
        deadline = time.time() + seconds
        while time.time() < deadline and proc.poll() is None:
            time.sleep(2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    text = log.read_text(encoding="utf-8", errors="replace")

    shaders = {}
    for fs_hash, size, mask, names in SKIP.findall(text):
        shaders[fs_hash] = (int(size), mask, names.strip())
    frames = max([int(m) for m in FPS.findall(text)] or [0])
    return {
        "title": title,
        "frames": frames,
        "refused_shaders": len(shaders),
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

    print("\n%-12s %8s %9s  %s" % ("title", "frames", "refused", "reasons (distinct shaders)"))
    print("-" * 78)
    for r in results:
        if r["frames"] == 0:
            verdict = "UNKNOWN -- rendered no frames, so it refused nothing by construction"
        elif r["refused_shaders"] == 0:
            # Never "clean": see the module docstring. A zero is scoped to the window, and for a
            # route-gated title the window is known not to contain the content.
            verdict = "none in window"
            if r["title"] in ROUTE_GATED:
                verdict += " -- BUT %s" % ROUTE_GATED[r["title"]]
        else:
            verdict = ", ".join("%s x%d" % (k, v) for k, v in r["reasons"].most_common())
        print("%-12s %8d %9s  %s"
              % (r["title"], r["frames"],
                 r["refused_shaders"] if r["frames"] else "-", verdict))

    affected = [r for r in results if r["frames"] and r["refused_shaders"]]
    unknown = [r for r in results if not r["frames"]]
    print("\n%d of %d titles refused at least one fragment shader in a %ds window; %d rendered"
          % (len(affected), len(results), args.seconds, len(unknown)))
    print("nothing at all. A zero is NOT a clean bill of health -- a title refuses nothing until it")
    print("renders the thing that would have been refused. GTA V rendered 5160 frames and reported")
    print("0 here while its world refuses 21 (#3464), because this window reaches only its menus.")
    print("Counts are DISTINCT SHADERS, never draws -- the renderer's message is inside a")
    print("shader-identity dedupe guard and the draw drop is outside it.")
    if unknown:
        print("A title that rendered no frames refuses nothing by construction; those rows are")
        print("UNKNOWN, not clean. Re-run them for longer or with a route.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: (dict(v) if isinstance(v, collections.Counter) else v)
              for k, v in r.items()} for r in results], indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
