#!/usr/bin/env python3
"""Human-authored render snapshots — import, check, and accept.

The scripted guards in `snapshot.py` answer "does this title still render?" by replaying a route and
comparing a window of frames against a reviewed baseline. The anchor is the weak part: a window
measured in seconds stops lining up the moment a title's pacing changes, and the guard then fails
while the picture is perfect. `blue-prince-hall` fails that way on master right now.

Here the anchor is a person. You play the game with `prosper-app`, and at any moment press:

    F6   this frame looks CORRECT     -> a positive snap
    F7   this frame looks WRONG       -> a negative snap

Both verdicts are kept, and the negative half is the one that does not exist today. A broken title
simply has no guard, so nothing notices when it starts rendering. A recorded known-bad frame turns
"broken" into tracked state: when it stops matching, something changed and a human should look at
whether it improved. That is information, never a failure.

    positive snap stops matching  ->  FAILURE   (a regression)
    negative snap stops matching  ->  INFO      (go look; it may have been fixed)

Every snap anchors on the PAD FLIP ORDINAL — display flips since the guest's first pad poll — which
is the axis `PROSPER_PAD_RECORD` writes routes against and `PROSPER_PAD_SCRIPT` replays. The route
recorded during authoring and the snaps taken during it therefore index each other, so a later
automated run returns to the exact moment judgement was passed. Unlike wall-clock it is
boot-speed-invariant (#302).

WHAT LIVES WHERE, and why it is split:

  prosper/tools/snapshot/snaps/<name>.json     committed. Signatures, verdicts, anchors, route path.
  prosper/scripts/<name>/route.pad             committed. The recorded input route (plain text).
  ~/.local/share/prosper/snap-refs/<name>/     LOCAL ONLY. The reference images.

Game imagery is never committed — that is a hard project rule, and it is also why the store holds a
144-byte luminance signature instead of a PNG. The consequence is worth stating plainly: on a fresh
clone the numbers still work, but "show me what it used to look like" only works for whoever
authored the run (or anyone who re-authors it).

USAGE
    snaps.py author --name <name> --dump <TITLE-app0>              play, and press F6/F7
    snaps.py import <capture-dir> --name <name> [--route <file>]   adopt an authoring session
    snaps.py check [name ...]                                      replay and compare; exit 1 on failure
    snaps.py accept <name> <index> [<index> ...]                   promote an actual to be the new snap
    snaps.py list
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROSPER_ROOT = os.path.dirname(os.path.dirname(HERE))
REPO_ROOT = os.path.dirname(PROSPER_ROOT)
SNAP_STORE = os.path.join(HERE, "snaps")
GAME_ROOT = os.environ.get("PROSPER_GAME_ROOT", REPO_ROOT)
APP = os.environ.get("PROSPER_APP_BIN",
                     os.path.join(PROSPER_ROOT, "build-linux", "prosper-app"))
REF_HOME = os.environ.get(
    "PROSPER_SNAP_REF_DIR",
    os.path.join(os.path.expanduser("~"), ".local", "share", "prosper", "snap-refs"))

# A positive snap must stay this similar to keep passing. Same bar the existing gameplay guards use,
# chosen there to survive subtle pixel improvements while catching a collapse.
DEFAULT_MIN_SSIM = 0.85

# How far either side of an authored anchor the check looks, in flips, and how many samples it takes
# across that span.
#
# This is not belt-and-braces, it is required. A flip anchor is stable against RENDERING changes but
# not against changes in how fast the guest gets there: Blue Prince's boot is asset-loading bound, so
# under machine load the title screen arrives hundreds of flips later than on an idle run. Measured
# during development -- an anchor that landed on the title screen on a quiet machine landed on a
# still-black loading frame while builds and CI were running, and reported a confident FAIL with
# "colors 14446 -> 1". Comparing only the exact flip makes the suite fail for reasons that have
# nothing to do with the renderer, which is the failure this whole design exists to remove.
DEFAULT_FLIP_WINDOW = 900
DEFAULT_WINDOW_SAMPLES = 7

# A SCAN snap searches a wide span forward of its anchor instead of a tight window.
#
# This is what makes the whole scheme robust without touching the guest's clock: drift stops being
# something to eliminate and becomes something to search past. It covers slow test runners, variable
# loading, FMVs and frame-rate differences with one mechanism.
#
# Scanning is FORWARD-BIASED and bounded rather than unbounded. What the bound buys is finite search
# cost and a finite run: `best_match` takes the ARGMAX over every sampled offset, not the first match,
# so bounding it does not protect against picking a wrong occurrence -- that framing was wrong and is
# corrected here rather than repeated.
#
# The real hazard is a scene that recurs INSIDE the span, which the bound does not address: at 60
# flips/s the default forward span is about 200 s of play, comfortably long enough to contain a menu
# you return to. A scan snap of a screen the title revisits can therefore match the wrong visit and
# score well doing it. Prefer an anchor snap for anything that recurs; keep scans for frames that sit
# after something of variable length and appear once.
DEFAULT_SCAN_FORWARD = 12000
DEFAULT_SCAN_BACK = 900
DEFAULT_SCAN_SAMPLES = 33


# ---------------------------------------------------------------------------------------------
# Image reading and the signature
# ---------------------------------------------------------------------------------------------

def read_bmp_rgb(path):
    """Return (width, height, rows) with rows[y][x] = (r, g, b), top row first.

    Only the 24/32-bit uncompressed BGR layout `write_frame_bmp` emits is supported; anything else
    raises rather than guessing, because a silently misparsed image would produce a confident and
    meaningless similarity score.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError(f"{path}: not a BMP")
    offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    raw_height = struct.unpack_from("<i", data, 22)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]
    if bpp not in (24, 32) or compression != 0:
        raise ValueError(f"{path}: unsupported BMP (bpp={bpp} compression={compression})")
    height = abs(raw_height)
    bottom_up = raw_height > 0
    stride = ((width * (bpp // 8) + 3) // 4) * 4
    step = bpp // 8
    rows = []
    for y in range(height):
        source = (height - 1 - y) if bottom_up else y
        base = offset + source * stride
        row = []
        for x in range(width):
            i = base + x * step
            if i + 2 >= len(data):
                row.append((0, 0, 0))
            else:
                row.append((data[i + 2], data[i + 1], data[i]))
        rows.append(row)
    return width, height, rows


def signature_of(path):
    """A 16x9 luminance thumbnail plus cheap content metrics.

    The thumbnail keeps large layer placement and deliberately discards raster detail, so a correct
    fix that shifts pixels slightly does not read as a regression. `nonblack_ratio` and
    `distinct_colors` sit beside it because SSIM alone cannot tell "the scene changed" from "the
    scene vanished" -- two flat frames of different flat colours score highly against each other.
    """
    width, height, rows = read_bmp_rgb(path)
    cells = []
    for cy in range(9):
        y0 = (cy * height) // 9
        y1 = max(y0 + 1, ((cy + 1) * height) // 9)
        for cx in range(16):
            x0 = (cx * width) // 16
            x1 = max(x0 + 1, ((cx + 1) * width) // 16)
            total = 0
            count = 0
            # Sample rather than sum every pixel: at 1920x1080 the full pass is ~2M tuples per
            # frame in pure Python and the check run does this for every snap of every title.
            for y in range(y0, y1, max(1, (y1 - y0) // 8)):
                for x in range(x0, x1, max(1, (x1 - x0) // 8)):
                    r, g, b = rows[y][x]
                    total += (r * 299 + g * 587 + b * 114) // 1000
                    count += 1
            cells.append(total // count if count else 0)

    nonblack = 0
    sampled = 0
    colors = set()
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            r, g, b = rows[y][x]
            sampled += 1
            if r > 16 or g > 16 or b > 16:
                nonblack += 1
            colors.add((r, g, b))
    return {
        "luma16x9": bytes(cells).hex(),
        "nonblack_ratio": round(nonblack / sampled, 6) if sampled else 0.0,
        "distinct_colors": len(colors),
        "width": width,
        "height": height,
    }


def decode_luma(value):
    if len(value) != 288:
        raise ValueError("luma16x9 must be 288 hexadecimal characters")
    return tuple(bytes.fromhex(value))


def structural_similarity(left, right):
    """SSIM over two equally sized luminance signatures.

    Copied in shape from snapshot.py's implementation so both systems score the same way while they
    coexist; the constants are the standard ones.
    """
    if len(left) != len(right) or not left:
        raise ValueError("SSIM inputs must have the same non-zero length")
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((v - mean_left) ** 2 for v in left) / count
    var_right = sum((v - mean_right) ** 2 for v in right) / count
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / count
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = ((mean_left ** 2 + mean_right ** 2 + c1) * (var_left + var_right + c2))
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0, ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / denominator))


# ---------------------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------------------

def store_path(name):
    return os.path.join(SNAP_STORE, f"{name}.json")


def load_entry(name):
    path = store_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"snaps: no snap set named '{name}' ({path} does not exist)")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_entry(name, entry):
    os.makedirs(SNAP_STORE, exist_ok=True)
    with open(store_path(name), "w", encoding="utf-8") as handle:
        json.dump(entry, handle, indent=2, sort_keys=True)
        handle.write("\n")


def ref_dir(name):
    return os.path.join(REF_HOME, name)


def list_names():
    if not os.path.isdir(SNAP_STORE):
        return []
    return sorted(f[:-5] for f in os.listdir(SNAP_STORE) if f.endswith(".json"))


# The guest clock must advance per FLIP, not per second, for both halves.
#
# Without this the anchors are frame-rate dependent and cannot correspond. Measured on the first real
# authoring session: authoring windowed ran at 60.4 fps, the headless check at 77.1 fps, and a
# time-based intro logo therefore burned ~28% more flips in the check. The observed drift matched the
# ratio almost exactly -- +600 flips at authored anchor 2516 against a predicted +694, and past the
# window entirely by 4336. The picture was identical; only the rate differed.
#
# PROSPER_DET_CLOCK makes the guest see exactly 1/DET_FPS seconds per flip, so a three-second logo
# always costs the same number of flips however fast the host renders. That turns "the same flip" into
# "the same guest time", which is the property flip anchoring assumes and otherwise does not have.
# The deterministic clock is now OFF by default. The FLIP PACER is not -- it runs unconditionally on
# both sides, and it is what replaced the clock. Pacing buys the same "same flip = same guest time"
# correspondence honestly: the guest keeps a real clock and simply flips at a fixed rate, the way a
# vsync-locked console does, so nothing has to lie to it about what time it is.
#
# PRECONDITION, and it is a real one: pacing can only ever SLOW a fast host down. flip_pace_wait()
# sleeps when it is ahead of schedule and re-anchors when it is behind, so on any title/host that
# cannot SUSTAIN det_fps the pacer is inert and the drift this section describes comes straight back.
# Vsync has the same shape -- it caps a maximum, it does not guarantee a rate. Blue Prince measured
# 180 fps windowed at its menu, 20 in gameplay and 4.8 during its FMV; the last two are below any
# sane det_fps, so its post-FMV anchors need SCAN mode rather than pacing to be found.
#
# Both existed to stop anchors drifting when the host renders at a different rate than it did during
# authoring. They bought that at a real cost: PROSPER_DET_CLOCK replaces EVERY time source the guest
# has -- sceKernelReadTsc, GetProcessTime, and the wall-clock anchor behind CLOCK_REALTIME,
# gettimeofday, time() and sceRtc* -- with anchor + flips*(1/DET_FPS). A guard recorded under that
# clock tests a machine nobody plays on, and GRIS proves the divergence is not theoretical: its
# opening FMV freezes with the clock on (1,680 frames and 47 forced-submit stalls against 42,000 and
# 0 without it).
#
# Drift is now handled where it belongs -- in the MATCHER, by scanning for the frame rather than
# demanding it appear at a fixed offset. That is robust to slow test runners, loading times, FMVs and
# frame-rate differences at once, and it needs no lie about time. Normal play is untouched: real
# clock, uncapped, 120 Hz fine.
#
# Both knobs remain available per title for a set that genuinely needs them.
DEFAULT_DET_FPS = 60


def apply_deterministic_clock(env, det_fps, enabled=False):
    """Pin the guest clock, unless this title is one the clock breaks.

    IT IS NOT UNIVERSALLY SAFE, and that has to be a per-title decision rather than a default nobody
    can override. GRIS is the measured counterexample: with a New Game route reaching its opening
    FMV, 150 s of run produced 42,000 frames with the clock OFF and 1,680 with it ON, plus 47
    "Forcing submitDone to avoid TRC R4089 breach" messages -- a 25x collapse and a frozen white
    screen where the movie should play. Booting to its title screen is unaffected; the FMV is the
    trigger.

    The mechanism is the obvious one in hindsight: video playback is A/V-sync sensitive, and a clock
    that advances per FLIP rather than per second makes the decoder wait for a presentation time that
    never arrives at the rate it expects.

    With the clock OFF the anchors become frame-rate dependent again, which is what the search window
    exists to absorb. A title that needs the clock off may therefore need a wider --flip-window.
    """
    if not enabled:
        # Explicitly clear, so an authoring shell that exported these does not leak them into a run
        # whose entry says the clock is off -- the check would then disagree with the authoring.
        env.pop("PROSPER_DET_CLOCK", None)
        env.pop("PROSPER_DET_FPS", None)
        return
    env["PROSPER_DET_CLOCK"] = "1"
    env["PROSPER_DET_FPS"] = str(det_fps)


# The keys `author` records and `import` reads back. Named once so a rename cannot desynchronise the
# producer from the consumer -- they were two hand-written literals, and renaming one left the whole
# suite green.
SESSION_KEYS = ("det_fps", "det_clock", "savedata")


def default_check_out_dir(name):
    """Where a check writes its captured frames.

    Deliberately NOT tempfile.gettempdir(). On the Linux box /tmp is a RAM-backed tmpfs with a
    per-user quota shared by every concurrent agent; exhausting it takes the machine's RAM with it
    rather than merely failing the write. A check writes one BMP per candidate flip, and scan mode
    raises that from 7 per snap to 34.
    """
    return os.path.join(os.path.expanduser("~"), ".cache", "prosper-snaps", name)


def reconcile_session(args, existing, was_passed=None):
    """Settle an appending run against what the directory was already authored under.

    Returns (record, adopted, conflicts):
      record    -- what to store in session.json
      adopted   -- keys taken FROM the stored session because the caller did not ask for them
      conflicts -- keys the caller explicitly asked for that contradict the stored session

    This must run BEFORE the run environment is built. An earlier version resolved it afterwards,
    so `--append --det-fps 60` on a directory authored at 30 ran the emulator paced at 60 while
    recording 30 -- the run and its own record disagreeing, which is the exact divergence this
    whole mechanism exists to prevent. It moved the damage from run 1 to run 2 rather than removing
    it.

    A directory describes ONE set of conditions, because its snaps share one route and one flip
    axis. So an explicit contradiction is refused rather than silently resolved; a mere default is
    adopted, since that is somebody continuing a session without retyping the flags.
    """
    was_passed = _flag_was_passed if was_passed is None else was_passed
    if not (getattr(args, "append", False) and existing):
        return session_record(args), [], []
    adopted, conflicts = [], []
    for key in SESSION_KEYS:
        if key not in existing or existing[key] == getattr(args, key):
            continue
        (conflicts if was_passed(key) else adopted).append(key)
    for key in adopted:
        setattr(args, key, existing[key])
    # Rewrite the record even when nothing differed: a session file written by an older version can
    # be missing a key, and this is the only chance to complete it.
    return session_record(args), adopted, conflicts


def session_record(args):
    """What an authoring session stores about itself, so `import` need not be told again."""
    record = {key: getattr(args, key) for key in SESSION_KEYS}
    record["name"] = args.name
    return record


def author_env(args, out_dir, base=None):
    """The environment an AUTHORING session runs under.

    Extracted for the same reason as replay_env: the pacing line here and the one there must agree,
    and while both were inlined nothing could assert that. Deleting either used to leave the suite
    fully green while every future session drifted against every future check.
    """
    env = dict(os.environ if base is None else base)
    env.update({
        "PROSPER_RENDER": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_SNAP_DIR": out_dir,
        "PROSPER_PAD_RECORD": os.path.join(out_dir, "route.pad"),
    })
    # Both halves read their rate through pace_fps(), off a dict shaped like a stored entry, so the
    # author and the check cannot disagree about what "60" means.
    env["PROSPER_FLIP_PACE_FPS"] = str(pace_fps({"det_fps": args.det_fps}))
    apply_deterministic_clock(env, args.det_fps, args.det_clock == "on")
    if args.savedata == "fresh":
        apply_fresh_savedata(env, out_dir)
    # Deliberately NOT offscreen: authoring is a person looking at a window.
    env.pop("SDL_VIDEODRIVER", None)
    return env


def clock_enabled(entry):
    """Does this stored set replay with the guest clock pinned?

    The fallback is "on" while the author/import CLI default is "off", and that asymmetry is the
    point: this can only ever apply to a set stored before the key existed, and there was no way to
    author one of those with the clock off -- cmd_author applied it unconditionally until 4bf1c80c.
    """
    return entry.get("det_clock", "on") == "on"


def pace_fps(entry):
    """The flip rate BOTH halves must agree on. Written by cmd_import for every new set."""
    return entry.get("det_fps", DEFAULT_DET_FPS)


def replay_env(entry, out_dir, base=None):
    """The environment a check run replays under. Extracted so it can be asserted directly."""
    env = dict(os.environ if base is None else base)
    # Pace the CHECK's flips to the rate the session was authored at. This is what lets the clock
    # stay off: routes are FLIP-anchored, so flip N is a fixed moment in the game only if flips
    # happen at a fixed RATE.
    env["PROSPER_FLIP_PACE_FPS"] = str(pace_fps(entry))
    apply_deterministic_clock(env, pace_fps(entry), clock_enabled(entry))
    if entry.get("savedata", "fresh") == "fresh":
        apply_fresh_savedata(env, out_dir)
    env.update({
        "SDL_VIDEODRIVER": "offscreen",
        "PROSPER_RENDER": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_SNAP_DIR": out_dir,
        "PROSPER_SNAP_AT_FLIPS": ",".join(str(f) for f in sorted(candidate_flips(entry))),
        "PROSPER_PAD_SCRIPT": "@" + os.path.join(REPO_ROOT, entry["route"]),
    })
    return env


def apply_fresh_savedata(env, root):
    """Point BOTH save roots at empty per-run directories under `root`.

    prosper has two independent save roots and a fresh console state needs both:

        PROSPER_SAVEDATA_DIR -> SaveDataMemory slots (the whole save path for Unity titles)
        PROSPER_SAVE0        -> the /savedata0 file mount (Blasphemous 2 writes slot0/slot1 here)

    Redirecting only the first leaves file-mount titles reading the developer's real saves.

    This matters for BOTH halves, which is why it lives here rather than in the check. A game with a
    save offers "Continue" and puts it above "New Game" -- so the same D-pad inputs select a
    different item, and the route does not merely mismatch, it diverges. Authoring against real
    saves also silently writes into them.
    """
    save_dir = os.path.join(root, "savedata")
    save0_dir = os.path.join(root, "savedata0")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save0_dir, exist_ok=True)
    env["PROSPER_SAVEDATA_DIR"] = save_dir
    env["PROSPER_SAVE0"] = save0_dir


# ---------------------------------------------------------------------------------------------
# author
# ---------------------------------------------------------------------------------------------

def cmd_author(args):
    """Launch the game for an authoring session: real window, audio, pad, route recording on.

    This exists so authoring is one command rather than four environment variables remembered
    correctly. Getting PROSPER_PAD_RECORD wrong is not a visible mistake -- you play the whole
    session, press F6/F7 happily, and only discover at import time that there is no route and the
    snaps can never be replayed.
    """
    dump = args.dump
    if not dump:
        raise SystemExit("snaps: --dump is required (e.g. --dump PPSA25009-app0)")
    dump_path = dump if os.path.isabs(dump) else os.path.join(GAME_ROOT, dump)
    if not os.path.isdir(dump_path):
        raise SystemExit(f"snaps: dump not found: {dump_path}")
    if not os.path.exists(APP):
        raise SystemExit(f"snaps: prosper-app not found at {APP} (set PROSPER_APP_BIN)")

    out_dir = args.out or os.path.join(os.path.expanduser("~"), "snaps", args.name)
    if os.path.exists(os.path.join(out_dir, "snaps.jsonl")) and not args.append:
        raise SystemExit(
            f"snaps: {out_dir} already holds an authoring session.\n"
            f"       Snap indices restart at 0 each run, so a second session would overwrite the\n"
            f"       first session's images while appending to its manifest. Use a fresh --out, or\n"
            f"       --append if you really mean to continue into the same directory.")
    os.makedirs(out_dir, exist_ok=True)

    # Pacing, clock, savedata and the recorded session all come from the shared helpers above, so
    # this half and the check half cannot drift apart.
    # Settle the session against anything already stored here FIRST -- author_env reads args, so a
    # reconciliation after this point would run the emulator under different settings than it
    # records.
    session_path = os.path.join(out_dir, "session.json")
    existing = {}
    if os.path.exists(session_path):
        try:
            with open(session_path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, ValueError):
            existing = {}
    record, adopted, conflicts = reconcile_session(args, existing)
    if conflicts:
        raise SystemExit(
            "snaps: this directory was authored with "
            + ", ".join(f"{k}={existing[k]!r}" for k in conflicts)
            + ", and you asked for "
            + ", ".join(f"{k}={getattr(args, k)!r}" for k in conflicts)
            + ".\n       Its existing snaps are anchored on flips recorded under the stored\n"
              "       settings, so one directory can only describe one set of conditions.\n"
              "       Use a fresh --out to author at different settings.")
    if adopted:
        print(f"  NOTE: continuing under the session's own "
              f"{', '.join(f'{k}={existing[k]!r}' for k in adopted)} rather than the command-line "
              f"default. This run is paced to match the snaps already here.")

    env = author_env(args, out_dir)
    if args.det_clock == "on":
        print(f"  guest clock: PINNED at {args.det_fps} fps, flips paced to match. This replaces "
              f"every guest time source; some titles break (GRIS).")
    else:
        print(f"  guest clock: REAL, flips paced to {args.det_fps}/s (as a vsync-locked console\n"
              f"              does). The pacing is what keeps a flip-anchored route landing at the\n"
              f"              same guest time on both sides; the clock itself is not touched.")
    if args.savedata == "fresh":
        print("  savedata: FRESH (your real saves are untouched, and the check starts here too)")
    else:
        print("  savedata: PRESERVE -- this session uses your real saves, so the route will NOT\n"
              "            reproduce on a machine whose save state differs. A title that offers\n"
              "            'Continue' puts it above 'New Game', so the same inputs pick a\n"
              "            different item. Only use this deliberately.")

    # Record the parameters this session is authored under, so `import` need not be told them again.
    # --det-fps is a separate argument on both subcommands, each defaulting to 60: authoring at 30
    # and importing without the flag used to record 60, and BOTH halves would then pace at 60
    # against a session played at 30. Nothing failed loudly; the anchors just moved.
    with open(session_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    print(f"authoring '{args.name}' -> {out_dir}")
    print("  F6 = this frame looks CORRECT     F7 = this frame looks WRONG")
    print("  take negative snaps too: they are the only thing that will tell you a broken")
    print("  title got fixed.\n")
    subprocess.run([APP, dump_path], env=env, check=False)

    manifest = os.path.join(out_dir, "snaps.jsonl")
    taken = 0
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as handle:
            taken = sum(1 for line in handle if line.strip())
    print(f"\nsession ended: {taken} snap(s) in {out_dir}")
    if not taken:
        print("  nothing to import -- no F6/F7 presses were recorded")
        return 0
    print(f"  import with: python3 {os.path.relpath(__file__, os.getcwd())} import {out_dir} "
          f"--name {args.name}")
    return 0


# ---------------------------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------------------------

def _flag_was_passed(key):
    """True if --key appears in argv. argparse cannot distinguish a default from an explicit value,
    and here the difference decides whether the stored session or the command line wins.

    Abbreviations count. argparse accepts any unambiguous prefix, so `--det-f 30` sets det_fps while
    matching neither `--det-fps` nor `--det-fps=`; the session would then silently override a flag
    the person did type, which is the exact failure this helper exists to prevent.
    """
    flag = "--" + key.replace("_", "-")
    for arg in sys.argv[1:]:
        name = arg.split("=", 1)[0]
        # A prefix of the flag, at least "--x", is how argparse resolves an abbreviation. Being
        # generous here is safe: the cost of a false positive is honouring the command line, which
        # is what an explicit flag should do anyway.
        if len(name) > 2 and name.startswith("--") and flag.startswith(name):
            return True
    return False


def cmd_import(args):
    manifest = os.path.join(args.capture_dir, "snaps.jsonl")
    if not os.path.exists(manifest):
        raise SystemExit(f"snaps: {manifest} does not exist -- was PROSPER_SNAP_DIR pointed here, "
                         f"and were any snaps taken with F6/F7?")
    records = []
    with open(manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"snaps: {manifest} is empty")

    # An unanchored snap has no route position, so it can never be replayed. Rejecting it loudly
    # beats recording an anchor of zero, which would replay to the first frame of the boot and
    # compare a menu against a logo.
    unanchored = [r for r in records if r.get("pad_flip", -1) < 0]
    usable = [r for r in records if r.get("pad_flip", -1) >= 0]
    for record in unanchored:
        print(f"  REJECTED snap {record['index']} ({record['verdict']}): taken before the guest's "
              f"first pad poll, so it has no replay anchor")
    if not usable:
        raise SystemExit("snaps: every snap in this session was unanchored; nothing to import")

    # Prefer what the session was actually authored under. An explicitly passed flag still wins, so
    # a deliberate override remains possible; what this removes is the SILENT disagreement.
    session = {}
    session_path = os.path.join(args.capture_dir, "session.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, "r", encoding="utf-8") as handle:
                session = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"  note: {session_path} unreadable ({exc}); falling back to the command line")
    for key in SESSION_KEYS:
        if key in session and not _flag_was_passed(key):
            if getattr(args, key) != session[key]:
                print(f"  {key}: using {session[key]!r} from the authoring session "
                      f"(command-line default was {getattr(args, key)!r})")
            setattr(args, key, session[key])

    route_src = args.route or os.path.join(args.capture_dir, "route.pad")
    if not os.path.exists(route_src):
        raise SystemExit(f"snaps: no route at {route_src}. An authoring run must set "
                         f"PROSPER_PAD_RECORD, or the snaps cannot be replayed.")

    title_id = usable[0].get("title_id") or ""
    dump = args.dump or f"{title_id}-app0"
    route_rel = os.path.join("prosper", "scripts", args.name, "route.pad")
    route_dst = os.path.join(REPO_ROOT, route_rel)
    os.makedirs(os.path.dirname(route_dst), exist_ok=True)
    shutil.copyfile(route_src, route_dst)

    refs = ref_dir(args.name)
    os.makedirs(refs, exist_ok=True)
    snaps = []
    for record in usable:
        image = os.path.join(args.capture_dir, record["file"])
        if not os.path.exists(image):
            print(f"  SKIPPED snap {record['index']}: {record['file']} is missing")
            continue
        signature = signature_of(image)
        shutil.copyfile(image, os.path.join(refs, f"{record['index']:04d}.bmp"))
        snaps.append({
            "index": record["index"],
            "verdict": record["verdict"],
            "mode": record.get("mode", "anchor"),
            "pad_flip": record["pad_flip"],
            **signature,
        })
        print(f"  snap {record['index']:>3} {record['verdict']:<9} "
              f"{record.get('mode','anchor'):<6} flip {record['pad_flip']:>6}  "
              f"{signature['distinct_colors']:>6} colors  "
              f"nonblack {signature['nonblack_ratio']:.3f}")

    if not snaps:
        raise SystemExit("snaps: no snap images could be read; nothing imported")

    entry = {
        "name": args.name,
        "title_id": title_id,
        "dump": dump,
        "route": route_rel,
        "timeout": args.timeout,
        "min_structural_similarity": args.min_ssim,
        "savedata": args.savedata,
        "det_fps": args.det_fps,
        "det_clock": args.det_clock,
        "flip_window": args.flip_window,
        "window_samples": args.window_samples,
        "snaps": sorted(snaps, key=lambda s: s["pad_flip"]),
    }
    save_entry(args.name, entry)
    positives = sum(1 for s in snaps if s["verdict"] == "correct")
    print(f"\nimported '{args.name}': {positives} correct, {len(snaps) - positives} incorrect, "
          f"{len(unanchored)} rejected")
    print(f"  store     {store_path(args.name)}")
    print(f"  route     {route_dst}")
    print(f"  reference {refs}   (local only -- never committed)")
    return 0


# ---------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------

def window_offsets(entry, snap=None):
    """Offsets for one snap. A snap marked `scan` searches a wide forward span instead."""
    if snap is not None and snap.get("mode") == "scan":
        return scan_offsets(entry)
    return _anchor_offsets(entry)


def scan_offsets(entry):
    """A wide, forward-biased sweep: a little before the anchor, a long way after.

    Sampling is uniform here rather than geometric. The anchor-window case knows the match is
    probably near the anchor; a scan is used precisely when it is NOT, so weighting toward the anchor
    would spend the samples in the least likely place.
    """
    # Floor both at 0. Neither has a CLI flag and cmd_import never writes them, so a negative can
    # only arrive from a hand-edited store -- but the clamp below is a filter, and a negative bound
    # makes it drop every offset INCLUDING the anchor itself. The snap would then report NOT REACHED
    # with nothing to indicate the store was the problem. Verified: {forward:100, back:-200} and
    # {forward:-10, back:-10} both yielded [] before this.
    forward = max(0, entry.get("scan_forward", DEFAULT_SCAN_FORWARD))
    back = max(0, entry.get("scan_back", DEFAULT_SCAN_BACK))
    samples = max(2, entry.get("scan_samples", DEFAULT_SCAN_SAMPLES))
    span = forward + back
    step = max(1, span // (samples - 1))
    # Clamp to the configured span. `step` floors at 1, so a sample count larger than the span would
    # otherwise walk past `forward` -- {forward:10, back:0, samples:33} produced offsets 0..32, more
    # than three times the span it was asked for, and a zero span produced 0..32 as well.
    offsets = sorted({-back + i * step for i in range(samples)} | {0})
    return [o for o in offsets if -back <= o <= forward]


def _anchor_offsets(entry):
    """Flip offsets sampled either side of each anchor, always including 0 (the exact anchor).

    Spacing is GEOMETRIC, not uniform: dense near the anchor and sparse at the edges. Measured
    reason -- two identical runs through Blue Prince's intro cutscene reached the same page of the
    same text at flip 10000, so drift across an FMV is SMALL. What made them score 0.465 was that
    one was mid-fade: a few flips of drift during a transition is a large luminance change even
    though the scene is the same. Uniform spacing puts most of its samples far away, where the match
    almost never is, and leaves the near region -- where a fade needs resolving -- coarse.
    """
    span = entry.get("flip_window", DEFAULT_FLIP_WINDOW)
    samples = max(1, entry.get("window_samples", DEFAULT_WINDOW_SAMPLES))
    if span <= 0 or samples == 1:
        return [0]
    per_side = max(1, (samples - 1) // 2)
    offsets = {0}
    for i in range(1, per_side + 1):
        # i/per_side raised to a power > 1 clusters the samples toward the anchor.
        magnitude = int(round(span * (i / per_side) ** 2.2))
        magnitude = max(magnitude, i)          # never collapse two samples onto each other
        offsets.add(magnitude)
        offsets.add(-magnitude)
    return sorted(offsets)


def candidate_flips(entry):
    """Every flip the replay should capture: each anchor, plus its window. Negatives are dropped --
    a flip before the origin does not exist."""
    out = set()
    for snap in entry["snaps"]:
        for offset in window_offsets(entry, snap):
            flip = snap["pad_flip"] + offset
            if flip >= 0:
                out.add(flip)
    return out


def best_match(snap, entry, actuals, out_dir):
    """Best (ssim, record, signature) across the anchor's window, or None if nothing was captured.

    Taking the BEST is the point: the authored frame is somewhere in this span, and which sample
    lands on it depends on how fast the machine happened to be. Taking the exact anchor alone makes
    the result a measure of system load.
    """
    best = None
    reference = decode_luma(snap["luma16x9"])
    for offset in window_offsets(entry, snap):
        flip = snap["pad_flip"] + offset
        record = actuals.get(flip)
        if not record:
            continue
        image = os.path.join(out_dir, record["file"])
        if not os.path.exists(image):
            continue
        signature = signature_of(image)
        score = structural_similarity(reference, decode_luma(signature["luma16x9"]))
        if best is None or score > best[0]:
            best = (score, record, signature, offset)
    return best


def run_replay(entry, out_dir):
    """Replay the authored route and capture the presented frame at every authored anchor."""
    dump = entry["dump"]
    dump_path = dump if os.path.isabs(dump) else os.path.join(GAME_ROOT, dump)
    if not os.path.isdir(dump_path):
        raise SystemExit(f"snaps: dump not found: {dump_path}")
    if not os.path.exists(APP):
        raise SystemExit(f"snaps: prosper-app not found at {APP} (set PROSPER_APP_BIN)")
    # The whole environment -- pacing, clock, fresh savedata, route, capture anchors -- is built by
    # replay_env() so a test can assert what a check actually replays under. It used to be inlined
    # here, which is how the reader fallback regressed unnoticed: there was nothing a test could call.
    env = replay_env(entry, out_dir)
    log = os.path.join(out_dir, "run.log")
    manifest = os.path.join(out_dir, "actuals.jsonl")

    def read_actuals():
        found = {}
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue     # a record still being written; it will be there next poll
                        found[record["target_flip"]] = record
        return found

    # Stop when the LAST anchor has been captured, not when a wall-clock timer expires. The timeout
    # is a safety net for a run that hangs, not the normal way the run ends.
    #
    # This matters because of FMVs. The anchor is a flip count, so a movie contributes the same
    # number of flips however slowly it renders -- which is exactly why flips beat wall-clock and why
    # an authored route survives a title whose intro plays at 4.8 fps (measured on Blue Prince,
    # ~37x slower than its menu). But the movie still eats real SECONDS, so a fixed timeout sized for
    # a quick boot would cut the run off mid-route and report every later snap as NOT REACHED -- a
    # failure with nothing to do with rendering, which is the whole class of bug this system exists
    # to remove.
    last_anchor = max(candidate_flips(entry)) if entry["snaps"] else 0
    deadline = time.time() + entry.get("timeout", 3600)
    with open(log, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen([APP, dump_path], env=env, stdout=handle,
                                stderr=subprocess.STDOUT)
        try:
            while time.time() < deadline:
                if proc.poll() is not None:
                    break                      # the guest exited on its own
                if last_anchor in read_actuals():
                    break                      # everything the check needs has been captured
                time.sleep(2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)
    return read_actuals(), log


def cmd_check(args):
    names = args.names or list_names()
    if not names:
        print("snaps: no snap sets to check")
        return 0
    failures = 0
    for name in names:
        entry = load_entry(name)
        out_dir = os.environ.get("PROSPER_SNAP_OUT", default_check_out_dir(name))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[snaps] {name}: replaying {len(entry['snaps'])} anchor(s)")
        actuals, log = run_replay(entry, out_dir)
        threshold = entry.get("min_structural_similarity", DEFAULT_MIN_SSIM)
        review = os.path.join(HERE, "failures")
        os.makedirs(review, exist_ok=True)

        for snap in entry["snaps"]:
            flip = snap["pad_flip"]
            label = (f"  snap {snap['index']:>3} {snap['verdict']:<9} "
                     f"{snap.get('mode','anchor'):<6} flip {flip:>6}")
            found = best_match(snap, entry, actuals, out_dir)
            if not found:
                # Never silently pass a snap that was not reached: that is the failure mode the
                # whole design exists to remove.
                print(f"{label}  NOT REACHED -- the route never got here (log: {log})")
                if snap["verdict"] == "correct":
                    failures += 1
                continue
            ssim, record, actual, offset = found
            # A best match sitting on the EDGE of the window is the signature of an anchor that has
            # drifted out of range, not of a rendering change -- and without saying so the failure
            # is indistinguishable from a real regression. Measured case: Blue Prince's boot reached
            # the title screen ~600-1100 flips later under machine load, and the edge match was the
            # only visible tell.
            span = entry.get("flip_window", DEFAULT_FLIP_WINDOW)
            at_edge = (snap.get("mode") != "scan") and span > 0 and abs(offset) >= span
            actual_image = os.path.join(out_dir, record["file"])
            matched = ssim >= threshold
            # Report where in the window the best match came from. A snap that only matches at the
            # edge of its window is a warning that the anchor is drifting and will fail outright
            # once the drift exceeds the span.
            drift = record["actual_flip"] - flip
            drift_note = "" if drift == 0 else f" (matched {offset:+d}, landed {drift:+d})"

            if snap["verdict"] == "correct":
                if matched:
                    print(f"{label}  OK    ssim {ssim:.3f}{drift_note}")
                else:
                    failures += 1
                    print(f"{label}  FAIL  ssim {ssim:.3f} < {threshold}{drift_note}")
                    if at_edge:
                        print(f"    NOTE: the best match is at the EDGE of the +/-{span} flip "
                              f"window, so the anchor may simply have drifted out of range rather "
                              f"than the picture having changed. Re-import with a wider "
                              f"--flip-window before treating this as a regression.")
                    _retain(review, name, snap, actual_image, actual)
            else:
                if matched:
                    print(f"{label}  still wrong  ssim {ssim:.3f}{drift_note}")
                else:
                    # Not a failure. A known-bad frame that stopped matching may have been FIXED,
                    # and this is the only signal that would ever say so.
                    print(f"{label}  CHANGED  ssim {ssim:.3f} -- known-bad frame no longer matches; "
                          f"look at it")
                    _retain(review, name, snap, actual_image, actual)
    return 1 if failures else 0


def _retain(review, name, snap, actual_image, actual):
    """Keep the pair a human needs to judge: what was approved, and what the run produced."""
    index = snap["index"]
    actual_dst = os.path.join(review, f"{name}-{index:04d}-actual.bmp")
    shutil.copyfile(actual_image, actual_dst)
    reference = os.path.join(ref_dir(name), f"{index:04d}.bmp")
    lines = [f"    actual   {actual_dst}"]
    if os.path.exists(reference):
        expected_dst = os.path.join(review, f"{name}-{index:04d}-expected.bmp")
        shutil.copyfile(reference, expected_dst)
        lines.insert(0, f"    expected {expected_dst}")
    else:
        lines.append("    expected <no local reference -- this run was authored on another machine>")
    lines.append(f"    colors {snap['distinct_colors']} -> {actual['distinct_colors']}, "
                 f"nonblack {snap['nonblack_ratio']:.3f} -> {actual['nonblack_ratio']:.3f}")
    lines.append(f"    accept with: snaps.py accept {name} {index}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# accept
# ---------------------------------------------------------------------------------------------

def cmd_accept(args):
    entry = load_entry(args.name)
    review = os.path.join(HERE, "failures")
    changed = 0
    for index in args.indices:
        actual_bmp = os.path.join(review, f"{args.name}-{index:04d}-actual.bmp")
        if not os.path.exists(actual_bmp):
            print(f"  snap {index}: no retained actual at {actual_bmp} -- run check first")
            continue
        target = next((s for s in entry["snaps"] if s["index"] == index), None)
        if target is None:
            print(f"  snap {index}: not in '{args.name}'")
            continue
        signature = signature_of(actual_bmp)
        target.update(signature)
        if args.verdict:
            target["verdict"] = args.verdict
        os.makedirs(ref_dir(args.name), exist_ok=True)
        shutil.copyfile(actual_bmp, os.path.join(ref_dir(args.name), f"{index:04d}.bmp"))
        print(f"  snap {index}: accepted as the new {target['verdict']} reference "
              f"({signature['distinct_colors']} colors)")
        changed += 1
    if changed:
        save_entry(args.name, entry)
        print(f"\nupdated {store_path(args.name)}")
    return 0


def cmd_list(args):
    names = list_names()
    if not names:
        print("no snap sets")
        return 0
    for name in names:
        entry = load_entry(name)
        positives = sum(1 for s in entry["snaps"] if s["verdict"] == "correct")
        negatives = len(entry["snaps"]) - positives
        print(f"{name:<28} {entry.get('title_id',''):<12} "
              f"{positives:>3} correct  {negatives:>3} incorrect")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    aut = sub.add_parser("author", help="play the game and take snaps (F6/F7)")
    aut.add_argument("--name", required=True)
    aut.add_argument("--dump", required=True, help="e.g. PPSA25009-app0")
    aut.add_argument("--out", help="capture directory (default ~/snaps/<name>)")
    aut.add_argument("--det-clock", dest="det_clock", choices=("on", "off"), default="off",
                     help="pin the guest clock; OFF by default -- it replaces every guest time "
                          "source and breaks some titles (GRIS freezes on its FMV)")
    aut.add_argument("--det-fps", dest="det_fps", type=int, default=DEFAULT_DET_FPS,
                     help="guest clock rate; must match at check time")
    aut.add_argument("--savedata", choices=("fresh", "preserve"), default="fresh",
                     help="fresh (default) isolates both save roots so the run is reproducible")
    aut.add_argument("--append", action="store_true",
                     help="continue into an existing session directory")
    aut.set_defaults(func=cmd_author)

    imp = sub.add_parser("import", help="adopt an authoring session")
    imp.add_argument("capture_dir")
    imp.add_argument("--name", required=True)
    imp.add_argument("--route")
    imp.add_argument("--dump")
    imp.add_argument("--timeout", type=int, default=3600,
                     help="safety net only; the run normally ends when the last "
                          "anchor is captured")
    imp.add_argument("--min-ssim", dest="min_ssim", type=float, default=DEFAULT_MIN_SSIM)
    imp.add_argument("--det-clock", dest="det_clock", choices=("on", "off"), default="off",
                     help="must match how the session was authored")
    imp.add_argument("--det-fps", dest="det_fps", type=int, default=DEFAULT_DET_FPS,
                     help="must match how the session was authored")
    imp.add_argument("--savedata", choices=("fresh", "preserve"), default="fresh",
                     help="must match how the session was authored")
    imp.add_argument("--flip-window", dest="flip_window", type=int, default=DEFAULT_FLIP_WINDOW,
                     help="flips either side of each anchor to search (0 = exact anchor only)")
    imp.add_argument("--window-samples", dest="window_samples", type=int,
                     default=DEFAULT_WINDOW_SAMPLES, help="samples taken across the window")
    imp.set_defaults(func=cmd_import)

    chk = sub.add_parser("check", help="replay and compare")
    chk.add_argument("names", nargs="*")
    chk.set_defaults(func=cmd_check)

    acc = sub.add_parser("accept", help="promote a retained actual to be the stored snap")
    acc.add_argument("name")
    acc.add_argument("indices", nargs="+", type=int)
    acc.add_argument("--verdict", choices=("correct", "incorrect"),
                     help="also reclassify: a known-bad frame that now renders correctly")
    acc.set_defaults(func=cmd_accept)

    lst = sub.add_parser("list", help="list snap sets")
    lst.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
