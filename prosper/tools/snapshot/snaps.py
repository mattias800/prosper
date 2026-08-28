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
import tempfile
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
    route = os.path.join(out_dir, "route.pad")

    env = dict(os.environ)
    env.update({
        "PROSPER_RENDER": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_SNAP_DIR": out_dir,
        "PROSPER_PAD_RECORD": route,
    })
    if args.savedata == "fresh":
        apply_fresh_savedata(env, out_dir)
        print("  savedata: FRESH (your real saves are untouched, and the check starts here too)")
    else:
        print("  savedata: PRESERVE -- this session uses your real saves, so the route will NOT\n"
              "            reproduce on a machine whose save state differs. A title that offers\n"
              "            'Continue' puts it above 'New Game', so the same inputs pick a\n"
              "            different item. Only use this deliberately.")
    # Deliberately NOT offscreen: authoring is a person looking at a window.
    env.pop("SDL_VIDEODRIVER", None)

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
            "pad_flip": record["pad_flip"],
            **signature,
        })
        print(f"  snap {record['index']:>3} {record['verdict']:<9} flip {record['pad_flip']:>6}  "
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

def window_offsets(entry):
    """Flip offsets sampled either side of each anchor, always including 0 (the exact anchor)."""
    span = entry.get("flip_window", DEFAULT_FLIP_WINDOW)
    samples = max(1, entry.get("window_samples", DEFAULT_WINDOW_SAMPLES))
    if span <= 0 or samples == 1:
        return [0]
    step = (2 * span) // (samples - 1)
    offsets = sorted({-span + i * step for i in range(samples)} | {0})
    return offsets


def candidate_flips(entry):
    """Every flip the replay should capture: each anchor, plus its window. Negatives are dropped --
    a flip before the origin does not exist."""
    out = set()
    for snap in entry["snaps"]:
        for offset in window_offsets(entry):
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
    for offset in window_offsets(entry):
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
    flips = ",".join(str(f) for f in sorted(candidate_flips(entry)))
    env = dict(os.environ)
    if entry.get("savedata", "fresh") == "fresh":
        apply_fresh_savedata(env, out_dir)
    env.update({
        "SDL_VIDEODRIVER": "offscreen",
        "PROSPER_RENDER": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_SNAP_DIR": out_dir,
        "PROSPER_SNAP_AT_FLIPS": flips,
        "PROSPER_PAD_SCRIPT": "@" + os.path.join(REPO_ROOT, entry["route"]),
    })
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
        out_dir = os.path.join(tempfile.gettempdir(), f"prosper-snaps-{name}")
        out_dir = os.environ.get("PROSPER_SNAP_OUT", out_dir)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[snaps] {name}: replaying {len(entry['snaps'])} anchor(s)")
        actuals, log = run_replay(entry, out_dir)
        threshold = entry.get("min_structural_similarity", DEFAULT_MIN_SSIM)
        review = os.path.join(HERE, "failures")
        os.makedirs(review, exist_ok=True)

        for snap in entry["snaps"]:
            flip = snap["pad_flip"]
            label = f"  snap {snap['index']:>3} {snap['verdict']:<9} flip {flip:>6}"
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
            at_edge = span > 0 and abs(offset) >= span
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
