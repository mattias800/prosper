#!/usr/bin/env python3
"""Golden-image snapshot tester for prosper's game rendering.

Runs a game through boot_trace, captures an EXACT rendered frame, hashes its
pixels, and compares against a stored baseline. On a mismatch it saves the
offending screenshot and exits non-zero — snapshot testing for the renderer, so
any agent can catch a rendering regression (e.g. a recompiler change that blanks
a title) before it ships.

The game dumps are NOT in the repo (gitignored) and MUST NOT be committed, so
this runs LOCALLY, never in CI. Only the small pixel HASHES live in the repo
(snapshots.json) — a hash is a checksum, not game imagery.

Usage:
  snapshot.py check  [name ...]   # compare captures to baselines; exit 1 on any diff/error
  snapshot.py update [name ...]   # (re)capture and store/overwrite baseline hashes
  snapshot.py verify [name ...]   # capture twice each; report whether the frame is deterministic
  snapshot.py list

Env overrides:
  PROSPER_GAME_ROOT   dir holding the <dump> subdirs   (default: /mnt/c/Users/matti/repos/ps5ys)
  PROSPER_BOOT_TRACE  path to the boot_trace binary    (default: <prosper>/build-linux/boot_trace)

A frame is targeted by RENDER_EVERY=1 (render every draw submit) + `frame`=F, so
frame_<F>.bmp is the F-th draw submit's render. Pick F in a STABLE-content window
(a static title/menu loop renders the same composite every submit → same hash);
use `verify` to confirm F is deterministic before trusting a baseline.
"""
import sys, os, json, time, hashlib, struct, subprocess, tempfile, shutil, signal

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "snapshots.json")
FAIL_DIR = os.path.join(HERE, "failures")
GAME_ROOT = os.environ.get("PROSPER_GAME_ROOT", "/mnt/c/Users/matti/repos/ps5ys")


def boot_trace_path():
    p = os.environ.get("PROSPER_BOOT_TRACE")
    if p:
        return p
    # <prosper>/tools/snapshot/ -> <prosper>/build-linux/boot_trace
    return os.path.normpath(os.path.join(HERE, "..", "..", "build-linux", "boot_trace"))


def load_manifest():
    if not os.path.exists(MANIFEST):
        return {"snapshots": []}
    with open(MANIFEST) as f:
        return json.load(f)


def save_manifest(m):
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")


def pixel_hash(bmp_path):
    """SHA-256 of the BMP's pixel data (header-independent so tool/metadata
    differences never trip it — only the actual rendered pixels do)."""
    with open(bmp_path, "rb") as f:
        d = f.read()
    if len(d) < 54 or d[:2] != b"BM":
        raise ValueError(f"{bmp_path}: not a BMP ({len(d)} bytes)")
    off = struct.unpack_from("<I", d, 10)[0]
    w = struct.unpack_from("<i", d, 18)[0]
    h = abs(struct.unpack_from("<i", d, 22)[0])
    return hashlib.sha256(d[off:]).hexdigest(), (w, h)


def distinct_colors(bmp_path):
    """Count distinct RGB pixel values + return (count, (w,h)). A content metric that is ROBUST to the
    run-to-run pixel variance of a threaded boot (where exact hashing flakes) while still catching the
    regression that matters: a dropped/rejected shader collapses the frame to the debug clear (1 color),
    whereas a real rendered frame has hundreds+. Use `min_colors` in the manifest for such titles."""
    with open(bmp_path, "rb") as f:
        d = f.read()
    if len(d) < 54 or d[:2] != b"BM":
        raise ValueError(f"{bmp_path}: not a BMP ({len(d)} bytes)")
    off = struct.unpack_from("<I", d, 10)[0]
    w = struct.unpack_from("<i", d, 18)[0]
    h = abs(struct.unpack_from("<i", d, 22)[0])
    bpp = struct.unpack_from("<H", d, 28)[0]
    step = bpp // 8
    px = d[off:]
    n = min(len(px) // step, w * h) if step else 0
    colors = {px[i:i + 3] for i in range(0, n * step, step)}
    return len(colors), (w, h)


def resolve_dump(entry):
    dump = entry["dump"]
    return dump if os.path.isabs(dump) else os.path.join(GAME_ROOT, dump)


def capture(entry, run_log=None):
    """Boot the game, wait for frame_<F>.bmp, return (bmp_bytes_path_copy, (w,h)).
    Raises RuntimeError on timeout / crash-before-frame."""
    bt = boot_trace_path()
    if not os.path.exists(bt):
        raise RuntimeError(f"boot_trace not found at {bt} (build it, or set PROSPER_BOOT_TRACE)")
    dump = resolve_dump(entry)
    if not os.path.exists(dump):
        raise RuntimeError(f"game dump not found: {dump} (set PROSPER_GAME_ROOT)")

    frame = int(entry["frame"])
    scale = int(entry.get("scale", 4))
    timeout = int(entry.get("timeout", 240))

    env = dict(os.environ)
    # Defaults every title needs to reach the frame loop, then the render/target knobs. The manifest
    # entry's own `env` wins so a title can add its specific switches.
    env.update({
        "PROSPER_GUEST_FS": "1",
        "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_RENDER": "1",
        "PROSPER_RENDER_EVERY": "1",     # render every draw submit -> frame_<F> == F-th draw submit
        "PROSPER_RENDER_FIRST": "0",
        "PROSPER_RENDER_SCALE": str(scale),
        "PROSPER_FAULT_ONSTACK": "1",    # a guest-thread fault is reported, not a silent kill
    })
    env.update(entry.get("env", {}))

    tmp = tempfile.mkdtemp(prefix="snap_")
    env["PROSPER_FRAME_DIR"] = tmp
    target = os.path.join(tmp, "frame_%04d.bmp" % frame)
    # Run a UNIQUELY-NAMED copy of the binary: this repo is worked by several agents at once, and a
    # concurrent `pkill -x boot_trace` would otherwise kill our capture mid-boot.
    bt_run = os.path.join(tmp, "bt_snap")
    shutil.copyfile(bt, bt_run)
    os.chmod(bt_run, 0o755)
    logf = open(run_log, "wb") if run_log else subprocess.DEVNULL
    try:
        proc = subprocess.Popen([bt_run, dump], env=env, stdout=logf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
        deadline = time.time() + timeout
        settle = int(entry.get("settle", 0))
        if settle > 0:
            # SETTLE MODE: return once the render STABILIZES — the last `settle` frames all hash
            # identical (a static/settled screen). This is deterministic across runs, unlike the
            # timing-dependent N-th-draw-submit target (`frame`), which varies with boot threading.
            seen, recent = {}, []
            while time.time() < deadline:
                exited = proc.poll() is not None
                frames = sorted(f for f in os.listdir(tmp) if f.startswith("frame_") and f.endswith(".bmp"))
                for i, f in enumerate(frames):
                    if f in seen:
                        continue
                    if i == len(frames) - 1 and not exited:
                        continue                         # newest frame may be mid-write; hash it next poll
                    p = os.path.join(tmp, f)
                    try:
                        if os.path.getsize(p) <= 0:
                            continue
                        h, _ = pixel_hash(p)
                    except OSError:
                        continue
                    seen[f] = h
                    recent.append((p, h))
                    if len(recent) >= settle and len({x[1] for x in recent[-settle:]}) == 1:
                        dst = os.path.join(tmp, "captured.bmp")
                        shutil.copyfile(p, dst)
                        return dst, tmp
                if exited:
                    raise RuntimeError(f"boot_trace exited (code {proc.returncode}) before settling to "
                                       f"{settle} stable frames (reached: {_last_frame(tmp)})")
                time.sleep(0.3)
            raise RuntimeError(f"timeout after {timeout}s waiting for {settle} stable frames "
                               f"(reached: {_last_frame(tmp)})")
        last_size = -1
        while time.time() < deadline:
            if proc.poll() is not None and not os.path.exists(target):
                raise RuntimeError(f"boot_trace exited (code {proc.returncode}) before frame {frame} "
                                   f"(reached: {_last_frame(tmp)})")
            if os.path.exists(target):
                sz = os.path.getsize(target)
                if sz > 0 and sz == last_size:          # size stable across a poll -> fully written
                    dst = os.path.join(tmp, "captured.bmp")
                    shutil.copyfile(target, dst)
                    return dst, tmp
                last_size = sz
            time.sleep(0.3)
        raise RuntimeError(f"timeout after {timeout}s waiting for frame {frame} "
                           f"(reached: {_last_frame(tmp)})")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        if run_log:
            logf.close()


def capture_richest(entry, run_log=None):
    """Run the whole boot and return the RICHEST frame it produced: (best_bmp_path, max_colors, dims, tmp).
    For a threaded boot whose per-frame output isn't reproducible, neither an exact hash nor a single
    settled frame is stable (the settled screen — a blank loading flicker vs real content — varies). The
    MAX distinct-color count OVER THE WHOLE RUN is robust: a working render always produces some rich
    frame; a rejected/dropped shader collapses every frame to the debug clear (~1 color). Powers the
    `min_colors` guard (see distinct_colors)."""
    bt = boot_trace_path()
    if not os.path.exists(bt):
        raise RuntimeError(f"boot_trace not found at {bt} (build it, or set PROSPER_BOOT_TRACE)")
    dump = resolve_dump(entry)
    if not os.path.exists(dump):
        raise RuntimeError(f"game dump not found: {dump} (set PROSPER_GAME_ROOT)")
    scale = int(entry.get("scale", 4))
    timeout = int(entry.get("timeout", 240))
    env = dict(os.environ)
    env.update({
        "PROSPER_GUEST_FS": "1", "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_RENDER": "1", "PROSPER_RENDER_EVERY": "1", "PROSPER_RENDER_FIRST": "0",
        "PROSPER_RENDER_SCALE": str(scale), "PROSPER_FAULT_ONSTACK": "1",
    })
    env.update(entry.get("env", {}))
    tmp = tempfile.mkdtemp(prefix="snap_")
    env["PROSPER_FRAME_DIR"] = tmp
    bt_run = os.path.join(tmp, "bt_snap")
    shutil.copyfile(bt, bt_run); os.chmod(bt_run, 0o755)
    logf = open(run_log, "wb") if run_log else subprocess.DEVNULL
    try:
        proc = subprocess.Popen([bt_run, dump], env=env, stdout=logf, stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid)
        deadline = time.time() + timeout
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1.0)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        if run_log:
            logf.close()
    frames = sorted(f for f in os.listdir(tmp) if f.startswith("frame_") and f.endswith(".bmp"))
    if not frames:
        raise RuntimeError(f"no frames rendered in {timeout}s")
    best_p, best_n, best_dims = None, -1, (0, 0)
    for f in frames:
        p = os.path.join(tmp, f)
        try:
            if os.path.getsize(p) <= 0:
                continue
            n, dims = distinct_colors(p)
        except (OSError, ValueError):
            continue
        if n > best_n:
            best_p, best_n, best_dims = p, n, dims
    dst = os.path.join(tmp, "captured.bmp")
    shutil.copyfile(best_p, dst)
    return dst, best_n, best_dims, tmp


def _last_frame(tmp):
    fs = sorted(f for f in os.listdir(tmp) if f.startswith("frame_") and f.endswith(".bmp"))
    return fs[-1] if fs else "no frames"


def _cleanup(tmp):
    shutil.rmtree(tmp, ignore_errors=True)


def select(m, names):
    snaps = m["snapshots"]
    if names:
        by = {s["name"]: s for s in snaps}
        missing = [n for n in names if n not in by]
        if missing:
            print(f"unknown snapshot(s): {', '.join(missing)}", file=sys.stderr)
            sys.exit(2)
        return [by[n] for n in names]
    return snaps


def cmd_list(m, names):
    for s in m["snapshots"]:
        print(f"  {s['name']:<28} dump={s['dump']} frame={s['frame']} "
              f"hash={'set' if s.get('hash') else 'MISSING'}")


def cmd_update(m, names):
    # Bless a NEW baseline hash — but only if the target is DETERMINISTIC. Capture twice and refuse to
    # write the hash unless both captures agree; a baseline that isn't reproducible is worse than none
    # (it fails `check` for every other agent/machine and trains people to ignore the guard). A target
    # like "the N-th draw submit" (PROSPER_RENDER_EVERY=1) is inherently timing/threading-dependent on a
    # threaded boot, so it will (correctly) be rejected here until the entry targets a settled/stable
    # capture point. `--force` blesses a single capture anyway (escape hatch; not recommended).
    force = "--force" in names
    names = [n for n in names if n != "--force"]
    rc = 0
    for s in select(m, names):
        try:
            if s.get("min_colors"):     # content-metric entry: nothing to bless; report the margin
                b, nc, dims, t = capture_richest(s); _cleanup(t)
                ok = nc >= int(s["min_colors"])
                print(f"[update] {s['name']}: {dims[0]}x{dims[1]}, richest frame {nc} distinct colors "
                      f"(threshold min_colors={s['min_colors']}, {'OK' if ok else 'BELOW'})")
                if not ok:
                    rc = 1
                continue
            b1, t1 = capture(s); h1, dims = pixel_hash(b1); _cleanup(t1)
            if force:
                s["hash"] = h1; s["dims"] = list(dims)
                print(f"[update] {s['name']}: {dims[0]}x{dims[1]} hash={h1[:16]}… (FORCED, unverified)")
                continue
            b2, t2 = capture(s); h2, _ = pixel_hash(b2); _cleanup(t2)
            if h1 != h2:
                print(f"[update] {s['name']}: REFUSED — non-deterministic ({h1[:12]} vs {h2[:12]}); "
                      f"target a settled/stable frame (or --force to override)", file=sys.stderr)
                rc = 1
                continue
            s["hash"] = h1; s["dims"] = list(dims)
            print(f"[update] {s['name']}: {dims[0]}x{dims[1]} hash={h1[:16]}… (verified deterministic)")
        except Exception as e:
            print(f"[update] {s['name']}: ERROR {e}", file=sys.stderr)
            rc = 1
    save_manifest(m)
    return rc


def cmd_verify(m, names):
    rc = 0
    for s in select(m, names):
        try:
            b1, t1 = capture(s); h1, _ = pixel_hash(b1); _cleanup(t1)
            b2, t2 = capture(s); h2, _ = pixel_hash(b2); _cleanup(t2)
            ok = h1 == h2
            print(f"[verify] {s['name']}: {'DETERMINISTIC' if ok else 'NON-DETERMINISTIC'} "
                  f"({h1[:12]} vs {h2[:12]})")
            if not ok:
                rc = 1
        except Exception as e:
            print(f"[verify] {s['name']}: ERROR {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_check(m, names):
    os.makedirs(FAIL_DIR, exist_ok=True)
    rc = 0
    for s in select(m, names):
        min_colors = s.get("min_colors")
        base = s.get("hash")
        if not min_colors and not base:
            print(f"[check] {s['name']}: NO BASELINE — run `snapshot.py update {s['name']}`", file=sys.stderr)
            rc = 1
            continue
        log = os.path.join(FAIL_DIR, f"{s['name']}.log")
        try:
            if min_colors:                                  # content-metric mode (robust to pixel variance)
                bmp, nc, dims, tmp = capture_richest(s, run_log=log)
                if nc >= int(min_colors):
                    print(f"[check] {s['name']}: OK ({dims[0]}x{dims[1]}, richest {nc} colors >= {min_colors})")
                    os.remove(log) if os.path.exists(log) else None
                else:
                    out = os.path.join(FAIL_DIR, f"{s['name']}.bmp")
                    shutil.copyfile(bmp, out)
                    print(f"[check] {s['name']}: FAIL — richest frame only {nc} distinct colors < min "
                          f"{min_colors} (render collapsed to the debug clear?)")
                    print(f"         screenshot: {out}")
                    print(f"         boot log:   {log}")
                    rc = 1
                _cleanup(tmp)
                continue
            bmp, tmp = capture(s, run_log=log)
            h, dims = pixel_hash(bmp)
            if h == base:
                print(f"[check] {s['name']}: OK ({dims[0]}x{dims[1]})")
                os.remove(log) if os.path.exists(log) else None
            else:
                out = os.path.join(FAIL_DIR, f"{s['name']}.bmp")
                shutil.copyfile(bmp, out)
                print(f"[check] {s['name']}: FAIL — hash {h[:16]}… != baseline {base[:16]}…")
                print(f"         screenshot: {out}")
                print(f"         boot log:   {log}")
                rc = 1
            _cleanup(tmp)
        except Exception as e:
            print(f"[check] {s['name']}: ERROR {e}  (log: {log})", file=sys.stderr)
            rc = 1
    return rc


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "update", "verify", "list"):
        print(__doc__)
        sys.exit(2)
    cmd, names = sys.argv[1], sys.argv[2:]
    m = load_manifest()
    rc = {"check": cmd_check, "update": cmd_update, "verify": cmd_verify, "list": cmd_list}[cmd](m, names)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
