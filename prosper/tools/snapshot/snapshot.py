#!/usr/bin/env python3
"""Local snapshot tester for prosper's game rendering.

Runs a game through the presented-screenshot frontend for content guards, or
through boot_trace for a targeted exact rendered-frame hash. On a mismatch it saves the evidence
and exits non-zero — snapshot testing for the renderer, so
any agent can catch a rendering regression (e.g. a recompiler change that blanks
a title) before it ships.

The game dumps are NOT in the repo (gitignored) and MUST NOT be committed, so
this runs LOCALLY, never in CI. Only small structural luminance signatures and
diagnostic hashes live in the repo (snapshots.json), never game imagery.

Usage:
  snapshot.py check  [name ...]   # compare captures to baselines; exit 1 on any diff/error
  snapshot.py verify [name ...]   # capture twice and retain multiple local review images
  snapshot.py update NAME --reviewed  # approve an inspected exact-hash candidate
  snapshot.py list

Env overrides:
  PROSPER_GAME_ROOT   dir holding the <dump> subdirs   (default: /mnt/c/Users/matti/repos/ps5ys)
  PROSPER_BOOT_TRACE  path to the boot_trace binary    (default: <prosper>/build-linux/boot_trace)
  PROSPER_SCREENSHOT  path to the screenshot binary    (default: <prosper>/build-linux/screenshot)

Exact mode targets a frame with RENDER_EVERY=1 plus `frame`=F, so frame_<F>.bmp
is the F-th draw submit's render. Pick F in a stable-content window and use
`verify` before trusting a baseline. Content mode can restrict evidence to a gameplay-only time
window of normal composited screenshots, requires multiple frames to meet `min_colors`, and can
require visible pixel changes. Its pass/fail contract uses SSIM and content coverage to catch major
collapse without rejecting subtle pixel improvements.
"""
import sys, os, json, time, hashlib, math, struct, subprocess, tempfile, shutil, signal, ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
PROSPER_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MANIFEST = os.path.join(HERE, "snapshots.json")
FAIL_DIR = os.path.join(HERE, "failures")
REVIEW_DIR = os.path.join(HERE, "review")
GAME_ROOT = os.environ.get("PROSPER_GAME_ROOT", "/mnt/c/Users/matti/repos/ps5ys")

_PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform.startswith("linux") else None


def _snapshot_child_setup(parent_pid):
    """Create an isolated process group and die if the snapshot harness disappears."""
    os.setsid()
    if _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent can die between fork() and prctl(). Once PR_SET_PDEATHSIG is
    # installed, this check closes that race without opening a new one.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _spawn_snapshot_process(command, env, logf):
    kwargs = {
        "env": env,
        "stdout": logf,
        "stderr": subprocess.STDOUT,
    }
    if sys.platform.startswith("linux"):
        parent_pid = os.getpid()
        kwargs["preexec_fn"] = lambda: _snapshot_child_setup(parent_pid)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def _stop_snapshot_process(proc):
    if proc is None:
        return
    try:
        if hasattr(os, "killpg"):
            # setsid() makes the child's PID its process-group ID. Address the
            # group directly so descendants are killed even if the leader exited.
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def boot_trace_path():
    p = os.environ.get("PROSPER_BOOT_TRACE")
    if p:
        return p
    # <prosper>/tools/snapshot/ -> <prosper>/build-linux/boot_trace
    return os.path.normpath(os.path.join(HERE, "..", "..", "build-linux", "boot_trace"))


def screenshot_path():
    p = os.environ.get("PROSPER_SCREENSHOT")
    if p:
        return p
    return os.path.normpath(os.path.join(HERE, "..", "..", "build-linux", "screenshot"))


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


def apply_entry_env(env, entry, tmp):
    """Apply route/save policy after generic renderer defaults.

    Routes are manifest data rather than shell working-directory assumptions. A fresh save lives in
    the capture temp directory and disappears with the run, so two verification runs start from the
    same console state without touching a developer's normal save.
    """
    env.update(entry.get("env", {}))
    route = entry.get("pad_script")
    if route:
        route = route if os.path.isabs(route) else os.path.join(PROSPER_ROOT, route)
        if not os.path.isfile(route):
            raise RuntimeError(f"pad script not found: {route}")
        env["PROSPER_PAD_SCRIPT"] = "@" + route
        env.setdefault("PROSPER_PAD_SCRIPT_LOG", "1")
    save_policy = entry.get("savedata_policy")
    if save_policy == "fresh":
        save_dir = os.path.join(tmp, "savedata")
        os.makedirs(save_dir, exist_ok=True)
        env["PROSPER_SAVEDATA_DIR"] = save_dir
    elif save_policy not in (None, "preserve"):
        raise RuntimeError(f"unknown savedata_policy={save_policy!r}; use fresh or preserve")


def decode_luma16x9(value):
    """Decode the screenshot manifest's fixed 16x9 8-bit luminance signature."""
    if not isinstance(value, str) or len(value) != 16 * 9 * 2:
        raise ValueError("luma16x9 must be 288 hexadecimal characters")
    try:
        return tuple(bytes.fromhex(value))
    except ValueError as exc:
        raise ValueError("luma16x9 is not valid hexadecimal") from exc


def structural_similarity(left, right):
    """Return the SSIM index for two equally sized luminance signatures.

    This is the standard SSIM luminance/contrast/structure equation over a fixed spatial thumbnail.
    It preserves large layer placement while deliberately discarding exact raster detail.
    """
    if len(left) != len(right) or not left:
        raise ValueError("SSIM inputs must have the same non-zero length")
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    variance_left = sum((value - mean_left) ** 2 for value in left) / count
    variance_right = sum((value - mean_right) ** 2 for value in right) / count
    covariance = sum((a - mean_left) * (b - mean_right)
                     for a, b in zip(left, right)) / count
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = ((mean_left ** 2 + mean_right ** 2 + c1) *
                   (variance_left + variance_right + c2))
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0,
        ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / denominator))


def analyze_content_manifest(tmp, entry):
    """Measure presented screenshots in the configured evidence window."""
    after = float(entry.get("capture_after_seconds", 0))
    before = entry.get("capture_before_seconds")
    before = float(before) if before is not None else None
    manifest = os.path.join(tmp, "capture.jsonl")
    if not os.path.isfile(manifest):
        raise RuntimeError("screenshot frontend did not write capture.jsonl")
    events = []
    with open(manifest) as f:
        for number, line in enumerate(f, 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid screenshot manifest line {number}: {exc}") from exc
    final = next((e for e in reversed(events) if e.get("type") == "summary"), None)
    if not final or final.get("exit_code") != 0 or final.get("saved") != final.get("requested"):
        raise RuntimeError(f"screenshot run did not complete successfully: {final or 'no summary'}")

    required_source = entry.get("capture_source", "composited")
    records = []
    for sample in (e for e in events if e.get("type") == "sample"):
        elapsed = float(sample.get("elapsed_seconds", -1))
        if elapsed < after or (before is not None and elapsed > before):
            continue
        if required_source and sample.get("source") != required_source:
            continue
        if "distinct_rgb_colors" not in sample:
            raise RuntimeError("screenshot manifest lacks distinct_rgb_colors; rebuild the screenshot tool")
        required_metrics = ("luma16x9", "nonblack_rgb_pixels", "average_hash", "difference_hash")
        if any(metric not in sample for metric in required_metrics):
            raise RuntimeError("screenshot manifest lacks structural metrics; rebuild the screenshot tool")
        path = sample.get("png", "")
        if not os.path.isabs(path):
            path = os.path.join(tmp, path)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise RuntimeError(f"manifest screenshot is missing or empty: {path}")
        dims = (int(sample["width"]), int(sample["height"]))
        if dims[0] <= 0 or dims[1] <= 0:
            raise RuntimeError(f"manifest screenshot has invalid dimensions: {dims}")
        records.append({
            "index": int(sample.get("index", len(records))),
            "path": path, "filename": os.path.basename(path), "elapsed": elapsed,
            "colors": int(sample["distinct_rgb_colors"]), "dims": dims,
            "hash": str(sample["pixel_crc32"]), "source": sample.get("source"),
            "source_seq": int(sample.get("source_seq", 0)),
            "frame_seq": int(sample.get("frame_seq", 0)),
            "present_count": int(sample.get("present_count", 0)),
            "front_index": int(sample.get("front_index", -1)),
            "average_hash": int(sample["average_hash"], 16),
            "difference_hash": int(sample["difference_hash"], 16),
            "luma16x9": decode_luma16x9(sample["luma16x9"]),
            "nonblack_ratio": int(sample["nonblack_rgb_pixels"]) / (dims[0] * dims[1]),
            "best_structural_similarity": None,
        })
    if not records:
        window = f"after {after:g}s" + (f" and before {before:g}s" if before is not None else "")
        raise RuntimeError(f"no {required_source or 'presented'} screenshots in evidence window ({window})")

    threshold = int(entry["min_colors"])
    qualifying = [r for r in records if r["colors"] >= threshold]
    pixel_changes = sum(a["hash"] != b["hash"] for a, b in zip(records, records[1:]))
    richest = max(records, key=lambda r: r["colors"])
    dims = {r["dims"] for r in records}
    references = entry.get("structural_references", [])
    similarities = []
    if references:
        parsed = [decode_luma16x9(reference["luma16x9"]) for reference in references]
        for record in records:
            similarity = max(structural_similarity(record["luma16x9"], reference)
                             for reference in parsed)
            record["best_structural_similarity"] = similarity
            similarities.append(similarity)
    minimum_similarity = float(entry.get("min_structural_similarity", 0.85))
    minimum_coverage = float(entry.get("min_nonblack_ratio", 0))
    return {
        "records": records,
        "qualifying": qualifying,
        "richest": richest,
        "dims": richest["dims"],
        "dimensions_consistent": len(dims) == 1,
        "pixel_changes": pixel_changes,
        "structural_similarities": similarities,
        "structural_matches": sum(value >= minimum_similarity for value in similarities),
        "nonblack_matches": sum(record["nonblack_ratio"] >= minimum_coverage
                                for record in records),
    }


def choose_review_records(summary, count=8):
    """Choose temporally spread window evidence, always including the richest frame."""
    pool = summary["records"]
    count = max(2, int(count))
    if len(pool) <= count:
        return pool
    indices = {0, len(pool) - 1}
    for i in range(1, count - 1):
        indices.add(round(i * (len(pool) - 1) / (count - 1)))
    richest = summary["richest"]
    selected = [pool[i] for i in sorted(indices)]
    if richest not in selected:
        selected[-2 if len(selected) > 1 else 0] = richest
    return sorted({r["path"]: r for r in selected}.values(), key=lambda r: r["elapsed"])


def content_result(entry, summary, include_structural=True):
    required_frames = int(entry.get("min_qualifying_frames", 2))
    required_changes = int(entry.get("min_pixel_changes", 0))
    match_ratio = float(entry.get("min_content_match_ratio", 0.75))
    expected_dims = tuple(entry["dims"]) if entry.get("dims") else None
    failures = []
    if not 0 <= match_ratio <= 1:
        failures.append(f"min_content_match_ratio {match_ratio} is outside [0, 1]")
    if len(summary["qualifying"]) < required_frames:
        failures.append(f"qualifying frames {len(summary['qualifying'])} < {required_frames}")
    if summary["pixel_changes"] < required_changes:
        failures.append(f"pixel changes {summary['pixel_changes']} < {required_changes}")
    if not summary["dimensions_consistent"]:
        failures.append("frame dimensions changed within evidence window")
    if expected_dims and summary["dims"] != expected_dims:
        failures.append(f"dimensions {summary['dims']} != {expected_dims}")
    references = entry.get("structural_references", [])
    if include_structural and references:
        required_matches = max(int(entry.get("min_structural_matches", required_frames)),
                               math.ceil(len(summary["records"]) * match_ratio))
        if summary["structural_matches"] < required_matches:
            failures.append(
                f"structural matches {summary['structural_matches']} < {required_matches} "
                f"(minimum SSIM {entry.get('min_structural_similarity', 0.85)})")
    if entry.get("min_nonblack_ratio") is not None:
        required_nonblack = max(int(entry.get("min_nonblack_matches", required_frames)),
                                math.ceil(len(summary["records"]) * match_ratio))
        if summary["nonblack_matches"] < required_nonblack:
            failures.append(
                f"non-black coverage matches {summary['nonblack_matches']} < {required_nonblack} "
                f"(minimum ratio {entry['min_nonblack_ratio']})")
    return not failures, failures


def save_content_evidence(name, run_number, summary):
    out_dir = os.path.join(REVIEW_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    selected = choose_review_records(summary)
    review_names = {}
    for index, record in enumerate(selected):
        stem, extension = os.path.splitext(record["filename"])
        leaf = f"run{run_number}_{index:02d}_{stem}_{record['colors']}colors{extension}"
        dst = os.path.join(out_dir, leaf)
        shutil.copyfile(record["path"], dst)
        saved.append(dst)
        review_names[record["path"]] = leaf
    save_content_metadata(os.path.join(out_dir, f"run{run_number}-evidence.json"),
                          summary, review_names)
    return saved


def save_content_metadata(path, summary, review_names=None):
    """Persist enough presentation identity to investigate every retained or rejected frame."""
    review_names = review_names or {}
    records = []
    for record in summary["records"]:
        records.append({
            "index": record["index"], "elapsed_seconds": record["elapsed"],
            "source": record["source"], "source_seq": record["source_seq"],
            "frame_seq": record["frame_seq"], "present_count": record["present_count"],
            "front_index": record["front_index"], "width": record["dims"][0],
            "height": record["dims"][1], "pixel_crc32": record["hash"],
            "distinct_rgb_colors": record["colors"],
            "nonblack_ratio": round(record["nonblack_ratio"], 8),
            "average_hash": f"{record['average_hash']:016x}",
            "difference_hash": f"{record['difference_hash']:016x}",
            "luma16x9": bytes(record["luma16x9"]).hex(),
            "best_structural_similarity": record["best_structural_similarity"],
            "review_png": review_names.get(record["path"]),
        })
    with open(path, "w") as f:
        json.dump({
            "schema": 1, "records": records,
            "qualifying_frames": len(summary["qualifying"]),
            "pixel_changes": summary["pixel_changes"],
        }, f, indent=2)
        f.write("\n")


def reset_review_evidence(name):
    out_dir = os.path.join(REVIEW_DIR, name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def entry_fingerprint(entry):
    stable = {k: v for k, v in entry.items()
              if k not in ("hash", "dims", "review", "structural_references",
                           "perceptual_references", "min_nonblack_ratio")}
    raw = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_exact_candidate(entry, first, second, pixel_digest, dims):
    out_dir = reset_review_evidence(entry["name"])
    first_out = os.path.join(out_dir, "run1.bmp")
    second_out = os.path.join(out_dir, "run2.bmp")
    shutil.copyfile(first, first_out)
    shutil.copyfile(second, second_out)
    candidate = {
        "mode": "exact", "name": entry["name"], "hash": pixel_digest, "dims": list(dims),
        "entry_fingerprint": entry_fingerprint(entry),
    }
    with open(os.path.join(out_dir, "candidate.json"), "w") as f:
        json.dump(candidate, f, indent=2)
        f.write("\n")
    return first_out, second_out


def approve_exact_candidate(entry):
    path = os.path.join(REVIEW_DIR, entry["name"], "candidate.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"no reviewed candidate at {path}; run verify first")
    with open(path) as f:
        candidate = json.load(f)
    if candidate.get("mode") != "exact":
        raise RuntimeError("review candidate is not an exact-frame capture")
    if candidate.get("entry_fingerprint") != entry_fingerprint(entry):
        raise RuntimeError("snapshot configuration changed after verify; regenerate review evidence")
    entry["hash"] = candidate["hash"]
    entry["dims"] = candidate["dims"]
    entry["review"] = (f"Pixel evidence approved with snapshot.py update --reviewed on "
                       f"{time.strftime('%Y-%m-%d')}")


def save_content_candidate(entry, summaries):
    out_dir = os.path.join(REVIEW_DIR, entry["name"])
    records = []
    for summary in summaries:
        records.extend(choose_review_records(summary))
    references = []
    seen = set()
    for record in records:
        signature = bytes(record["luma16x9"]).hex()
        if signature in seen:
            continue
        seen.add(signature)
        references.append({
            "luma16x9": signature,
            "average_hash": f"{record['average_hash']:016x}",
            "difference_hash": f"{record['difference_hash']:016x}",
        })
    minimum_reviewed_coverage = min(record["nonblack_ratio"] for record in records)
    candidate = {
        "mode": "content", "name": entry["name"],
        "structural_references": references,
        "suggested_min_nonblack_ratio": round(minimum_reviewed_coverage * 0.5, 6),
        "evidence_count": len(records),
        "evidence_files": sorted(
            filename for filename in os.listdir(out_dir)
            if filename.startswith("run") and filename.lower().endswith((".png", ".bmp"))),
        "dims": list(summaries[0]["dims"]),
        "entry_fingerprint": entry_fingerprint(entry),
    }
    with open(os.path.join(out_dir, "candidate.json"), "w") as f:
        json.dump(candidate, f, indent=2)
        f.write("\n")


def approve_content_candidate(entry):
    path = os.path.join(REVIEW_DIR, entry["name"], "candidate.json")
    if not os.path.isfile(path):
        raise RuntimeError(f"no reviewed candidate at {path}; run verify first")
    with open(path) as f:
        candidate = json.load(f)
    if candidate.get("mode") != "content":
        raise RuntimeError("review candidate is not a content capture")
    if candidate.get("entry_fingerprint") != entry_fingerprint(entry):
        raise RuntimeError("snapshot configuration changed after verify; regenerate review evidence")
    if candidate.get("evidence_count", 0) < 4 or not candidate.get("structural_references"):
        raise RuntimeError("candidate does not contain enough multi-frame review evidence")
    evidence_files = candidate.get("evidence_files", [])
    if len(evidence_files) != candidate["evidence_count"] or any(
            not os.path.isfile(os.path.join(os.path.dirname(path), filename))
            for filename in evidence_files):
        raise RuntimeError("review evidence is incomplete; run verify again")
    entry["structural_references"] = candidate["structural_references"]
    entry.setdefault("min_nonblack_ratio", candidate["suggested_min_nonblack_ratio"])
    entry["dims"] = candidate["dims"]
    entry["review"] = (
        f"Approved {candidate['evidence_count']} composited images from two independent runs on "
        f"{time.strftime('%Y-%m-%d')}; intended scene, layers, and progression visually confirmed.")


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
    tmp = tempfile.mkdtemp(prefix="snap_")
    apply_entry_env(env, entry, tmp)
    env["PROSPER_FRAME_DIR"] = tmp
    target = os.path.join(tmp, "frame_%04d.bmp" % frame)
    # Run a UNIQUELY-NAMED copy of the binary: this repo is worked by several agents at once, and a
    # concurrent `pkill -x boot_trace` would otherwise kill our capture mid-boot.
    bt_run = os.path.join(tmp, "bt_snap")
    shutil.copyfile(bt, bt_run)
    os.chmod(bt_run, 0o755)
    proc = None
    logf = None
    failed = False
    try:
        logf = open(run_log, "wb") if run_log else subprocess.DEVNULL
        proc = _spawn_snapshot_process([bt_run, dump], env, logf)
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
    except Exception:
        failed = True
        raise
    finally:
        _stop_snapshot_process(proc)
        if run_log and logf is not None:
            logf.close()
        if failed:
            _cleanup(tmp)


def capture_content(entry, run_log=None):
    """Run the title and measure normal composited screenshots, never renderer intermediates."""
    frontend = screenshot_path()
    if not os.path.exists(frontend):
        raise RuntimeError(f"screenshot not found at {frontend} (build it, or set PROSPER_SCREENSHOT)")
    dump = resolve_dump(entry)
    if not os.path.exists(dump):
        raise RuntimeError(f"game dump not found: {dump} (set PROSPER_GAME_ROOT)")
    scale = int(entry.get("scale", 4))
    timeout = int(entry.get("timeout", 240))
    sample_seconds = float(entry.get("sample_seconds", 1))
    if sample_seconds <= 0:
        raise RuntimeError("sample_seconds must be positive")
    env = dict(os.environ)
    env.update({
        "PROSPER_GUEST_FS": "1", "PROSPER_GUEST_ARGS": "-force-gfx-direct",
        "PROSPER_RENDER": "1", "PROSPER_RENDER_EVERY": "1", "PROSPER_RENDER_FIRST": "0",
        "PROSPER_RENDER_SCALE": str(scale), "PROSPER_FAULT_ONSTACK": "1",
    })
    tmp = tempfile.mkdtemp(prefix="snap_")
    proc = None
    logf = None
    failed = False
    try:
        apply_entry_env(env, entry, tmp)
        frontend_run = os.path.join(tmp, "screenshot_snap")
        shutil.copyfile(frontend, frontend_run)
        os.chmod(frontend_run, 0o755)
        capture_end = float(entry.get("capture_before_seconds", timeout))
        count = max(2, int(math.ceil(capture_end / sample_seconds)) + 1)
        runner_timeout = max(timeout, int(math.ceil(capture_end))) + 30
        command = [
            frontend_run, dump, "--seconds", str(sample_seconds), "--count", str(count),
            "--out", tmp, "--manifest", os.path.join(tmp, "capture.jsonl"),
            "--timeout", str(runner_timeout), "--require-composited-frame",
        ]
        logf = open(run_log, "wb") if run_log else subprocess.DEVNULL
        proc = _spawn_snapshot_process(command, env, logf)
        deadline = time.time() + runner_timeout + 15
        while time.time() < deadline and proc.poll() is None:
            time.sleep(1.0)
        if proc.poll() is None:
            raise RuntimeError(f"screenshot frontend exceeded {runner_timeout + 15}s")
        if proc.returncode != 0:
            raise RuntimeError(f"screenshot frontend exited with code {proc.returncode}")
        summary = analyze_content_manifest(tmp, entry)
        return summary["richest"]["path"], summary, tmp
    except Exception:
        failed = True
        raise
    finally:
        _stop_snapshot_process(proc)
        if run_log and logf is not None:
            logf.close()
        if failed:
            _cleanup(tmp)


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
        if "min_colors" in s:
            mode = (f"min_colors={s['min_colors']} "
                    f"min_frames={s.get('min_qualifying_frames', 2)} "
                    f"min_changes={s.get('min_pixel_changes', 0)} "
                    f"references={len(s.get('structural_references', []))}")
        else:
            mode = (f"frame={s.get('frame', 'MISSING')} "
                    f"hash={'set' if s.get('hash') else 'MISSING'}")
        print(f"  {s['name']:<28} dump={s['dump']} {mode}")


def cmd_update(m, names):
    # Exact hashes require two identical verify captures plus explicit review of the saved images.
    reviewed = "--reviewed" in names
    names = [n for n in names if n != "--reviewed"]
    rc = 0
    for s in select(m, names):
        try:
            if s.get("min_colors"):
                if not reviewed:
                    print(f"[update] {s['name']}: REFUSED — run verify, inspect every image from "
                          f"both runs, then rerun update {s['name']} --reviewed", file=sys.stderr)
                    rc = 1
                    continue
                approve_content_candidate(s)
                print(f"[update] {s['name']}: approved "
                      f"{len(s['structural_references'])} structural references")
                continue
            if not reviewed:
                print(f"[update] {s['name']}: REFUSED — run verify, inspect both review images, "
                      f"then rerun update {s['name']} --reviewed", file=sys.stderr)
                rc = 1
                continue
            approve_exact_candidate(s)
            print(f"[update] {s['name']}: approved {s['dims'][0]}x{s['dims'][1]} "
                  f"hash={s['hash'][:16]}…")
        except Exception as e:
            print(f"[update] {s['name']}: ERROR {e}", file=sys.stderr)
            rc = 1
    save_manifest(m)
    return rc


def cmd_verify(m, names):
    rc = 0
    for s in select(m, names):
        temps = []
        try:
            if s.get("min_colors"):
                out_dir = reset_review_evidence(s["name"])
                _, summary1, t1 = capture_content(s, run_log=os.path.join(out_dir, "run1.log"))
                temps.append(t1)
                saved1 = save_content_evidence(s["name"], 1, summary1)
                _cleanup(t1)
                _, summary2, t2 = capture_content(s, run_log=os.path.join(out_dir, "run2.log"))
                temps.append(t2)
                saved2 = save_content_evidence(s["name"], 2, summary2)
                _cleanup(t2)
                ok1, why1 = content_result(s, summary1, include_structural=False)
                ok2, why2 = content_result(s, summary2, include_structural=False)
                ok = ok1 and ok2 and summary1["dims"] == summary2["dims"]
                if ok:
                    save_content_candidate(s, (summary1, summary2))
                print(f"[verify] {s['name']}: {'CONTENT-STABLE' if ok else 'UNSTABLE'} "
                      f"(richest={summary1['richest']['colors']}/{summary2['richest']['colors']}, "
                      f"qualifying={len(summary1['qualifying'])}/{len(summary2['qualifying'])}, "
                      f"changes={summary1['pixel_changes']}/{summary2['pixel_changes']}, "
                      f"dims={summary1['dims'][0]}x{summary1['dims'][1]}/"
                      f"{summary2['dims'][0]}x{summary2['dims'][1]})")
                if why1 or why2:
                    print(f"         failures: run1={why1 or 'none'} run2={why2 or 'none'}")
                print(f"         REVIEW REQUIRED: inspect {len(saved1) + len(saved2)} images in {out_dir}")
                if not ok:
                    rc = 1
                continue
            b1, t1 = capture(s); temps.append(t1); h1, dims1 = pixel_hash(b1)
            b2, t2 = capture(s); temps.append(t2); h2, dims2 = pixel_hash(b2)
            ok = h1 == h2 and dims1 == dims2
            print(f"[verify] {s['name']}: {'DETERMINISTIC' if ok else 'NON-DETERMINISTIC'} "
                  f"({h1[:12]} vs {h2[:12]})")
            if ok:
                first, second = save_exact_candidate(s, b1, b2, h1, dims1)
                print(f"         REVIEW REQUIRED: inspect {first} and {second}")
            _cleanup(t1); _cleanup(t2)
            if not ok:
                rc = 1
        except Exception as e:
            for tmp in temps:
                _cleanup(tmp)
            print(f"[verify] {s['name']}: ERROR {e}", file=sys.stderr)
            rc = 1
    return rc


def cmd_check(m, names):
    os.makedirs(FAIL_DIR, exist_ok=True)
    rc = 0
    for s in select(m, names):
        tmp = None
        min_colors = s.get("min_colors")
        base = s.get("hash")
        if not min_colors and not base:
            print(f"[check] {s['name']}: NO BASELINE — run verify, inspect its review images, then "
                  f"run `snapshot.py update {s['name']} --reviewed`", file=sys.stderr)
            rc = 1
            continue
        if base and (not s.get("review") or str(s["review"]).lower() == "pending"):
            print(f"[check] {s['name']}: baseline lacks a completed visual-review note", file=sys.stderr)
            rc = 1
            continue
        log = os.path.join(FAIL_DIR, f"{s['name']}.log")
        try:
            if min_colors:                                  # content-metric mode (robust to pixel variance)
                if not s.get("review") or str(s["review"]).lower() == "pending":
                    raise RuntimeError("content guard lacks a completed visual-review note")
                if not s.get("structural_references"):
                    raise RuntimeError("content guard lacks reviewed structural references")
                _, summary, tmp = capture_content(s, run_log=log)
                ok, failures = content_result(s, summary)
                if ok:
                    print(f"[check] {s['name']}: OK ({summary['dims'][0]}x{summary['dims'][1]}, "
                          f"frames={len(summary['records'])}, qualifying={len(summary['qualifying'])}, "
                          f"richest={summary['richest']['colors']} colors, "
                          f"pixel_changes={summary['pixel_changes']}, "
                          f"structural_matches={summary['structural_matches']}, "
                          f"nonblack_matches={summary['nonblack_matches']})")
                    os.remove(log) if os.path.exists(log) else None
                else:
                    failure_paths = []
                    review_names = {}
                    for i, record in enumerate(choose_review_records(summary)):
                        extension = os.path.splitext(record["path"])[1]
                        out = os.path.join(FAIL_DIR, f"{s['name']}-{i:02d}{extension}")
                        shutil.copyfile(record["path"], out)
                        failure_paths.append(out)
                        review_names[record["path"]] = os.path.basename(out)
                    metadata = os.path.join(FAIL_DIR, f"{s['name']}-evidence.json")
                    save_content_metadata(metadata, summary, review_names)
                    print(f"[check] {s['name']}: FAIL — {'; '.join(failures)}")
                    print(f"         screenshots: {', '.join(failure_paths)}")
                    print(f"         metadata:    {metadata}")
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
            if tmp:
                _cleanup(tmp)
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
