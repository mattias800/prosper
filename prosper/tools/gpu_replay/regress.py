#!/usr/bin/env python3
# regress.py (#1258) — a fast, cross-title capsule-hash regression gate.
#
# Replays every .prgcap in a LOCAL corpus directory through gpu_replay and diffs each capsule's
# rendered output hash against a committed baseline. Runs in seconds (deterministic offline replay,
# no live game boot), so a shared GPU/executor/recompiler change can be checked against many scenes
# before the ~18-minute live-boot snapshot suite.
#
#   regress.py update <corpus-dir> [--baseline FILE]   # (re)record baseline hashes after an INTENDED change
#   regress.py check  <corpus-dir> [--baseline FILE]    # compare current hashes to the baseline; exit 1 on any diff
#   regress.py list   <corpus-dir>                      # just print each capsule's current hash
#
# The corpus (.prgcap files) is LOCAL and gitignored — capsules are large and carry game imagery, exactly
# like the game dumps. Only the small baseline (basename -> hash, JSON) is meant to be committed/shared.
#
# SCOPE (important): a replay hash validates the TRANSLATION path (recompiler, AGC/PM4 decode, render-state
# resolve, executor ordering, detile) — it is the right guard for changes to THAT code. It does NOT exercise
# live GPU residency; a change to live_gpu_targets/residency can be hash-identical yet wrong (see
# docs blind-spot note / issue #1103). Use this to catch translation regressions fast, not as full coverage.

import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

HASH_RE = re.compile(r"\bhash=([0-9a-fA-F]{1,16})\b")
DEFAULT_REPLAY = os.environ.get("PROSPER_GPU_REPLAY", "build-linux/gpu_replay")


def parse_output_hash(text: str):
    """Extract the rendered output hash from gpu_replay's log — the `hash=` on the render summary line
    `[gpureplay] output=WxH ... hash=H`. Take the LAST such line so per-resource/-seed hashes earlier in
    the log never shadow the final output hash. Returns the lowercase hex string, or None if absent."""
    out_hash = None
    for line in text.splitlines():
        if "output=" in line and "hash=" in line:
            m = HASH_RE.search(line)
            if m:
                out_hash = m.group(1).lower()
    return out_hash


def classify_check(current: dict, baseline: dict):
    """Compare current capsule hashes to the baseline. Returns (regressions, errors, new, missing):
    regressions = [(name, baseline_hash, current_hash)] a committed hash CHANGED (the real signal);
    errors = capsules that failed to replay (current hash is None); new = present now but not baselined;
    missing = in the baseline but absent from the corpus. Only regressions+errors fail the gate."""
    regressions = [(n, baseline[n], current[n]) for n in current
                   if current[n] is not None and n in baseline and baseline[n] != current[n]]
    errors = [n for n in current if current[n] is None]
    new = [n for n in current if current[n] is not None and n not in baseline]
    missing = [n for n in baseline if n not in current]
    return regressions, errors, new, missing


def replay_hash(replay_bin: str, capsule: Path, timeout: int) -> str:
    """Run gpu_replay on one capsule and return its rendered output hash. Raises on failure so a broken
    capsule is loud, not silent."""
    # Scrub PROSPER_* from the child env so a diagnostic left in the caller's shell (PROSPER_RENDER_SCALE,
    # PROSPER_DETILE, ...) can't perturb the hash and produce a false regression — the gate must be
    # reproducible regardless of the caller's environment. gpu_replay applies the capsule's own renderer_env
    # from the capture metadata, so it needs no PROSPER_* from the parent.
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("PROSPER_")}
    proc = subprocess.run([replay_bin, str(capsule)], capture_output=True, text=True,
                          timeout=timeout, env=child_env)
    out_hash = parse_output_hash(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gpu_replay failed for {capsule.name} (exit {proc.returncode})"
            + (f" after reporting hash {out_hash}" if out_hash is not None else "")
            + "\n--- stderr tail ---\n" + "\n".join(proc.stderr.splitlines()[-8:]))
    if out_hash is None:
        raise RuntimeError(
            f"no output hash from gpu_replay for {capsule.name} (exit {proc.returncode})\n"
            f"--- stderr tail ---\n" + "\n".join((proc.stderr).splitlines()[-8:]))
    return out_hash


def corpus_hashes(replay_bin: str, corpus: Path, timeout: int) -> dict:
    capsules = sorted(corpus.glob("*.prgcap"))
    if not capsules:
        sys.exit(f"regress: no .prgcap files in {corpus}")
    result = {}
    for cap in capsules:
        try:
            result[cap.name] = replay_hash(replay_bin, cap, timeout)
            print(f"  {cap.name:48} {result[cap.name]}")
        except Exception as e:
            print(f"  {cap.name:48} ERROR: {e}")
            result[cap.name] = None
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="capsule-hash regression gate (#1258)")
    ap.add_argument("mode", choices=["check", "update", "list"])
    ap.add_argument("corpus", type=Path, help="directory of .prgcap capsules (local, gitignored)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline JSON (default: <corpus>/regress-baseline.json)")
    ap.add_argument("--replay", default=DEFAULT_REPLAY, help="gpu_replay binary (or $PROSPER_GPU_REPLAY)")
    ap.add_argument("--timeout", type=int, default=600, help="per-capsule replay timeout seconds")
    ap.add_argument("--strict", action="store_true",
                    help="check: also fail when a baselined capsule is MISSING from the corpus "
                         "(so a locally-deleted capsule can't silently shrink coverage)")
    args = ap.parse_args()

    if not args.corpus.is_dir():
        sys.exit(f"regress: corpus dir not found: {args.corpus}")
    # One clear message if the replay binary is wrong, instead of N identical FileNotFoundErrors.
    if shutil.which(args.replay) is None and not Path(args.replay).is_file():
        sys.exit(f"regress: gpu_replay not found: {args.replay} "
                 f"(set --replay or $PROSPER_GPU_REPLAY to the built binary)")
    baseline_path = args.baseline or (args.corpus / "regress-baseline.json")

    print(f"[regress] {args.mode} corpus={args.corpus} replay={args.replay}")
    current = corpus_hashes(args.replay, args.corpus, args.timeout)

    if args.mode == "list":
        return 1 if any(v is None for v in current.values()) else 0

    if args.mode == "update":
        # Refuse to bake an ERROR (None) hash into the baseline — that would mask a broken capsule as "expected".
        if any(v is None for v in current.values()):
            print("[regress] FAILED: some capsules did not replay; baseline NOT written")
            return 1
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"[regress] wrote baseline for {len(current)} capsule(s) -> {baseline_path}")
        return 0

    # check
    if not baseline_path.exists():
        sys.exit(f"regress: no baseline at {baseline_path} (run `regress.py update {args.corpus}` first)")
    try:
        baseline = json.loads(baseline_path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"regress: baseline {baseline_path} is corrupt JSON: {e}")
    if not isinstance(baseline, dict):
        sys.exit(f"regress: baseline {baseline_path} is not a name->hash object")
    regressions, errors, new, missing = classify_check(current, baseline)

    for name, exp, got in regressions:
        print(f"[regress] CHANGED  {name}: baseline {exp} -> now {got}")
    for name in errors:
        print(f"[regress] ERROR    {name}: did not replay")
    for name in new:
        print(f"[regress] new      {name}: not in baseline (run update to record)")
    for name in missing:
        print(f"[regress] missing  {name}: in baseline but not in corpus"
              + (" [FAIL under --strict]" if args.strict else ""))

    strict_missing = missing if args.strict else []
    if regressions or errors or strict_missing:
        print(f"[regress] FAIL: {len(regressions)} changed, {len(errors)} errored"
              + (f", {len(strict_missing)} missing (--strict)" if strict_missing else ""))
        return 1
    print(f"[regress] OK: {len(current)} capsule(s) match baseline"
          + (f" ({len(new)} new, {len(missing)} missing — informational)" if (new or missing) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
