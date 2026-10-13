#!/usr/bin/env python3
"""Recompile a directory of raw RDNA2 shader blobs and record or compare what came out.

ADR 0006 (spec GPU-1) migrates the shader recompiler to a pure function one entry point at a time,
and checks each step by recompiling a corpus before and after and comparing the output. This is that
check for raw blobs: `shader_inspect --stage` is run on every `cs_*`/`ps_*`/`vs_*` file, and the
verdict (stage-recompile status, SPIR-V word count, each recompile-reject line) is recorded. For
vertex and fragment blobs it also runs `--synthetic-table --spirv-out` and records the SPIR-V
sha256, so those are compared byte for byte. Compute blobs carry no descriptors in a raw dump, so
shader_inspect reports a verdict but writes no SPIR-V; that half compares verdicts only.

  recompile_corpus.py record <corpus-dir> --inspect <shader_inspect> --out BASELINE.json
  recompile_corpus.py check  <corpus-dir> --inspect <shader_inspect> --baseline BASELINE.json
  recompile_corpus.py dumps  <before-dir> <after-dir>

`dumps` compares two PROSPER_SHADER_DUMP_SUCCESS directories from live runs. Each file name is
`success_<stage>_at_<addr>_<spirv-hash>_<raw-hash>[_main_...]`, so the SPIR-V hash is in the name:
for every (stage, address, raw-hash) present in both, the SPIR-V hash must be equal. Shaders present
in only one run are reported but are not a failure, since a run may not reach the same programs.

The corpus is local (Sony or title bytes are never committed); only the hash file may be shared.
`check` exits 1 on any difference, 2 when it could not run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

STAGES = {"cs": "compute", "ps": "fragment", "vs": "vertex"}
VERDICT_PREFIXES = ("stage-recompile stage=", "[recompile-reject]")


def stage_of(path: Path):
    return STAGES.get(path.name.split("_", 1)[0])


def verdict_lines(text: str) -> list[str]:
    """Lines that state what the recompiler decided, in order. Addresses are not normalised: the
    corpus is fixed bytes, so a changed line is a changed answer."""
    return [ln.strip() for ln in text.splitlines() if ln.startswith(VERDICT_PREFIXES)]


class InspectorError(Exception):
    """shader_inspect could not run, as opposed to running and refusing a shader."""


def run(cmd: list[str], env: dict[str, str]) -> str:
    """Run shader_inspect and return its output. A refusal is an answer (it prints a
    `stage-recompile` line); an inspector that crashes, times out or prints no such line is not."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as err:
        raise InspectorError(f"{cmd[0]} did not run: {err}") from err
    text = done.stdout + done.stderr
    if not any(ln.startswith("stage-recompile stage=") for ln in text.splitlines()):
        raise InspectorError(
            f"no stage-recompile verdict from {Path(cmd[1]).name} (exit {done.returncode})"
        )
    return text


def recompile(inspect: str, blob: Path, stage: str) -> dict:
    # PROSPER_DBG is fixed on so the reject lines exist; it only logs, so it never varies the output.
    env = {**os.environ, "PROSPER_DBG": "1"}
    entry = {
        "stage": stage,
        "verdict": verdict_lines(run([inspect, str(blob), "--stage", stage], env)),
    }
    if stage in ("vertex", "fragment"):
        with tempfile.TemporaryDirectory() as tmp:
            spv = Path(tmp) / "out.spv"
            text = run(
                [
                    inspect,
                    str(blob),
                    "--stage",
                    stage,
                    "--synthetic-table",
                    "--spirv-out",
                    str(spv),
                ],
                env,
            )
            entry["synthetic_verdict"] = verdict_lines(text)
            entry["spirv_sha256"] = (
                hashlib.sha256(spv.read_bytes()).hexdigest() if spv.exists() else None
            )
    return entry


def collect(corpus: Path, inspect: str) -> dict:
    blobs = sorted(p for p in corpus.rglob("*.bin") if stage_of(p))
    if not blobs:
        raise SystemExit(f"no cs_/ps_/vs_ blobs under {corpus}")
    # Keyed by the path under the corpus: two subdirectories may hold the same basename.
    return {p.relative_to(corpus).as_posix(): recompile(inspect, p, stage_of(p)) for p in blobs}


def diff(base: dict, now: dict) -> list[str]:
    out = [f"missing from run: {n}" for n in sorted(base.keys() - now.keys())]
    out += [f"not in baseline: {n}" for n in sorted(now.keys() - base.keys())]
    out += [f"changed: {n}" for n in sorted(base.keys() & now.keys()) if base[n] != now[n]]
    return out


def dump_keys(directory: Path) -> dict:
    """{(stage, address, raw-hash + continuation): {spirv-hash}} from success_*.spv names. One
    program can compile in several contexts, so a key holds every SPIR-V hash seen for it."""
    out: dict = {}
    for f in directory.glob("success_*.spv"):
        parts = f.stem.split("_")
        if len(parts) < 6 or parts[2] != "at":
            continue
        out.setdefault((parts[1], parts[3], "_".join(parts[5:])), set()).add(parts[4])
    return out


def compare_dumps(before: dict, after: dict) -> tuple[list[str], int]:
    """(differences, shared count): a shared key is a difference when a SPIR-V hash seen before is
    gone after. A hash only in the after run is a context the run reached and the other did not
    (measured: a second boot of one title reached one extra context), so it is not a difference."""
    shared = before.keys() & after.keys()
    bad = [
        f"spirv differs: {k[0]} at {k[1]} raw {k[2]}: {sorted(before[k] - after[k])} gone, after has {sorted(after[k])}"
        for k in sorted(shared)
        if before[k] - after[k]
    ]
    return bad, len(shared)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("record", "check", "dumps"))
    ap.add_argument("corpus", type=Path)
    ap.add_argument("after", nargs="?", type=Path, help="dumps: the second directory")
    ap.add_argument("--inspect", help="path to the shader_inspect binary (record/check)")
    ap.add_argument("--out", type=Path, help="record: baseline file to write")
    ap.add_argument("--baseline", type=Path, help="check: baseline file to compare against")
    args = ap.parse_args(argv)
    if args.mode == "dumps":
        if not args.after or not args.corpus.is_dir() or not args.after.is_dir():
            print("dumps needs two existing directories", file=sys.stderr)
            return 2
        before, after = dump_keys(args.corpus), dump_keys(args.after)
        bad, shared = compare_dumps(before, after)
        for line in bad:
            print(line)
        print(
            f"{shared} shared programs, {len(bad)} differ, "
            f"{len(before.keys() - after.keys())} only before, {len(after.keys() - before.keys())} only after"
        )
        if shared == 0:
            print("no program in common: nothing was compared", file=sys.stderr)
            return 2
        return 1 if bad else 0
    if not args.inspect or not args.corpus.is_dir() or not Path(args.inspect).exists():
        print("corpus directory or shader_inspect binary not found", file=sys.stderr)
        return 2
    # Absolute: Windows CreateProcess does not resolve a relative, forward-slashed executable path.
    try:
        results = collect(args.corpus.resolve(), str(Path(args.inspect).resolve()))
    except InspectorError as err:
        print(err, file=sys.stderr)
        return 2
    if args.mode == "record":
        if not args.out:
            print("record needs --out", file=sys.stderr)
            return 2
        args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
        print(f"recorded {len(results)} blobs -> {args.out}")
        return 0
    if not args.baseline or not args.baseline.exists():
        print("check needs an existing --baseline", file=sys.stderr)
        return 2
    problems = diff(json.loads(args.baseline.read_text()), results)
    for line in problems:
        print(line)
    print(f"{len(results)} blobs, {len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
