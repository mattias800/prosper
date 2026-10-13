#!/usr/bin/env python3
"""Replay a directory of recorded fragment compiler cases and compare two builds' answers.

ADR 0006 (spec GPU-1) moves the recompiler's inputs into a request one entry point at a time, and
checks each step by recompiling a corpus before and after and comparing the output. For the fragment
entry point the corpus is a directory of `.prfc` cases (`PROSPER_FRAGMENT_COMPILE_CASE_DIR`, see
README.md). Each case carries every compiler input and the identity of the build that recorded it, so
`--baseline` only matches on that build, while `--candidate` replays the same inputs on any build.

  corpus.py run <case-dir> --replay <fragment_compile_replay> --mode baseline  --out BASE.json
  corpus.py run <case-dir> --replay <fragment_compile_replay> --mode candidate --out CAND.json
  corpus.py compare BASE.json CAND.json

Run the first with the recording build and the second with the build under test. A case's answer is
its SPIR-V sha256 (or none for a refusal) and the replay's verdict with the mode stripped; the exit
code is not part of it, since a baseline that reproduces a recorded refusal exits 0 and a candidate
that still refuses exits 3. The corpus is local (cases hold guest code and are never committed); only
the result files may be shared. `compare` exits 1 on any difference. Exit 2 means the replay could
not run: the tool exited with anything but 0 (reproduced) or 3 (a complete candidate still refused),
printed no status line, or reported a produced module but wrote none.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

VERDICT_EXITS = frozenset({0, 3})
MODE_PREFIXES = ("BASELINE MATCH ", "CANDIDATE ")


class ReplayError(Exception):
    """The replay tool could not give an answer for a case, as opposed to answering 'refused'."""


def verdict_of(output: str) -> str:
    """The replay's answer: its status line with the mode prefix removed, so the two modes compare,
    plus the `actual_refusal=` line a refusal prints (it names the refusing pc, so a refusal that
    moves is a change)."""
    answer = []
    for raw in output.splitlines():
        line = raw.strip()
        for prefix in MODE_PREFIXES:
            if line.startswith(prefix):
                answer.append(line[len(prefix) :])
        if line.startswith("actual_refusal="):
            answer.append(line)
    if not answer:
        raise ReplayError("replay printed no BASELINE/CANDIDATE status line")
    return "\n".join(answer)


def run_case(replay: str, mode: str, case: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "out.spv"
        cmd = [replay, f"--{mode}", str(case), "--output", str(module)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as err:
            raise ReplayError(f"{case.name}: replay did not run: {err}") from err
        if done.returncode not in VERDICT_EXITS:
            raise ReplayError(f"{case.name}: replay exited {done.returncode}, not a verdict")
        verdict = verdict_of(done.stdout + done.stderr)
        if "PRODUCED" in verdict.split("\n")[0] and not module.exists():
            raise ReplayError(f"{case.name}: replay reported a module but wrote none")
        sha = hashlib.sha256(module.read_bytes()).hexdigest() if module.exists() else None
        return {"spirv_sha256": sha, "verdict": verdict}


def collect(cases: Path, replay: str, mode: str) -> dict:
    files = sorted(cases.glob("*.prfc"))
    if not files:
        raise ReplayError(f"no .prfc cases under {cases}")
    return {f.name: run_case(replay, mode, f) for f in files}


def diff(base: dict, cand: dict) -> list[str]:
    out = [f"missing from candidate: {n}" for n in sorted(base.keys() - cand.keys())]
    out += [f"not in baseline: {n}" for n in sorted(cand.keys() - base.keys())]
    out += [f"changed: {n}" for n in sorted(base.keys() & cand.keys()) if base[n] != cand[n]]
    return out


def json_path(path: Path) -> Path | None:
    """The resolved path of a result-file argument, or None unless it names a `.json` file."""
    resolved = path.resolve()
    return resolved if resolved.suffix == ".json" else None


def run_run(args) -> int:
    out = json_path(args.out) if args.out else None
    if out is None or not args.cases.is_dir() or not args.replay:
        print(
            "run needs an existing case directory, --replay and --out <file>.json", file=sys.stderr
        )
        return 2
    try:
        results = collect(args.cases.resolve(), str(Path(args.replay).resolve()), args.mode)
        out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    except (ReplayError, OSError) as err:
        print(err, file=sys.stderr)
        return 2
    print(f"{len(results)} cases -> {out}")
    return 0


def run_compare(args) -> int:
    try:
        base = json.loads(args.cases.read_text())
        cand = json.loads(args.other.read_text())
    except (OSError, ValueError, AttributeError) as err:
        print(f"could not read results: {err}", file=sys.stderr)
        return 2
    if not isinstance(base, dict) or not isinstance(cand, dict) or not (base.keys() & cand.keys()):
        print("no case in common: nothing was compared", file=sys.stderr)
        return 2
    problems = diff(base, cand)
    for line in problems:
        print(line)
    print(f"{len(base.keys() & cand.keys())} shared cases, {len(problems)} differences")
    return 1 if problems else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "compare"))
    ap.add_argument("cases", type=Path, help="run: case directory; compare: BASE.json")
    ap.add_argument("other", nargs="?", type=Path, help="compare: CAND.json")
    ap.add_argument("--replay", help="path to fragment_compile_replay (run)")
    ap.add_argument("--mode", choices=("baseline", "candidate"), default="candidate")
    ap.add_argument("--out", type=Path, help="run: result file to write")
    args = ap.parse_args(argv)
    return run_run(args) if args.command == "run" else run_compare(args)


if __name__ == "__main__":
    sys.exit(main())
