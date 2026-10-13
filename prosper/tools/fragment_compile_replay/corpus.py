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
printed no valid status/refusal, or disagreed with its module size. Result records include the input
case's SHA256; changed inputs are differences even when their output is identical. Malformed result
files are unevaluated, never an equal comparison. Regenerate both files when upgrading the driver.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

VERDICT_EXITS = frozenset({0, 3})
MODE_PREFIXES = ("BASELINE MATCH ", "CANDIDATE ")
STATUS = re.compile(r"(PRODUCED|REFUSED) source_words=(0|[1-9][0-9]*)")
SHA256 = re.compile(r"[0-9a-f]{64}")


class ReplayError(Exception):
    """The replay tool could not give an answer for a case, as opposed to answering 'refused'."""


def checked_verdict(verdict: str) -> tuple[str, int]:
    """Require the replay CLI's one status and optional actual-refusal diagnostic."""
    lines = verdict.splitlines()
    match = STATUS.fullmatch(lines[0]) if lines else None
    if (
        match is None
        or len(lines) > 2
        or (
            len(lines) == 2
            and (not lines[1].startswith("actual_refusal=") or lines[1] == "actual_refusal=")
        )
    ):
        raise ReplayError("invalid or duplicate replay status/refusal line")
    kind = match[1]
    try:
        words = int(match[2])
    except ValueError as err:
        raise ReplayError("invalid source word count") from err
    expected_lines = 1 if kind == "PRODUCED" else 2
    if (kind == "PRODUCED") != (words > 0) or len(lines) != expected_lines:
        raise ReplayError("replay status contradicts source count or refusal")
    return kind, words


def verdict_of(output: str, mode: str | None = None) -> str:
    """The replay's answer: its status line with the mode prefix removed, so the two modes compare,
    plus the `actual_refusal=` line a refusal prints (it names the refusing pc, so a refusal that
    moves is a change)."""
    statuses, refusals = [], []
    for raw in output.splitlines():
        line = raw.strip()
        for prefix in MODE_PREFIXES:
            if line.startswith(prefix):
                expected_prefix = "BASELINE MATCH " if mode == "baseline" else "CANDIDATE "
                if mode is not None and prefix != expected_prefix:
                    raise ReplayError("replay status has the wrong mode")
                statuses.append(line[len(prefix) :])
        if line.startswith("actual_refusal="):
            refusals.append(line)
    if len(statuses) != 1:
        raise ReplayError("replay must print exactly one BASELINE/CANDIDATE status line")
    verdict = "\n".join(statuses + refusals)
    checked_verdict(verdict)
    return verdict


def case_digest(case: Path) -> str:
    with case.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_case(replay: str, mode: str, case: Path) -> dict:
    input_sha = case_digest(case)
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "out.spv"
        cmd = [replay, f"--{mode}", str(case), "--output", str(module)]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as err:
            raise ReplayError(f"{case.name}: replay did not run: {err}") from err
        if done.returncode not in VERDICT_EXITS:
            raise ReplayError(f"{case.name}: replay exited {done.returncode}, not a verdict")
        verdict = verdict_of(done.stdout + "\n" + done.stderr, mode)
        kind, words = checked_verdict(verdict)
        expected_exit = 3 if mode == "candidate" and kind == "REFUSED" else 0
        if done.returncode != expected_exit:
            raise ReplayError(f"{case.name}: replay exit contradicts its status")
        if kind == "PRODUCED" and not module.exists():
            raise ReplayError(f"{case.name}: replay reported a module but wrote none")
        if kind == "REFUSED" and module.exists():
            raise ReplayError(f"{case.name}: refused replay wrote a module")
        if kind == "PRODUCED" and module.stat().st_size != words * 4:
            raise ReplayError(f"{case.name}: module size disagrees with source_words")
        if case_digest(case) != input_sha:
            raise ReplayError(f"{case.name}: case changed during replay")
        sha = hashlib.sha256(module.read_bytes()).hexdigest() if module.exists() else None
        return {"case_sha256": input_sha, "spirv_sha256": sha, "verdict": verdict}


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
    try:
        for results in (base, cand):
            for name, record in results.items():
                if (
                    not isinstance(record, dict)
                    or set(record) != {"case_sha256", "spirv_sha256", "verdict"}
                    or not isinstance(record["case_sha256"], str)
                    or SHA256.fullmatch(record["case_sha256"]) is None
                    or not isinstance(record["verdict"], str)
                ):
                    raise ReplayError(f"{name}: invalid result record")
                kind, _ = checked_verdict(record["verdict"])
                sha = record["spirv_sha256"]
                if (kind == "REFUSED" and sha is not None) or (
                    kind == "PRODUCED"
                    and (not isinstance(sha, str) or SHA256.fullmatch(sha) is None)
                ):
                    raise ReplayError(f"{name}: module hash contradicts status")
    except ReplayError as err:
        print(f"could not read results: {err}", file=sys.stderr)
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
