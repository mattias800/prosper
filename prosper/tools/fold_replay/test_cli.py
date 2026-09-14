"""Execute the production capture hook and CLI with independent synthetic live memory."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

fixture, replay = sys.argv[1:]
env = {k: v for k, v in os.environ.items() if not k.startswith("PROSPER_")}


def run(*args, extra=None, ok=True):
    p = subprocess.run(args, env=env | (extra or {}), text=True,
                       capture_output=True, timeout=60)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stdout, p.stderr)
    return p.stdout + p.stderr


with tempfile.TemporaryDirectory(prefix="prosper-fold-cli-") as temp:
    root = Path(temp)
    file = root / "fixture.prfold"
    run(fixture, "--emit", str(file))
    report = run(replay, str(file), "--iterations", "3")
    assert "mode=compare-all-outputs" in report and "changed_fetches=0" in report, report
    assert "iterations=3" in report and "evaluated_instructions=6" in report, report
    assert "control_plan_builds=0" in report.split("repeat_mode=warm", 1)[1], report
    mutated = run(replay, str(file), "--iterations", "0", "--mutate-word", "1=0x77770004")
    assert "mode=explicit-mutation" in mutated and "changed_fetches=1" in mutated, mutated
    for options in [("--iterations", "1000001"), ("--iterations", "-1"),
                    ("--iterations", "3junk"), ("--mutate-word", "0=4"),
                    ("--mutate-word", "999=4"), ("--unknown", "2"), ("--iterations",)]:
        assert "REFUSED" in run(replay, str(file), *options, ok=False)
    broken = root / "broken.prfold"
    broken.write_bytes(file.read_bytes()[:-1])
    assert "REFUSED" in run(replay, str(broken), ok=False)

    capture_root = root / "live"
    capture_env = {"PROSPER_FOLD_CAPTURE_DIR": str(capture_root), "PROSPER_FOLD_CAPTURE_LIMIT": "1"}
    for _ in range(2):
        report = run(fixture, "--live", "normal", extra=capture_env)
        assert "[fold-capture] complete" in report, report
    files = list(capture_root.glob("session-*/*.prfold"))
    assert len(files) == 2 and files[0].parent != files[1].parent, files
    for path in files:
        run(replay, str(path), "--iterations", "3")

    report = run(fixture, "--live", "occupied", extra=capture_env)
    assert "refused call=0: capture requires an empty initial SRT vector" in report, report
    # A file where the capture root directory must be makes I/O fail after evaluation.
    report = run(fixture, "--live", "normal", extra=capture_env | {"PROSPER_FOLD_CAPTURE_DIR": str(file)})
    assert "[fold-capture] unusable call=0" in report, report
    report = run(fixture, "--live", "normal", extra=capture_env | {"PROSPER_FOLD_CAPTURE_SKIP": "1"})
    assert "[fold-capture] complete" not in report, report
    for name, value in [("PROSPER_FOLD_CAPTURE_LIMIT", "257"), ("PROSPER_FOLD_CAPTURE_SKIP", "-1"),
                        ("PROSPER_FOLD_CAPTURE_LIMIT", "1junk")]:
        report = run(fixture, "--live", "normal", extra=capture_env | {name: value})
        assert "invalid capture settings; disabled" in report, report
    assert len(list(capture_root.glob("session-*/*.prfold"))) == 2

print("fold replay CLI and production hook: passed")
