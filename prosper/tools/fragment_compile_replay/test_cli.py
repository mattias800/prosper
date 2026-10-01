"""Real CPU CLI and producing-hook controls on project-owned instruction words."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

fixture, replay = sys.argv[1:]
validator = shutil.which("spirv-val")
assert validator, "strict fragment compile CLI verification requires spirv-val"
env = {k: v for k, v in os.environ.items() if not k.startswith("PROSPER_")}
scratch = Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR", Path.cwd() / "test-scratch"))
scratch.mkdir(parents=True, exist_ok=True)
checks = 0


def run(*args, code=0, extra=None):
    global checks
    p = subprocess.run(args, env=env | (extra or {}), capture_output=True,
                       text=True, timeout=60)
    checks += 1
    assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
    return p.stdout + p.stderr


with tempfile.TemporaryDirectory(prefix="fragment-case-cli-", dir=scratch) as temp:
    root = Path(temp)
    run(fixture, "--emit", str(root))
    linked = run(replay, "--compiler-identity").strip()
    assert len(linked.split(":")) == 2 and "unknown" not in linked, linked
    for width in (32, 64):
        case = root / f"wave{width}.prfc"
        baseline, candidate = root / f"baseline{width}.spv", root / f"candidate{width}.spv"
        report = run(replay, "--baseline", str(case), "--output", str(baseline),
                     extra={"PROSPER_FS_TAP": "0:0", "PROSPER_MIMG_SOFT": "1"})
        assert "BASELINE MATCH PRODUCED" in report and "input=COMPLETE" in report, report
        assert f"producing_compiler={linked}" in report, report
        assert "CANDIDATE PRODUCED" in run(replay, "--candidate", str(case), "--output", str(candidate))
        assert baseline.read_bytes() == candidate.read_bytes()
        assert baseline.read_bytes()[:4] == b"\x03\x02\x23\x07"
        run(validator, "--target-env", "vulkan1.1", str(baseline))
        run(validator, "--target-env", "vulkan1.1", str(candidate))
    assert "BASELINE MATCH REFUSED" in run(replay, "--baseline", str(root / "refused.prfc"))
    saved = root / "preserved.spv"
    saved.write_bytes(b"previous-output")
    assert "CANDIDATE REFUSED" in run(replay, "--candidate", str(root / "refused.prfc"),
                                     "--output", str(saved), code=3)
    assert saved.read_bytes() == b"previous-output"
    assert "INCOMPLETE" in run(replay, "--inspect-only", str(root / "incomplete.prfc"))
    assert "INCOMPLETE" in run(replay, "--candidate", str(root / "incomplete.prfc"), code=2)
    assert "identity mismatch" in run(replay, "--baseline", str(root / "different.prfc"), code=2)
    assert "CANDIDATE PRODUCED" in run(replay, "--candidate", str(root / "different.prfc"))
    assert "SOURCE differs" in run(replay, "--baseline", str(root / "wrong-source.prfc"),
                                    "--output", str(saved), code=2)
    assert saved.read_bytes() == b"previous-output"
    malformed = root / "malformed.prfc"
    malformed.write_bytes((root / "wave64.prfc").read_bytes()[:-1])
    assert "REPLAY REFUSED" in run(replay, "--candidate", str(malformed), code=2)
    for args in [(), ("--bogus", str(root / "wave64.prfc")),
                 ("--baseline", str(root / "wave64.prfc"), "--candidate"),
                 ("--inspect-only", str(root / "wave64.prfc"), "--output", str(saved)),
                 ("--candidate", str(root / "wave64.prfc"), "--wrong", str(saved))]:
        assert "REPLAY REFUSED" in run(replay, *args, code=2)
    assert saved.read_bytes() == b"previous-output"
    assert "CPU-only" in run(replay, "--help")
    invalid_spv = root / "invalid.spv"
    invalid_spv.write_bytes(b"not-a-SPIR-V-module")
    run(validator, "--target-env", "vulkan1.1", str(invalid_spv), code=1)

print(f"fragment compile replay CLI: {checks} real invocations passed, four strict modules and validator rejection; no GPU/game")
