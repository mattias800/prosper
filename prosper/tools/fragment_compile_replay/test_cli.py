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


def rechecksum(blob):
    value = 14695981039346656037
    for byte in blob[:-8]:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    blob[-8:] = value.to_bytes(8, "little")
    return blob


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
    mode_outputs = []
    for name, mode in (("flush", 0), ("preserve", 16)):
        output = root / f"{name}.spv"
        report = run(replay, "--baseline", str(root / f"{name}.prfc"), "--output", str(output))
        assert f"guest_float_mode={mode}" in report and "BASELINE MATCH PRODUCED" in report
        run(validator, "--target-env", "vulkan1.1", str(output))
        mode_outputs.append(output.read_bytes())
    assert mode_outputs[0] != mode_outputs[1], "captured guest mode must reach live compiler relation"
    for profile, value in (("unknown", 0), ("implicit", 1), ("explicit-nonfinite32", 2)):
        case = root / f"transport-{profile}.prfc"
        wire = case.read_bytes()
        assert wire[8:12] == (5).to_bytes(4, "little") and wire[-21] == value and wire[-20:-12] == bytes(8) and wire[-12:-8] == bytes(4), "exact official profile/launch tails followed by nested5 zero count"
        source = root / f"transport-{profile}.spv"
        candidate = root / f"transport-{profile}-candidate.spv"
        report = run(replay, "--baseline", str(case), "--output", str(source))
        assert "input=COMPLETE" in report and f"host_float_transport={profile}" in report and "BASELINE MATCH PRODUCED" in report
        report = run(replay, "--candidate", str(case), "--output", str(candidate))
        assert f"host_float_transport={profile}" in report and "CANDIDATE PRODUCED" in report
        assert source.read_bytes() == candidate.read_bytes()
        run(validator, "--target-env", "vulkan1.1", str(source))
        run(validator, "--target-env", "vulkan1.1", str(candidate))
    for ieee in (0, 1):
        for dx10 in (0, 1):
            case = root / f"flags-i{ieee}-d{dx10}.prfc"
            wire = case.read_bytes()
            assert wire[-20:-17] == bytes((1, ieee, dx10)) and wire[-17:-12] == bytes(5)
            for mode in ("--baseline", "--candidate"):
                report = run(replay, mode, str(case))
                assert "input=COMPLETE" in report and "guest_float_mode=unknown" in report
                assert f"guest_ieee_mode={ieee}" in report and f"guest_dx10_clamp={dx10}" in report
                assert "PRODUCED" in report
                assert "rsrc1_ps_evidence=unknown" in report
    for value in (0,1<<29,(1<<29)|(1<<22)):
        case=root/f"raw-launch-{value}.prfc"
        output=root/f"raw-launch-{value}.spv"
        wire=case.read_bytes()
        assert wire[-20:-17]==bytes(3) and wire[-17]==1
        assert int.from_bytes(wire[-16:-12],"little")==value
        for mode in ("--baseline","--candidate"):
            report=run(replay,mode,str(case),*( ("--output",str(output)) if mode=="--baseline" else () ))
            assert "input=COMPLETE" in report and "guest_float_mode=unknown" in report
            assert "guest_ieee_mode=unknown" in report and "guest_dx10_clamp=unknown" in report
            assert f"rsrc1_ps_evidence={value}" in report and "PRODUCED" in report
        run(validator,"--target-env","vulkan1.1",str(output))
    refusal = run(replay, "--baseline", str(root / "refused.prfc"))
    assert "BASELINE MATCH REFUSED" in refusal and "actual_refusal=" in refusal and "no supported export" in refusal
    saved = root / "preserved.spv"
    saved.write_bytes(b"previous-output")
    current = bytearray((root / "transport-unknown.prfc").read_bytes())
    official4 = bytearray(current)
    del official4[-12:-8]
    official4[8:12] = (4).to_bytes(4,"little")
    official_case = root / "official-schema4.prfc"
    official_case.write_bytes(rechecksum(official4))
    report = run(replay, "--baseline", str(official_case))
    assert "input=COMPLETE" in report and "BASELINE MATCH PRODUCED" in report
    legacy3 = bytearray(official4)
    del legacy3[-16:-8]  # remove flags and raw evidence to recover the actual schema-3 prefix
    legacy3[8:12] = (3).to_bytes(4, "little")
    legacy3_case = root / "legacy-schema3.prfc"
    legacy3_case.write_bytes(rechecksum(legacy3))
    report = run(replay, "--inspect-only", str(legacy3_case))
    assert "input=INCOMPLETE" in report and "fragment-float-flags-unavailable" in report
    assert "guest_ieee_mode=unknown" in report and "guest_dx10_clamp=unknown" in report
    for mode in ("--baseline", "--candidate"):
        assert "INCOMPLETE" in run(replay, mode, str(legacy3_case), "--output", str(saved), code=2)
        assert saved.read_bytes() == b"previous-output"
    legacy = bytearray(legacy3)
    del legacy[-9]  # remove transport only, retaining the actual schema2 marker tail
    legacy[8:12] = (2).to_bytes(4, "little")
    legacy_case = root / "legacy-schema2.prfc"
    legacy_case.write_bytes(rechecksum(legacy))
    report = run(replay, "--inspect-only", str(legacy_case))
    assert "input=INCOMPLETE" in report and "fragment-transport-config-unavailable" in report and "host_float_transport=unknown" in report
    for mode in ("--baseline", "--candidate"):
        assert "INCOMPLETE" in run(replay, mode, str(legacy_case), "--output", str(saved), code=2)
        assert saved.read_bytes() == b"previous-output"
    assert legacy[-12:-8] == bytes(4), "resource-free schema2 has exact zero marker count"
    del legacy[-12:-8]
    legacy[8:12] = (1).to_bytes(4, "little")
    legacy_case = root / "legacy-schema1.prfc"
    legacy_case.write_bytes(rechecksum(legacy))
    report = run(replay, "--inspect-only", str(legacy_case))
    assert "input=INCOMPLETE" in report and "fragment-transport-config-unavailable" in report and "host_float_transport=unknown" in report
    for mode in ("--baseline", "--candidate"):
        assert "INCOMPLETE" in run(replay, mode, str(legacy_case), "--output", str(saved), code=2)
        assert saved.read_bytes() == b"previous-output"
    malformed_profiles = []
    relabeled = bytearray(current)
    relabeled[8:12] = (2).to_bytes(4, "little")
    malformed_profiles.append((relabeled, "trailing data"))
    invalid_profile = bytearray(current)
    invalid_profile[-21] = 255
    malformed_profiles.append((invalid_profile, "noncanonical float transport"))
    truncated_profile = bytearray(current)
    del truncated_profile[-21]
    malformed_profiles.append((truncated_profile, "truncated"))
    for offset in (-20, -19, -18, -17):
        invalid_flag = bytearray(current)
        invalid_flag[offset] = 2
        malformed_profiles.append((invalid_flag, "boolean"))
    invalid_flag = bytearray(current)
    invalid_flag[-19] = 1  # unavailable payload must be all clear, not fabricated known IEEE
    malformed_profiles.append((invalid_flag, "noncanonical float flags"))
    invalid_raw=bytearray(current)
    invalid_raw[-16:-12]=(1).to_bytes(4,"little")
    malformed_profiles.append((invalid_raw,"noncanonical RSRC1_PS"))
    for index, (wire, reason) in enumerate(malformed_profiles):
        path = root / f"malformed-profile-{index}.prfc"
        path.write_bytes(rechecksum(wire))
        assert reason in run(replay, "--candidate", str(path), "--output", str(saved), code=2)
        assert saved.read_bytes() == b"previous-output"
    refusal = run(replay, "--candidate", str(root / "refused.prfc"), "--output", str(saved), code=3)
    assert "CANDIDATE REFUSED" in refusal and "actual_refusal=" in refusal and "no supported export" in refusal
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

print(f"fragment compile replay CLI: {checks} real invocations passed, fifteen strict modules and validator rejection; no GPU/game")
