"""Tests for corpus.py: what counts as an answer, and what counts as a changed one."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import corpus as cp  # noqa: E402


class _Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def fake_replay(
    monkeypatch, *, code=0, stdout="BASELINE MATCH PRODUCED source_words=9\n", module=b"M"
):
    """Replace subprocess.run: write `module` to the --output path (when not None) and answer."""

    def run(cmd, **_kw):
        if module is not None:
            Path(cmd[-1]).write_bytes(module)
        return _Done(stdout, code)

    monkeypatch.setattr(cp.subprocess, "run", run)


def make_cases(directory: Path, *names: str) -> Path:
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\x00")
    return directory


def test_verdict_strips_the_mode_prefix_so_the_two_modes_compare():
    assert cp.verdict_of("noise\nBASELINE MATCH PRODUCED source_words=9\n") == (
        "PRODUCED source_words=9"
    )
    assert cp.verdict_of("CANDIDATE PRODUCED source_words=9\n") == "PRODUCED source_words=9"
    assert cp.verdict_of("CANDIDATE REFUSED reason=x\n") == "REFUSED reason=x"


def test_silent_replay_is_not_a_verdict():
    with pytest.raises(cp.ReplayError):
        cp.verdict_of("\n  \n")


def test_a_produced_module_is_hashed(tmp_path, monkeypatch):
    fake_replay(monkeypatch, module=b"abc")
    result = cp.run_case("replay", "candidate", tmp_path / "f_1.prfc")
    assert result["spirv_sha256"] is not None
    assert result["verdict"] == "PRODUCED source_words=9"


def test_a_complete_candidate_that_still_refuses_is_an_answer(tmp_path, monkeypatch):
    fake_replay(monkeypatch, code=3, stdout="CANDIDATE REFUSED reason=x\n", module=None)
    result = cp.run_case("replay", "candidate", tmp_path / "f_1.prfc")
    assert result == {"spirv_sha256": None, "verdict": "REFUSED reason=x"}


@pytest.mark.parametrize("code", [1, 2, 139, -11])
def test_any_other_exit_is_a_replay_that_could_not_run(tmp_path, monkeypatch, code):
    fake_replay(monkeypatch, code=code)
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", "baseline", tmp_path / "f_1.prfc")


def test_success_without_a_module_is_not_recorded(tmp_path, monkeypatch):
    fake_replay(monkeypatch, module=None)
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", "baseline", tmp_path / "f_1.prfc")


def test_diff_reports_changed_missing_and_new_cases():
    base = {"a": {"v": 1}, "b": {"v": 1}, "c": {"v": 1}}
    cand = {"a": {"v": 1}, "b": {"v": 2}, "d": {"v": 1}}
    assert cp.diff(base, cand) == [
        "missing from candidate: c",
        "not in baseline: d",
        "changed: b",
    ]


def test_run_then_compare_round_trip(tmp_path, monkeypatch, capsys):
    cases = make_cases(tmp_path / "cases", "f_1.prfc", "f_2.prfc")
    fake_replay(monkeypatch, module=b"same")
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    replay = tmp_path / "replay.exe"
    replay.write_bytes(b"")
    assert (
        cp.main(
            ["run", str(cases), "--replay", str(replay), "--mode", "baseline", "--out", str(base)]
        )
        == 0
    )
    assert cp.main(["run", str(cases), "--replay", str(replay), "--out", str(cand)]) == 0
    assert sorted(json.loads(base.read_text())) == ["f_1.prfc", "f_2.prfc"]
    assert cp.main(["compare", str(base), str(cand)]) == 0
    fake_replay(monkeypatch, module=b"different")
    assert cp.main(["run", str(cases), "--replay", str(replay), "--out", str(cand)]) == 0
    assert cp.main(["compare", str(base), str(cand)]) == 1
    assert "changed: f_1.prfc" in capsys.readouterr().out


def test_compare_with_nothing_in_common_is_unevaluated(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"x": {}}))
    b.write_text(json.dumps({"y": {}}))
    assert cp.main(["compare", str(a), str(b)]) == 2


def test_run_without_cases_or_a_json_output_is_unevaluated(tmp_path, monkeypatch):
    fake_replay(monkeypatch)
    empty = tmp_path / "empty"
    empty.mkdir()
    replay = tmp_path / "replay.exe"
    replay.write_bytes(b"")
    out = tmp_path / "o.json"
    assert cp.main(["run", str(empty), "--replay", str(replay), "--out", str(out)]) == 2
    assert not out.exists()
    cases = make_cases(tmp_path / "cases", "f_1.prfc")
    assert (
        cp.main(["run", str(cases), "--replay", str(replay), "--out", str(tmp_path / "o.txt")]) == 2
    )


def test_a_baseline_that_reproduces_a_recorded_refusal_needs_no_module(tmp_path, monkeypatch):
    # A recorded refusal replays with exit 0 and writes nothing; the refusing pc follows the status.
    out = "BASELINE MATCH REFUSED source_words=0\nactual_refusal=recompile-reject pc=130 op=0x6\n"
    fake_replay(monkeypatch, stdout=out, module=None)
    result = cp.run_case("replay", "baseline", tmp_path / "f_1.prfc")
    assert result["spirv_sha256"] is None
    assert result["verdict"] == (
        "REFUSED source_words=0\nactual_refusal=recompile-reject pc=130 op=0x6"
    )


def test_a_refusal_that_moves_is_a_difference(tmp_path, monkeypatch):
    base_out = "BASELINE MATCH REFUSED source_words=0\nactual_refusal=recompile-reject pc=130\n"
    cand_out = "CANDIDATE REFUSED source_words=0\nactual_refusal=recompile-reject pc=334\n"
    fake_replay(monkeypatch, stdout=base_out, module=None)
    base = {"f_1.prfc": cp.run_case("replay", "baseline", tmp_path / "f_1.prfc")}
    fake_replay(monkeypatch, stdout=cand_out, module=None)
    cand = {"f_1.prfc": cp.run_case("replay", "candidate", tmp_path / "f_1.prfc")}
    assert cp.diff(base, cand) == ["changed: f_1.prfc"]


def test_refusal_text_on_stderr_is_part_of_the_answer(tmp_path, monkeypatch):
    class Split(_Done):
        def __init__(self):
            super().__init__("CANDIDATE REFUSED source_words=0\n", 3)
            self.stderr = "actual_refusal=recompile-reject pc=9\n"

    monkeypatch.setattr(cp.subprocess, "run", lambda cmd, **_kw: Split())
    result = cp.run_case("replay", "candidate", tmp_path / "f_1.prfc")
    assert result["verdict"].endswith("actual_refusal=recompile-reject pc=9")
