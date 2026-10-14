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


def fake_replay(monkeypatch, *, code=0, stdout=None, module=b"M" * 36):
    """Replace subprocess.run: write `module` to the --output path (when not None) and answer."""

    def run(cmd, **_kw):
        if module is not None:
            Path(cmd[-1]).write_bytes(module)
        prefix = "BASELINE MATCH " if cmd[1] == "--baseline" else "CANDIDATE "
        answer = stdout if stdout is not None else prefix + "PRODUCED source_words=9\n"
        return _Done(answer, code)

    monkeypatch.setattr(cp.subprocess, "run", run)


def make_cases(directory: Path, *names: str) -> Path:
    directory.mkdir(exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\x00")
    return directory


@pytest.fixture
def case_path(tmp_path):
    """A real input file so replay records can bind its immutable bytes."""
    return make_cases(tmp_path, "f_1.prfc") / "f_1.prfc"


def test_verdict_strips_the_mode_prefix_so_the_two_modes_compare():
    assert cp.verdict_of("noise\nBASELINE MATCH PRODUCED source_words=9\n") == (
        "PRODUCED source_words=9"
    )
    assert cp.verdict_of("CANDIDATE PRODUCED source_words=9\n") == "PRODUCED source_words=9"
    assert cp.verdict_of("CANDIDATE REFUSED source_words=0\nactual_refusal=pc=9\n") == (
        "REFUSED source_words=0\nactual_refusal=pc=9"
    )


def test_silent_replay_is_not_a_verdict():
    with pytest.raises(cp.ReplayError):
        cp.verdict_of("\n  \n")


def test_a_produced_module_is_hashed(tmp_path, case_path, monkeypatch):
    fake_replay(monkeypatch, module=b"abc" * 12)
    result = cp.run_case("replay", "candidate", case_path)
    assert result["spirv_sha256"] is not None
    assert result["verdict"] == "PRODUCED source_words=9"


def test_a_complete_candidate_that_still_refuses_is_an_answer(tmp_path, case_path, monkeypatch):
    fake_replay(
        monkeypatch,
        code=3,
        stdout="CANDIDATE REFUSED source_words=0\nactual_refusal=pc=9\n",
        module=None,
    )
    result = cp.run_case("replay", "candidate", case_path)
    assert result["spirv_sha256"] is None
    assert result["verdict"] == "REFUSED source_words=0\nactual_refusal=pc=9"


@pytest.mark.parametrize("code", [1, 2, 139, -11])
def test_any_other_exit_is_a_replay_that_could_not_run(tmp_path, case_path, monkeypatch, code):
    fake_replay(monkeypatch, code=code)
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", "baseline", case_path)


def test_success_without_a_module_is_not_recorded(tmp_path, case_path, monkeypatch):
    fake_replay(monkeypatch, module=None)
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", "baseline", case_path)


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
    fake_replay(monkeypatch, module=b"same" * 9)
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
    fake_replay(monkeypatch, module=b"different" * 4)
    assert cp.main(["run", str(cases), "--replay", str(replay), "--out", str(cand)]) == 0
    assert cp.main(["compare", str(base), str(cand)]) == 1
    assert "changed: f_1.prfc" in capsys.readouterr().out


def test_compare_with_nothing_in_common_is_unevaluated(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    record = {
        "case_sha256": "0" * 64,
        "spirv_sha256": None,
        "verdict": "REFUSED source_words=0\nactual_refusal=pc=9",
    }
    a.write_text(json.dumps({"x.prfc": record}))
    b.write_text(json.dumps({"y.prfc": record}))
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


def test_a_baseline_that_reproduces_a_recorded_refusal_needs_no_module(
    tmp_path, case_path, monkeypatch
):
    # A recorded refusal replays with exit 0 and writes nothing; the refusing pc follows the status.
    out = "BASELINE MATCH REFUSED source_words=0\nactual_refusal=recompile-reject pc=130 op=0x6\n"
    fake_replay(monkeypatch, stdout=out, module=None)
    result = cp.run_case("replay", "baseline", case_path)
    assert result["spirv_sha256"] is None
    assert result["verdict"] == (
        "REFUSED source_words=0\nactual_refusal=recompile-reject pc=130 op=0x6"
    )


def test_a_refusal_that_moves_is_a_difference(tmp_path, case_path, monkeypatch):
    base_out = "BASELINE MATCH REFUSED source_words=0\nactual_refusal=recompile-reject pc=130\n"
    cand_out = "CANDIDATE REFUSED source_words=0\nactual_refusal=recompile-reject pc=334\n"
    fake_replay(monkeypatch, stdout=base_out, module=None)
    base = {"f_1.prfc": cp.run_case("replay", "baseline", case_path)}
    fake_replay(monkeypatch, code=3, stdout=cand_out, module=None)
    cand = {"f_1.prfc": cp.run_case("replay", "candidate", case_path)}
    assert cp.diff(base, cand) == ["changed: f_1.prfc"]


def test_refusal_text_on_stderr_is_part_of_the_answer(tmp_path, case_path, monkeypatch):
    class Split(_Done):
        def __init__(self):
            super().__init__("CANDIDATE REFUSED source_words=0\n", 3)
            self.stderr = "actual_refusal=recompile-reject pc=9\n"

    monkeypatch.setattr(cp.subprocess, "run", lambda cmd, **_kw: Split())
    result = cp.run_case("replay", "candidate", case_path)
    assert result["verdict"].endswith("actual_refusal=recompile-reject pc=9")


@pytest.mark.parametrize(
    "output",
    [
        "actual_refusal=recompile-reject pc=9\n",
        "CANDIDATE REFUSED source_words=0\n",
        "CANDIDATE SOMETHING source_words=0\n",
        "CANDIDATE REFUSED source_words=0\nCANDIDATE REFUSED source_words=0\n",
        "BASELINE MATCH REFUSED source_words=0\n",
        "CANDIDATE REFUSED source_words=0\nactual_refusal=\n",
    ],
)
def test_missing_unknown_or_duplicate_status_is_unevaluated(
    tmp_path, case_path, monkeypatch, output
):
    fake_replay(monkeypatch, code=3, stdout=output, module=None)
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", "candidate", case_path)


@pytest.mark.parametrize(
    ("code", "output", "module"),
    [
        (3, "CANDIDATE PRODUCED source_words=1\n", b"abcd"),
        (0, "BASELINE MATCH REFUSED source_words=0\nactual_refusal=pc=9\n", b"abcd"),
        (0, "CANDIDATE PRODUCED source_words=9\n", b""),
        (0, "CANDIDATE PRODUCED source_words=9\n", b"abcd"),
        (0, "CANDIDATE REFUSED source_words=0\nactual_refusal=pc=9\n", None),
        (3, "BASELINE MATCH REFUSED source_words=0\nactual_refusal=pc=9\n", None),
    ],
)
def test_inconsistent_exit_or_module_is_unevaluated(
    tmp_path, case_path, monkeypatch, code, output, module
):
    fake_replay(monkeypatch, code=code, stdout=output, module=module)
    mode = "baseline" if output.startswith("BASELINE") else "candidate"
    with pytest.raises(cp.ReplayError):
        cp.run_case("replay", mode, case_path)


@pytest.mark.parametrize("record", [None, {}, {"spirv_sha256": None, "verdict": "noise"}])
def test_equal_malformed_records_are_unevaluated(tmp_path, record):
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    for path in (base, cand):
        path.write_text(json.dumps({"f_1.prfc": record}))
    assert cp.main(["compare", str(base), str(cand)]) == 2


def test_changed_case_bytes_cannot_compare_as_the_same_corpus(tmp_path, monkeypatch):
    cases = make_cases(tmp_path / "cases", "f_1.prfc")
    fake_replay(monkeypatch, module=b"same" * 9)
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    replay = tmp_path / "replay"
    replay.write_bytes(b"")
    assert cp.main(["run", str(cases), "--replay", str(replay), "--out", str(base)]) == 0
    (cases / "f_1.prfc").write_bytes(b"changed input")
    assert cp.main(["run", str(cases), "--replay", str(replay), "--out", str(cand)]) == 0
    assert cp.main(["compare", str(base), str(cand)]) == 1


def test_baseline_and_candidate_refusals_compare_equal(tmp_path, monkeypatch):
    cases = make_cases(tmp_path / "cases", "f_1.prfc")
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    diagnostic = "actual_refusal=recompile-reject pc=9\n"
    for mode, code, output in (
        ("baseline", 0, base),
        ("candidate", 3, cand),
    ):
        prefix = "BASELINE MATCH " if mode == "baseline" else "CANDIDATE "
        fake_replay(
            monkeypatch,
            code=code,
            stdout=prefix + "REFUSED source_words=0\n" + diagnostic,
            module=None,
        )
        assert (
            cp.main(
                [
                    "run",
                    str(cases),
                    "--replay",
                    "replay",
                    "--mode",
                    mode,
                    "--out",
                    str(output),
                ]
            )
            == 0
        )
    assert cp.main(["compare", str(base), str(cand)]) == 0


def test_a_case_changed_during_replay_is_unevaluated(case_path, monkeypatch):
    def mutate(cmd, **_kw):
        case_path.write_bytes(b"different input")
        return _Done("CANDIDATE REFUSED source_words=0\nactual_refusal=pc=9\n", 3)

    monkeypatch.setattr(cp.subprocess, "run", mutate)
    with pytest.raises(cp.ReplayError, match="changed during replay"):
        cp.run_case("replay", "candidate", case_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_sha256", None),
        ("case_sha256", "bad hash"),
        ("spirv_sha256", "bad hash"),
        ("spirv_sha256", "0" * 64),
        ("verdict", "PRODUCED source_words=0"),
        ("verdict", "REFUSED source_words=0\nactual_refusal="),
    ],
)
def test_invalid_record_fields_are_unevaluated(tmp_path, field, value):
    record = {
        "case_sha256": "0" * 64,
        "spirv_sha256": None,
        "verdict": "REFUSED source_words=0\nactual_refusal=pc=9",
    }
    record[field] = value
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    for path in (base, cand):
        path.write_text(json.dumps({"f_1.prfc": record}))
    assert cp.main(["compare", str(base), str(cand)]) == 2
