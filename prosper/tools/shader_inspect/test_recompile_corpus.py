"""Tests for recompile_corpus.py: the pure parts that decide what counts as a changed answer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import recompile_corpus as rc  # noqa: E402


def test_stage_of_uses_filename_prefix():
    assert rc.stage_of(Path("cs_40059d0a00_ab.bin")) == "compute"
    assert rc.stage_of(Path("ps_1_2.bin")) == "fragment"
    assert rc.stage_of(Path("vs_1_2.bin")) == "vertex"
    assert rc.stage_of(Path("index.txt")) is None


def test_verdict_lines_keep_only_decisions_in_order():
    text = "\n".join(
        [
            "pc=0001 len=1 fmt=SOPP op=0x001",
            "[recompile-reject] pc=9 mode=unresolved-operand",
            "generic-coverage total=56",
            "stage-recompile stage=compute status=ok spirv_dwords=12",
        ]
    )
    assert rc.verdict_lines(text) == [
        "[recompile-reject] pc=9 mode=unresolved-operand",
        "stage-recompile stage=compute status=ok spirv_dwords=12",
    ]


def test_diff_reports_changed_missing_and_new_blobs():
    base = {"a": {"v": 1}, "b": {"v": 1}, "c": {"v": 1}}
    now = {"a": {"v": 1}, "b": {"v": 2}, "d": {"v": 1}}
    assert rc.diff(base, now) == ["missing from run: c", "not in baseline: d", "changed: b"]
    assert rc.diff(base, base) == []


def test_dump_keys_and_compare(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "success_ps_at_0000000000001000_aaaa_1111.spv").write_bytes(b"")
    (a / "success_cs_at_0000000000002000_bbbb_2222.spv").write_bytes(b"")
    (b / "success_ps_at_0000000000001000_cccc_1111.spv").write_bytes(b"")
    (b / "success_cs_at_0000000000002000_bbbb_2222.spv").write_bytes(b"")
    (b / "success_vs_at_0000000000003000_dddd_3333.spv").write_bytes(b"")
    bad, shared = rc.compare_dumps(rc.dump_keys(a), rc.dump_keys(b))
    assert shared == 2
    assert len(bad) == 1 and "['aaaa'] gone" in bad[0]
    assert rc.compare_dumps(rc.dump_keys(a), rc.dump_keys(a)) == ([], 2)


def test_extra_context_in_after_run_is_not_a_difference():
    before = {("ps", "1000", "r"): {"aaaa"}}
    after = {("ps", "1000", "r"): {"aaaa", "bbbb"}}
    assert rc.compare_dumps(before, after) == ([], 1)


class _Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def test_duplicate_basenames_in_different_directories_are_both_recorded(tmp_path, monkeypatch):
    for sub, verdict in (("a", "status=one"), ("b", "status=two")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "cs_same.bin").write_bytes(verdict.encode())

    def fake_run(cmd, **_kw):
        blob = Path(cmd[1])
        return _Done(f"stage-recompile stage=compute {blob.read_text()}\n")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    results = rc.collect(tmp_path, "inspector")
    assert sorted(results) == ["a/cs_same.bin", "b/cs_same.bin"]
    assert results["a/cs_same.bin"] != results["b/cs_same.bin"]


def test_inspector_failure_is_not_recorded_as_a_verdict(tmp_path, monkeypatch):
    (tmp_path / "cs_x.bin").write_bytes(b"\0")
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, **_kw: _Done("", returncode=1))
    with pytest.raises(rc.InspectorError):
        rc.collect(tmp_path, "inspector")


def test_main_reports_could_not_run_for_a_failing_inspector(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cs_x.bin").write_bytes(b"\0")
    inspector = tmp_path / "inspector.exe"
    inspector.write_bytes(b"")
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, **_kw: _Done("", returncode=1))
    out = tmp_path / "base.json"
    code = rc.main(["record", str(corpus), "--inspect", str(inspector), "--out", str(out)])
    assert code == 2
    assert not out.exists()


def test_exit_zero_without_a_stage_recompile_line_is_an_inspector_failure(tmp_path, monkeypatch):
    (tmp_path / "cs_x.bin").write_bytes(b"\0")
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, **_kw: _Done("noise\n", returncode=0))
    with pytest.raises(rc.InspectorError):
        rc.collect(tmp_path, "inspector")


def test_nonzero_exit_with_a_stage_recompile_line_is_an_accepted_refusal(tmp_path, monkeypatch):
    (tmp_path / "cs_x.bin").write_bytes(b"\0")
    refusal = "stage-recompile stage=compute status=undetermined-no-resource-table\n"
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, **_kw: _Done(refusal, returncode=3))
    results = rc.collect(tmp_path, "inspector")
    assert results["cs_x.bin"]["verdict"] == [refusal.strip()]


@pytest.mark.parametrize("status", [2, -11, 3221225477])
def test_failure_after_printed_verdict_is_not_recorded(tmp_path, monkeypatch, status):
    """A flushed verdict can precede an IO failure or a process crash."""
    (tmp_path / "cs_x.bin").write_bytes(b"\0")
    text = "stage-recompile stage=compute status=ok spirv_dwords=12\n"
    monkeypatch.setattr(rc.subprocess, "run", lambda cmd, **_kw: _Done(text, status))
    with pytest.raises(rc.InspectorError, match="exit"):
        rc.collect(tmp_path, "inspector")


def test_synthetic_write_failure_does_not_create_baseline(tmp_path, monkeypatch):
    """shader_inspect prints synthetic success before it checks output-stream errors."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "vs_x.bin").write_bytes(b"\0")
    inspector = tmp_path / "inspector.exe"
    inspector.write_bytes(b"")

    def fake_run(cmd, **_kw):
        if "--synthetic-table" in cmd:
            return _Done("stage-recompile stage=vertex status=ok-synthetic-table\n", 2)
        return _Done("stage-recompile stage=vertex status=ok\n")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    baseline = tmp_path / "base.json"
    assert (
        rc.main(["record", str(corpus), "--inspect", str(inspector), "--out", str(baseline)]) == 2
    )
    assert not baseline.exists()


@pytest.mark.parametrize("status", [4, 5])
def test_synthetic_success_and_refusal_have_distinct_artifact_results(
    tmp_path, monkeypatch, status
):
    (tmp_path / "vs_x.bin").write_bytes(b"\0")

    def fake_run(cmd, **_kw):
        if "--synthetic-table" in cmd:
            if status == 4:
                Path(cmd[-1]).write_bytes(b"synthetic module")
            verdict = "ok-synthetic-table" if status == 4 else "rejected-synthetic-table"
            return _Done(f"stage-recompile stage=vertex status={verdict}\n", status)
        return _Done("stage-recompile stage=vertex status=rejected\n", 1)

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    entry = rc.collect(tmp_path, "inspector")["vs_x.bin"]
    assert (entry["spirv_sha256"] is not None) == (status == 4)


def test_synthetic_success_without_output_is_not_recorded(tmp_path, monkeypatch):
    (tmp_path / "vs_x.bin").write_bytes(b"\0")

    def fake_run(cmd, **_kw):
        if "--synthetic-table" in cmd:
            return _Done("stage-recompile stage=vertex status=ok-synthetic-table\n", 4)
        return _Done("stage-recompile stage=vertex status=ok\n")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    with pytest.raises(rc.InspectorError, match="SPIR-V output"):
        rc.collect(tmp_path, "inspector")
