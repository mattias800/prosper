"""Tests for recompile_corpus.py: the pure parts that decide what counts as a changed answer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import recompile_corpus as rc  # noqa: E402


def test_stage_of_uses_filename_prefix():
    assert rc.stage_of(Path("cs_40059d0a00_ab.bin")) == "compute"
    assert rc.stage_of(Path("ps_1_2.bin")) == "fragment"
    assert rc.stage_of(Path("vs_1_2.bin")) == "vertex"
    assert rc.stage_of(Path("index.txt")) is None


def test_verdict_lines_keep_only_decisions_in_order():
    text = "\n".join([
        "pc=0001 len=1 fmt=SOPP op=0x001",
        "[recompile-reject] pc=9 mode=unresolved-operand",
        "generic-coverage total=56",
        "stage-recompile stage=compute status=ok spirv_dwords=12",
    ])
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
