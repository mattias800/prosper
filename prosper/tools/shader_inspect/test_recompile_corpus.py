"""Tests for recompile_corpus.py: the pure parts that decide what counts as a changed answer."""

import json
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
    assert len(bad) == 1
    assert "['aaaa'] gone" in bad[0]
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


def test_empty_corpus_is_unevaluated_without_starting_inspector(tmp_path, monkeypatch):
    inspector = tmp_path / "inspector.exe"
    inspector.write_bytes(b"")
    monkeypatch.setattr(
        rc.subprocess, "run", lambda *_a, **_k: pytest.fail("empty corpus ran inspector")
    )
    assert rc.main(["record", str(tmp_path), "--inspect", str(inspector)]) == 2


@pytest.mark.parametrize("text", ["not JSON", "[]", "null"])
def test_invalid_baseline_is_unevaluated(tmp_path, monkeypatch, text):
    inspector = tmp_path / "inspector.exe"
    inspector.write_bytes(b"")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(text)
    monkeypatch.setattr(rc, "collect", lambda *_a: {"cs_x.bin": {"stage": "compute"}})
    assert (
        rc.main(["check", str(tmp_path), "--inspect", str(inspector), "--baseline", str(baseline)])
        == 2
    )


@pytest.mark.parametrize("output_kind", ["directory", "missing-parent", "file"])
def test_record_classifies_baseline_write_outcome(tmp_path, monkeypatch, capsys, output_kind):
    """A failed baseline write is unevaluated, not a shader difference or a successful record."""
    inspector = tmp_path / "inspector.exe"
    inspector.write_bytes(b"")
    results = {"cs_x.bin": {"stage": "compute"}}
    monkeypatch.setattr(rc, "collect", lambda *_a: results)
    output = tmp_path / "baseline.json"
    if output_kind == "directory":
        output.mkdir()
    elif output_kind == "missing-parent":
        output = tmp_path / "absent" / "baseline.json"

    code = rc.main(["record", str(tmp_path), "--inspect", str(inspector), "--out", str(output)])
    captured = capsys.readouterr()
    if output_kind == "file":
        assert code == 0
        assert json.loads(output.read_text()) == results
        assert "recorded 1 blobs" in captured.out
        assert captured.err == ""
    else:
        assert code == 2
        assert "could not write baseline:" in captured.err
        assert captured.out == ""
        assert not output.is_file()


def test_baseline_arguments_must_name_json_files(tmp_path):
    assert rc.json_path(tmp_path / "b.json") == (tmp_path / "b.json").resolve()
    assert rc.json_path(tmp_path / "b.txt") is None
    assert rc.json_path(tmp_path / "noextension") is None
    assert rc.json_path(tmp_path / "b.json.bak") is None


# ---- the all-stage corpus gate (shader_corpus.py) -------------------------------------------

import shader_corpus as sc  # noqa: E402

BCNT = "d7640007,00020e09"  # v_bcnt_u32_b32 v7, s7, s9 (VOP3)


def test_stage_of_names_chained_vertex_mains():
    assert rc.stage_of(Path("vsmain_1_2.bin")) == "vertex"


def test_classify_names_the_refused_instruction_from_its_own_words():
    live = (
        f"recompile-reject sh=bfa00003/7 mode=unresolved-operand pc=12 words={BCNT} fmt=9 op=0x364"
    )
    got = sc.classify(live)
    assert got == {
        "reason": "recompile-reject mode=unresolved-operand",
        "pc": 12,
        "opcode": "V_BCNT_U32_B32",
    }
    # The --stage spelling of the same reject classifies identically.
    stage = (
        f"[recompile-reject] mode=unresolved-operand pc=12 words={BCNT} fmt=9 op=0x364 "
        "stage=compute program=none role=terminal"
    )
    assert sc.classify(stage) == got


def test_classify_keeps_mode_and_reason_and_reads_a_wrappers_inner_reject():
    fresh = (
        "recompile-reject sh=1/2 mode=unsupported-in-stage reason=fabricated-fresh-mask-source"
        " pc=40 words=7c020300 fmt=8 op=0x1"
    )
    assert sc.classify(fresh)["reason"] == (
        "recompile-reject mode=unsupported-in-stage reason=fabricated-fresh-mask-source"
    )
    assert sc.classify(fresh)["opcode"] == "V_CMP_LT_F32"
    ngg = (
        "ngg-subgroup-reject reason=ngg-compile-rejected inner=recompile-reject sh=1/2 "
        "mode=unresolved-operand pc=7 words=80000201 fmt=0 op=0x0"
    )
    assert sc.classify(ngg) == {
        "reason": "ngg-subgroup-reject reason=ngg-compile-rejected inner-mode=unresolved-operand",
        "pc": 7,
        "opcode": "S_ADD_U32",
    }


def test_classify_falls_back_to_the_program_then_to_format_and_opcode():
    cfg = sc.classify(
        "compute-cfg-reject pc=1 reason=wave64-ambiguous-mask-read",
        code=bytes(4) + (0x80000201).to_bytes(4, "little"),
    )
    assert cfg == {
        "reason": "compute-cfg-reject reason=wave64-ambiguous-mask-read",
        "pc": 1,
        "opcode": "S_ADD_U32",
    }
    assert sc.classify("cfg-recompile-reject mode=x pc=900 fmt=14 op=0x8")["opcode"] == "MIMG:0x8"
    free = sc.classify("recompile-reject counted-loop enters with narrowed EXEC 12")
    assert free == {
        "reason": "recompile-reject counted-loop enters with narrowed EXEC N",
        "pc": None,
        "opcode": None,
    }
    assert sc.classify("") == {"reason": None, "pc": None, "opcode": None}


def _dump(parent: Path, stamp: str, files: dict, index: str = "") -> Path:
    d = parent / f"refused_shaders_{stamp}Z_000001"
    d.mkdir(parents=True)
    for name, data in files.items():
        (d / name).write_bytes(data)
    if index:
        (d / "index.txt").write_text(index)
    return d


def test_harvest_dedups_by_content_titles_every_dump_and_never_touches_sources(tmp_path):
    home = tmp_path / "home"
    a = _dump(
        home / "runs" / "PPSA11111",
        "20261010-082040",
        {
            "cs_10_aa.bin": b"\1\0\0\0",
            "cs_10_aa.inputs": b"PRRSINP\0\4\0\0\0",
            "ps_20_bb.bin": b"\2\0\0\0",
        },
        'cs addr=0x10 file=cs_10_aa.bin reject="recompile-reject mode=m pc=1"\n',
    )
    run = home / "kena-run"
    run.mkdir(parents=True)
    (run / "run.log").write_text("absent module: /mnt/testdata/PPSA22222-app0/Media/x\n")
    b = _dump(
        run / "cap",
        "20261011-000000",
        {"vsmain_30_cc.bin": b"\3\0\0\0", "ps_20_bb.bin": b"\2\0\0\0"},
    )
    # Same code as a titled dump's compute program, nothing else to go on: inherits its title.
    c = _dump(home / "scratch", "20261012-000000", {"cs_99_aa.bin": b"\1\0\0\0"})
    build = home / "wt" / "build-linux"
    build.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("")
    _dump(build, "20261013-000000", {"cs_40_dd.bin": b"\4\0\0\0"})
    _dump(home / "prosper-testdata" / "x", "20261013-000000", {"cs_50_ee.bin": b"\5\0\0\0"})
    before = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}

    corpus = tmp_path / "corpus"
    manifest = sc.harvest(corpus, [home], [home / "prosper-testdata"], log=lambda _m: None)

    assert {p: p.read_bytes() for p in home.rglob("*") if p.is_file()} == before
    progs = manifest["programs"]
    # cs with its sidecar, the same cs code bare (a different entry), ps from two runs, vsmain;
    # the ctest dump and the excluded tree contribute nothing.
    assert sorted(e["prefix"] for e in progs.values()) == ["cs", "cs", "ps", "vsmain"]
    assert all(n.startswith(f"{e['stage']}/{e['prefix']}_") for n, e in progs.items())
    ps = next(e for e in progs.values() if e["prefix"] == "ps")
    assert ps["titles"] == ["PPSA11111", "PPSA22222"]
    assert {s["title_from"] for s in ps["sources"]} == {"path", "run-files"}
    cs, bare = sorted(
        (e for e in progs.values() if e["prefix"] == "cs"), key=lambda e: e["inputs_sha256"] is None
    )
    assert cs["code_sha256"] == bare["code_sha256"] and bare["inputs_sha256"] is None
    assert [(s["title"], s["title_from"]) for s in bare["sources"]] == [
        ("PPSA11111", "shared-code")
    ]
    assert str(c) in bare["sources"][0]["path"]
    assert cs["inputs_schema"] == 4 and cs["sources"][0]["date"] == "2026-10-10T08:20:40Z"
    assert cs["sources"][0]["live_reject"] == "recompile-reject mode=m pc=1"
    vsmain = next(e for e in progs.values() if e["prefix"] == "vsmain")
    assert vsmain["stage"] == "vertex" and vsmain["titles"] == ["PPSA22222"]
    for name, e in progs.items():
        assert (corpus / name).exists()
        assert (corpus / name).with_suffix(".inputs").exists() == bool(e["inputs_sha256"])
    assert str(a) in cs["sources"][0]["path"] and any(str(b) in s["path"] for s in ps["sources"])

    # Re-running adds nothing it already has.
    again = sc.harvest(corpus, [home], [home / "prosper-testdata"], log=lambda _m: None)
    assert sum(len(e["sources"]) for e in again["programs"].values()) == 5


class _FakeInspector:
    """A shader_inspect stand-in: answers from a {code bytes: answer} table and counts calls."""

    def __init__(self, stage: dict, replay: dict | None = None, inputs_spirv=True):
        self.stage, self.replay, self.inputs_spirv, self.calls = (
            stage,
            replay or {},
            inputs_spirv,
            [],
        )

    def __call__(self, cmd, **_kw):
        self.calls.append(cmd)
        if len(cmd) == 1:
            spv = " [--spirv-out FILE]" if self.inputs_spirv else ""
            return _Done(f"usage: x <stage_*.bin> --inputs <stage_*.inputs>{spv}\n", 2)
        code = Path(cmd[1]).read_bytes()
        out = cmd[cmd.index("--spirv-out") + 1] if "--spirv-out" in cmd else None
        if "--inputs" in cmd:
            answer = self.replay.get(code, ("unreadable", ""))
            if answer[0] == "unreadable":
                return _Done("shader_inspect: x.inputs: refused-shader inputs schema\n", 2)
            if answer[0] == "ok" and out:
                Path(out).write_bytes(answer[1].encode())
            return _Done(
                f'inputs-replay status={answer[0]} reject="{answer[1] if answer[0] != "ok" else ""}" '
                "spirv_dwords=0\n",
                6 if answer[0] == "ok" else 0,
            )
        status, reject = self.stage[code]
        synthetic = "--synthetic-table" in cmd
        if status == "ok":
            if out:
                Path(out).write_bytes(reject.encode())
            s = "ok-synthetic-table" if synthetic else "ok"
            return _Done(
                f"stage-recompile stage=x status={s} spirv_dwords=9\n", 4 if synthetic else 0
            )
        s = "rejected-synthetic-table" if synthetic else "rejected"
        return _Done(
            f"[{reject}] stage=x program=none role=terminal\n"
            f"stage-recompile stage=x status={s} spirv_dwords=0\n",
            5 if synthetic else 1,
        )


def _corpus(tmp_path, programs: dict) -> Path:
    """programs: {name: (code bytes, inputs bytes or None, titles)}."""
    corpus = tmp_path / "corpus"
    entries = {}
    for name, (code, inputs, titles) in programs.items():
        (corpus / name).parent.mkdir(parents=True, exist_ok=True)
        (corpus / name).write_bytes(code)
        if inputs is not None:
            (corpus / name).with_suffix(".inputs").write_bytes(inputs)
        entries[name] = {
            "stage": name.split("/")[0],
            "prefix": name.split("/")[1].split("_")[0],
            "code_sha256": code.hex(),
            "inputs_sha256": "x" if inputs else None,
            "titles": titles,
            "sources": [],
        }
    (corpus / "manifest.json").write_text(
        json.dumps({"format": sc.CORPUS_FORMAT, "programs": entries})
    )
    return corpus


GUARD = (
    "recompile-reject mode=unsupported-in-stage reason=fabricated-fresh-mask-source "
    "pc=4 words=7c020300"
)


def test_survey_answers_each_distinct_code_once_and_replays_every_sidecar(tmp_path, monkeypatch):
    corpus = _corpus(
        tmp_path,
        {
            "compute/cs_a.bin": (b"\1\0\0\0", b"in-a", ["PPSA1"]),
            "compute/cs_b.bin": (b"\1\0\0\0", b"in-b", ["PPSA2"]),  # same code, other run
            "fragment/ps_c.bin": (b"\2\0\0\0", None, ["PPSA1"]),
        },
    )
    fake = _FakeInspector(
        {b"\1\0\0\0": ("ok", ""), b"\2\0\0\0": ("rejected", GUARD)}, {b"\1\0\0\0": ("ok", "module")}
    )
    monkeypatch.setattr(sc.subprocess, "run", fake)
    res = sc.survey(corpus, "inspector", jobs=2, log=lambda _m: None)
    stage_calls = [c for c in fake.calls if "--stage" in c]
    assert len(stage_calls) == 2
    assert ["--synthetic-table" in c for c in stage_calls].count(True) == 1
    p = res["programs"]
    assert (
        p["compute/cs_a.bin"]["replay"]["spirv_sha256"]
        == p["compute/cs_b.bin"]["replay"]["spirv_sha256"]
        is not None
    )
    assert "replay" not in p["fragment/ps_c.bin"]
    assert p["fragment/ps_c.bin"]["stage"]["opcode"] == "V_CMP_LT_F32"


def test_flips_group_by_refusal_and_rank_by_titles(tmp_path):
    def row(stage, replay=None):
        return {"stage": stage, **({"replay": replay} if replay else {})}

    acc = {"verdict": "accepted", "spirv_sha256": "s1"}
    ref = {"verdict": "refused", "reason": "R", "opcode": "V_CMP_LT_F32", "pc": 4}
    other = {"verdict": "refused", "reason": "Q", "opcode": "S_ADD_U32", "pc": 9}
    unread = {"verdict": "unreadable"}
    manifest = {
        "programs": {
            "compute/cs_1.bin": {"stage": "compute", "code_sha256": "c1", "titles": ["PPSA1"]},
            "compute/cs_2.bin": {"stage": "compute", "code_sha256": "c1", "titles": ["PPSA1"]},
            "fragment/ps_3.bin": {"stage": "fragment", "code_sha256": "c3", "titles": ["PPSA2"]},
            "vertex/vs_4.bin": {"stage": "vertex", "code_sha256": "c4", "titles": ["PPSA3"]},
            "vertex/vs_5.bin": {"stage": "vertex", "code_sha256": "c5", "titles": []},
            "vertex/vs_6.bin": {"stage": "vertex", "code_sha256": "c6", "titles": ["PPSA3"]},
        }
    }
    base = {
        "programs": {
            "compute/cs_1.bin": row(ref, acc),  # replay decides: accepted -> refused
            "compute/cs_2.bin": row(ref, acc),  # same code, second run: counted once
            "fragment/ps_3.bin": row(acc, unread),  # old build cannot read: stage decides
            "vertex/vs_4.bin": row(other),
            "vertex/vs_5.bin": row(acc),
            "vertex/vs_6.bin": row(acc),
        }
    }
    cand = {
        "programs": {
            "compute/cs_1.bin": row(acc, ref),
            "compute/cs_2.bin": row(acc, ref),
            "fragment/ps_3.bin": row(ref, acc),
            "vertex/vs_4.bin": row(acc),
            "vertex/vs_5.bin": row({**acc, "spirv_sha256": "s2"}),
            "vertex/vs_6.bin": row(acc),
        }
    }
    rep = sc.flips(manifest, base, cand)
    assert rep["modes"] == {"replay": 2, "stage": 4}
    assert rep["changes"] == {"newly-refused": 3, "newly-accepted": 1, "output-changed": 1}
    [g] = rep["groups"]["newly-refused"]
    assert (g["reason"], g["opcode"], g["programs"], g["entries"]) == ("R", "V_CMP_LT_F32", 2, 3)
    assert g["titles"] == {"PPSA1": 1, "PPSA2": 1} and g["stages"] == {"compute": 1, "fragment": 1}
    [acc_group] = rep["groups"]["newly-accepted"]
    assert (acc_group["reason"], acc_group["titles"]) == ("Q", {"PPSA3": 1})
    text = "\n".join(sc.format_flips(rep, "old", "new"))
    assert "2 titles     2 programs  V_CMP_LT_F32" in text
    assert sc.change_of(ref, other) == "refused-later"
    assert sc.change_of(other, ref) == "refused-earlier"
    assert sc.change_of(ref, {**other, "pc": None}) == "refusal-moved"
    assert rep["changes_by_mode"]["replay"] == {"newly-refused": 2}
    assert sc.change_of(ref, {"verdict": "incomplete"}) == "verdict-changed"
    assert sc.change_of(acc, dict(acc)) is None
    # A build that wrote no module did not change the output; it was not compared.
    assert sc.change_of(acc, {**acc, "spirv_sha256": None}) is None
    assert rep["accepted_outputs"] == {"compared": 2}


def test_compare_exits_one_on_a_flip_zero_without_and_two_without_an_inspector(
    tmp_path, monkeypatch, capsys
):
    corpus = _corpus(tmp_path, {"compute/cs_a.bin": (b"\1\0\0\0", None, ["PPSA1"])})
    inspector = tmp_path / "inspector"
    inspector.write_bytes(b"")
    answers = iter([("ok", ""), ("rejected", GUARD)])
    current = {}

    def fake(cmd, **kw):
        if len(cmd) == 1:
            current["answer"] = next(answers, ("ok", ""))
        return _FakeInspector({b"\1\0\0\0": current["answer"]})(cmd, **kw)

    monkeypatch.setattr(sc.subprocess, "run", fake)
    argv = [
        "compare",
        str(corpus),
        "--base",
        str(inspector),
        "--candidate",
        str(inspector),
        "--out-dir",
        str(tmp_path / "out"),
        "--jobs",
        "1",
    ]
    assert rc.main(argv) == 1
    report = json.loads((tmp_path / "out" / "flips.json").read_text())
    assert report["changes"] == {"newly-refused": 1}
    assert "fabricated-fresh-mask-source" in capsys.readouterr().out
    assert rc.main(argv) == 0  # both arms now answer "ok"
    assert (
        rc.main(
            [
                "flips",
                str(corpus),
                str(tmp_path / "out" / "base.json"),
                str(tmp_path / "out" / "candidate.json"),
            ]
        )
        == 0
    )

    monkeypatch.setattr(sc.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("x")))
    assert rc.main(argv) == 2


def test_flips_refuses_a_file_that_is_not_a_results_document(tmp_path):
    corpus = _corpus(tmp_path, {"compute/cs_a.bin": (b"\1\0\0\0", None, [])})
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert rc.main(["flips", str(corpus), str(bad), str(bad)]) == 2
