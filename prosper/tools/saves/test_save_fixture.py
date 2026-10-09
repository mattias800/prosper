"""Tests for save_fixture.py and its snapshot.py integration (no GPU, no game dump needed)."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SF = _load("test_save_fixture_under_test", HERE / "save_fixture.py")
SNAPSHOT = _load("test_save_fixture_snapshot", HERE.parent / "snapshot" / "snapshot.py")

TITLE = "PPSA00001"


def _write_param(dump_root: Path, version: str | None) -> None:
    sce_sys = dump_root / f"{TITLE}-app0" / "sce_sys"
    sce_sys.mkdir(parents=True, exist_ok=True)
    body = {"titleId": TITLE}
    if version is not None:
        body["contentVersion"] = version
    (sce_sys / "param.json").write_text(json.dumps(body))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated fixture root, manifest, dump root and a source pair of save roots."""
    monkeypatch.setenv("PROSPER_SAVE_FIXTURES", str(tmp_path / "fixtures"))
    monkeypatch.setenv("PROSPER_SAVE_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("PROSPER_GAME_ROOT", str(tmp_path / "dumps"))
    dump_root = tmp_path / "dumps"
    _write_param(dump_root, "01.000.000")
    savedata = tmp_path / "src-savedata"
    save0 = tmp_path / "src-save0"
    (savedata / "slot0").mkdir(parents=True)
    (savedata / "slot0" / "data.bin").write_bytes(b"\x01\x02\x03 settings")
    (save0 / "profile").mkdir(parents=True)
    (save0 / "profile" / "eula.txt").write_text("accepted")
    (save0 / "empty-dir").mkdir()
    return {"tmp": tmp_path, "dump_root": dump_root, "savedata": savedata, "save0": save0}


def _capture(env, state="post-setup", **kwargs):
    return SF.capture(TITLE, state, env["savedata"], env["save0"], note="past EULA", **kwargs)


def _snapshot(path: Path) -> dict:
    return {p.relative_to(path).as_posix(): (p.read_bytes() if p.is_file() else None)
            for p in sorted(path.rglob("*"))}


def test_capture_seed_round_trip(env):
    meta = _capture(env, route="prosper/scripts/x/route.pad")
    assert meta["title_id"] == TITLE
    assert meta["content_version"] == "01.000.000"
    assert meta["description"] == "past EULA"
    assert meta["route"] == "prosper/scripts/x/route.pad"
    assert meta["file_count"] == 2
    assert len(meta["tree_sha256"]) == 64

    run = env["tmp"] / "run"
    out = SF.seed(TITLE, "post-setup", run)

    assert out == {"PROSPER_SAVEDATA_DIR": str(run / "savedata"),
                   "PROSPER_SAVE0": str(run / "save0")}
    assert _snapshot(run / "savedata") == _snapshot(env["savedata"])
    assert _snapshot(run / "save0") == _snapshot(env["save0"])


def test_seed_yields_fresh_dirs_and_never_touches_the_fixture(env):
    _capture(env)
    fixture = SF.fixture_dir(TITLE, "post-setup")
    before = _snapshot(fixture)

    first = env["tmp"] / "run1"
    SF.seed(TITLE, "post-setup", first)
    # A run mutates its save: overwrite, add, delete.
    (first / "savedata" / "slot0" / "data.bin").write_bytes(b"CORRUPTED BY A RUN")
    (first / "savedata" / "slot0" / "extra.bin").write_bytes(b"new")
    (first / "save0" / "profile" / "eula.txt").unlink()

    assert _snapshot(fixture) == before
    SF.verify_fixture(TITLE, "post-setup")

    second = env["tmp"] / "run2"
    SF.seed(TITLE, "post-setup", second)
    assert _snapshot(second / "savedata") == _snapshot(env["savedata"])
    assert _snapshot(second / "save0") == _snapshot(env["save0"])


def test_seed_refuses_a_run_dir_that_already_holds_a_save(env):
    _capture(env)
    run = env["tmp"] / "run"
    (run / "savedata").mkdir(parents=True)
    (run / "savedata" / "stale.bin").write_bytes(b"stale")

    with pytest.raises(SF.FixtureError, match="not empty"):
        SF.seed(TITLE, "post-setup", run)
    assert (run / "savedata" / "stale.bin").read_bytes() == b"stale"


def test_verify_catches_tampering(env):
    _capture(env)
    assert SF.verify_fixture(TITLE, "post-setup")["title_id"] == TITLE
    target = SF.fixture_dir(TITLE, "post-setup") / "savedata" / "slot0" / "data.bin"

    target.write_bytes(b"\x01\x02\x03 settingX")        # same size, one byte changed
    with pytest.raises(SF.FixtureError, match="modified after capture"):
        SF.verify_fixture(TITLE, "post-setup")
    # seed must refuse the same fixture and leave no half-seeded directories behind
    run = env["tmp"] / "run"
    with pytest.raises(SF.FixtureError, match="modified after capture"):
        SF.seed(TITLE, "post-setup", run)
    assert not (run / "savedata").exists() and not (run / "save0").exists()

    target.write_bytes(b"\x01\x02\x03 settings")        # restored: clean again
    SF.verify_fixture(TITLE, "post-setup")
    (target.parent / "added.bin").write_bytes(b"x")     # an added file is tampering too
    with pytest.raises(SF.FixtureError, match="modified after capture"):
        SF.verify_fixture(TITLE, "post-setup")


def test_version_mismatch_warns_loudly_and_strict_fails(env):
    _capture(env)
    warnings: list[str] = []

    SF.seed(TITLE, "post-setup", env["tmp"] / "same", warn=warnings.append)
    assert warnings == []

    _write_param(env["dump_root"], "01.002.000")
    SF.seed(TITLE, "post-setup", env["tmp"] / "newer", warn=warnings.append)
    assert len(warnings) == 1
    assert "CONTENT VERSION MISMATCH" in warnings[0]
    assert "01.000.000" in warnings[0] and "01.002.000" in warnings[0]

    with pytest.raises(SF.FixtureError, match="CONTENT VERSION MISMATCH"):
        SF.seed(TITLE, "post-setup", env["tmp"] / "strict", strict_version=True)
    assert not (env["tmp"] / "strict").exists()


def test_unreadable_dump_version_is_a_warning_not_silence(env):
    _capture(env)
    _write_param(env["dump_root"], None)
    warnings: list[str] = []
    SF.seed(TITLE, "post-setup", env["tmp"] / "run", warn=warnings.append)
    assert len(warnings) == 1 and "cannot compare" in warnings[0]


def test_manifest_pins_sha_and_version(env):
    _capture(env)
    path = SF.pin(TITLE, "post-setup")
    assert path == env["tmp"] / "manifest.json"
    entry = json.loads(path.read_text())["fixtures"][TITLE]["post-setup"]
    assert entry["sha256"] == SF.load_metadata(TITLE, "post-setup")["tree_sha256"]
    assert entry["content_version"] == "01.000.000"
    SF.seed(TITLE, "post-setup", env["tmp"] / "ok")

    # A recapture with different content no longer matches what the manifest pins.
    (env["savedata"] / "slot0" / "data.bin").write_bytes(b"different save")
    _capture(env, force=True)
    with pytest.raises(SF.FixtureError, match="committed manifest"):
        SF.verify_fixture(TITLE, "post-setup")
    with pytest.raises(SF.FixtureError, match="committed manifest"):
        SF.seed(TITLE, "post-setup", env["tmp"] / "blocked")


def test_capture_refuses_overwrite_empty_and_symlinks(env):
    _capture(env)
    with pytest.raises(SF.FixtureError, match="already exists"):
        _capture(env)

    empty_a = env["tmp"] / "ea"
    empty_b = env["tmp"] / "eb"
    empty_a.mkdir()
    empty_b.mkdir()
    with pytest.raises(SF.FixtureError, match="nothing to capture"):
        SF.capture(TITLE, "empty", empty_a, empty_b)
    assert not SF.fixture_exists(TITLE, "empty")

    (env["savedata"] / "link").symlink_to(env["save0"])
    with pytest.raises(SF.FixtureError, match="symlink"):
        _capture(env, state="linked")
    assert not SF.fixture_dir(TITLE, "linked").exists()


@pytest.mark.parametrize("title,state", [("../evil", "s"), (TITLE, "../evil"), (TITLE, ".."),
                                         (TITLE, "a/b"), ("", "s"), (TITLE, "x\n"),
                                         (TITLE + "\n", "s"), (TITLE, "x y")])
def test_names_cannot_escape_the_fixture_root(env, title, state):
    with pytest.raises(SF.FixtureError, match="invalid"):
        SF.fixture_dir(title, state)


def test_list_reports_local_and_missing_manifest_entries(env):
    _capture(env)
    SF.pin(TITLE, "post-setup")
    manifest = json.loads((env["tmp"] / "manifest.json").read_text())
    manifest["fixtures"][TITLE]["later"] = {"sha256": "0" * 64, "content_version": "1"}
    (env["tmp"] / "manifest.json").write_text(json.dumps(manifest))

    rows = {(r["title"], r["state"]): r["status"] for r in SF.list_fixtures()}
    assert rows == {(TITLE, "post-setup"): "local", (TITLE, "later"): "MISSING"}


def test_cli_capture_seed_env_verify(env, capsys):
    base = ["capture", TITLE, "post-setup", "--savedata-dir", str(env["savedata"]),
            "--save0-dir", str(env["save0"]), "--note", "past EULA"]
    assert SF.main(base) == 0
    capsys.readouterr()

    run = env["tmp"] / "cli-run"
    assert SF.main(["seed", TITLE, "post-setup", str(run), "--env"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "PROSPER_SAVEDATA_DIR": str(run / "savedata"), "PROSPER_SAVE0": str(run / "save0")}

    run2 = env["tmp"] / "cli-run2"
    assert SF.main(["seed", TITLE, "post-setup", str(run2)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [shlex.split(line) for line in lines] == [
        ["export", f"PROSPER_SAVEDATA_DIR={run2 / 'savedata'}"],
        ["export", f"PROSPER_SAVE0={run2 / 'save0'}"]]

    assert SF.main(["verify"]) == 0
    (SF.fixture_dir(TITLE, "post-setup") / "save0" / "profile" / "eula.txt").write_text("tampered")
    assert SF.main(["verify"]) == 1
    assert SF.main(["seed", TITLE, "missing-state", str(env["tmp"] / "r3")]) == 1


# ---- snapshot.py integration ---------------------------------------------------------------

def _entry(**extra):
    entry = {"name": "needs-fixture", "dump": f"{TITLE}-app0", "min_colors": 1,
             "review": "ok", "structural_references": [{"luma16x9": "00"}],
             "save_fixture": "post-setup"}
    entry.update(extra)
    return entry


def test_snapshot_skips_a_snap_whose_fixture_is_missing(env, monkeypatch, capsys):
    ran: list[str] = []

    def fail_capture(*args, **kwargs):
        ran.append("capture")
        raise AssertionError("a snap with a missing fixture must not boot")

    monkeypatch.setattr(SNAPSHOT, "capture_content", fail_capture)
    monkeypatch.setattr(SNAPSHOT, "capture", fail_capture)
    monkeypatch.setattr(SNAPSHOT, "FAIL_DIR", str(env["tmp"] / "failures"))
    manifest = {"snapshots": [_entry()]}

    assert SNAPSHOT.cmd_check(manifest, []) == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out and f"{TITLE}/post-setup" in out and "not found" in out
    assert "0 of 1 guard(s) ran, 1 skipped (missing save fixtures)" in out
    assert SNAPSHOT.cmd_verify(manifest, []) == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "1 skipped (missing save fixtures)" in out
    assert ran == []


def test_snapshot_apply_entry_env_seeds_both_roots_from_the_fixture(env):
    _capture(env)
    assert SNAPSHOT.save_fixture_skip_reason(_entry()) is None
    run_env: dict = {}
    tmp = env["tmp"] / "snaprun"

    SNAPSHOT.apply_entry_env(run_env, _entry(savedata_policy="fresh"), str(tmp))

    assert run_env["PROSPER_SAVEDATA_DIR"] == str(tmp / "savedata")
    assert run_env["PROSPER_SAVE0"] == str(tmp / "save0")
    assert _snapshot(tmp / "savedata") == _snapshot(env["savedata"])
    assert _snapshot(tmp / "save0") == _snapshot(env["save0"])


def test_snapshot_apply_entry_env_never_falls_back_to_fresh(env):
    run_env: dict = {}
    with pytest.raises(RuntimeError, match="not found"):
        SNAPSHOT.apply_entry_env(run_env, _entry(savedata_policy="fresh"),
                                 str(env["tmp"] / "snaprun"))
    assert "PROSPER_SAVEDATA_DIR" not in run_env

    _capture(env)
    with pytest.raises(RuntimeError, match="conflicts"):
        SNAPSHOT.apply_entry_env({}, _entry(savedata_policy="preserve"),
                                 str(env["tmp"] / "snaprun2"))


def test_snapshot_without_save_fixture_is_unchanged(env):
    entry = _entry()
    del entry["save_fixture"]
    entry["savedata_policy"] = "fresh"
    assert SNAPSHOT.save_fixture_skip_reason(entry) is None
    run_env: dict = {}
    SNAPSHOT.apply_entry_env(run_env, entry, str(env["tmp"] / "plain"))
    assert os.listdir(run_env["PROSPER_SAVEDATA_DIR"]) == []
    assert os.path.basename(run_env["PROSPER_SAVE0"]) == "savedata0"


def test_empty_directories_are_part_of_the_tree_hash(env):
    _capture(env)
    base = SF.tree_sha256(SF.fixture_dir(TITLE, "post-setup"))[0]
    store = SF.fixture_dir(TITLE, "post-setup")

    (store / "save0" / "another-empty-dir").mkdir()
    assert SF.tree_sha256(store)[0] != base
    with pytest.raises(SF.FixtureError, match="modified after capture"):
        SF.verify_fixture(TITLE, "post-setup")
    (store / "save0" / "another-empty-dir").rmdir()
    (store / "save0" / "empty-dir").rmdir()
    with pytest.raises(SF.FixtureError, match="modified after capture"):
        SF.verify_fixture(TITLE, "post-setup")


@pytest.mark.parametrize("where", ["savedata", "save0", "fixture", "newstate", "title", "store",
                                   "link"])
def test_seed_refuses_run_dirs_that_overlap_the_fixture_store(env, where):
    _capture(env)
    store = SF.fixtures_root()
    fixture = SF.fixture_dir(TITLE, "post-setup")
    before = _snapshot(fixture)
    run = {"savedata": fixture / "savedata", "save0": fixture / "save0", "fixture": fixture,
           "newstate": store / TITLE / "newstate", "title": store / TITLE, "store": store,
           "link": env["tmp"] / "link"}[where]
    if where == "link":
        run.symlink_to(fixture / "savedata")

    with pytest.raises(SF.FixtureError, match="overlaps the fixture store"):
        SF.seed(TITLE, "post-setup", run)

    assert _snapshot(fixture) == before
    SF.verify_fixture(TITLE, "post-setup")
    assert not (store / TITLE / "newstate").exists()
    assert {r["status"] for r in SF.list_fixtures()} == {"local"}


def test_seed_refuses_an_ancestor_of_the_store(env):
    _capture(env)
    with pytest.raises(SF.FixtureError, match="overlaps the fixture store"):
        SF.seed(TITLE, "post-setup", env["tmp"])


def test_seed_refuses_symlinked_destinations_with_a_clear_error(env):
    _capture(env)
    run = env["tmp"] / "run"
    run.mkdir()
    (env["tmp"] / "elsewhere").mkdir()
    (run / "savedata").symlink_to(env["tmp"] / "elsewhere")
    with pytest.raises(SF.FixtureError, match="symlink"):
        SF.seed(TITLE, "post-setup", run)


@pytest.mark.parametrize("bad", ["", 7, None, ["post-setup"], "../x", "a b"])
def test_snapshot_malformed_save_fixture_is_an_error_not_a_skip(env, monkeypatch, capsys, bad):
    monkeypatch.setattr(SNAPSHOT, "FAIL_DIR", str(env["tmp"] / "failures"))
    monkeypatch.setattr(SNAPSHOT, "capture_content",
                        lambda *a, **k: pytest.fail("must not boot"))
    entry = _entry(save_fixture=bad)

    with pytest.raises(RuntimeError, match="save_fixture"):
        SNAPSHOT.save_fixture_skip_reason(entry)
    assert SNAPSHOT.cmd_check({"snapshots": [entry]}, []) == 1
    assert SNAPSHOT.cmd_verify({"snapshots": [entry]}, []) == 1
    captured = capsys.readouterr()
    assert "SKIPPED" not in captured.out and "ERROR" in captured.err


def test_snapshot_undecodable_dump_is_an_error_not_a_skip(env):
    with pytest.raises(RuntimeError, match="invalid save_fixture"):
        SNAPSHOT.save_fixture_skip_reason(_entry(dump="not-a-title-dir"))
