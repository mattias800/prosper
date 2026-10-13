"""The visible adapter keeps the seventeen existing guard requests and route environments."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("visible_snapshot_subject", HERE / "snapshot.py")
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


def test_visible_frontend_uses_the_normal_app_cpu_publication_path():
    env = {"PROSPER_SNAPSHOT_APP": "build/prosper-app", "PROSPER_PAD_SCRIPT": "@route.pad"}
    assert SNAPSHOT.content_capture_frontend(env) == ("build/prosper-app", True)
    assert env["PROSPER_APP_GPU_PRESENT"] == "0"
    assert env["PROSPER_FLIP_PACE_FPS"] == "0"
    assert env["PROSPER_PAD_SCRIPT"] == "@route.pad"


@pytest.mark.parametrize("driver", ["offscreen", "dummy"])
def test_visible_frontend_refuses_hidden_drivers(driver):
    with pytest.raises(RuntimeError, match="refuses"):
        SNAPSHOT.content_capture_frontend(
            {"PROSPER_SNAPSHOT_APP": "app", "SDL_VIDEODRIVER": driver}
        )


def test_guard_explicit_flip_pacing_is_preserved():
    env = {"PROSPER_SNAPSHOT_APP": "app", "PROSPER_FLIP_PACE_FPS": "60"}
    SNAPSHOT.content_capture_frontend(env)
    assert env["PROSPER_FLIP_PACE_FPS"] == "60"


def test_capture_content_wires_the_actual_visible_command_and_guard_policy(monkeypatch, tmp_path):
    app = tmp_path / "prosper-app"
    app.write_text("not executed")
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setenv("PROSPER_SNAPSHOT_APP", str(app))
    monkeypatch.setenv("SDL_VIDEODRIVER", "wayland")
    monkeypatch.setattr(SNAPSHOT, "resolve_dump", lambda entry: str(dump))
    captured = []
    stopped = []

    class Process:
        returncode = 0

        def poll(self):
            return 0

    process = Process()

    def spawn(command, env, log):
        captured.append((command, dict(env)))
        return process

    monkeypatch.setattr(SNAPSHOT, "_spawn_snapshot_process", spawn)
    monkeypatch.setattr(SNAPSHOT, "_stop_snapshot_process", stopped.append)
    monkeypatch.setattr(
        SNAPSHOT,
        "analyze_content_manifest",
        lambda directory, entry: {"richest": {"path": "unused"}},
    )
    entry = json.loads((HERE / "snapshots.json").read_text())["snapshots"][0]
    _, _, directory = SNAPSHOT.capture_content(entry)
    try:
        command, env = captured[0]
        assert "--snapshot-capture" in command
        assert "--snapshot-require-composited-frame" in command
        assert env["PROSPER_APP_GPU_PRESENT"] == "0"
        assert env["PROSPER_FLIP_PACE_FPS"] == "0"
        assert command[command.index("--snapshot-count") + 1] == "320"
        assert command[command.index("--snapshot-timeout") + 1] == "350"
        assert env["PROSPER_RENDER_SCALE"] == "4"
        assert env["PROSPER_RENDER_EVERY"] == "1"
        assert env["PROSPER_GUEST_ARGS"] == "-force-gfx-direct"
        assert env["PROSPER_PAD_SCRIPT"].endswith(entry["pad_script"])
        assert env["PROSPER_SAVEDATA_DIR"] == str(Path(directory) / "savedata")
        assert env["PROSPER_SAVE0"] == str(Path(directory) / "savedata0")
        assert stopped == [process]
    finally:
        SNAPSHOT._cleanup(directory)


def test_legacy_frontend_keeps_its_environment(monkeypatch):
    monkeypatch.setenv("PROSPER_SCREENSHOT", "legacy")
    env = {"PROSPER_PAD_SCRIPT": "@route.pad"}
    assert SNAPSHOT.content_capture_frontend(env) == ("legacy", False)
    assert env == {"PROSPER_PAD_SCRIPT": "@route.pad"}


def test_visible_selection_never_falls_back_to_a_headless_exact_capture(monkeypatch):
    monkeypatch.setenv("PROSPER_SNAPSHOT_APP", "app")
    with pytest.raises(RuntimeError, match="separate boot_trace contract"):
        SNAPSHOT.capture({"frame": 10})


@pytest.mark.parametrize(
    "entry",
    json.loads((HERE / "snapshots.json").read_text())["snapshots"],
    ids=lambda entry: entry["name"],
)
def test_every_guard_keeps_identical_sampler_values(entry):
    interval = float(entry.get("sample_seconds", 1))
    end = float(entry.get("capture_before_seconds", entry["timeout"]))
    count = max(2, math.ceil(end / interval) + 1)
    timeout = max(entry["timeout"], math.ceil(end)) + 30
    old = SNAPSHOT.content_capture_command("legacy", "dump", "capture", interval, count, timeout)
    visible = SNAPSHOT.content_capture_command(
        "app", "dump", "capture", interval, count, timeout, visible=True
    )
    assert visible[:3] == ["app", "dump", "--snapshot-capture"]
    translated = [argument.replace("--snapshot-", "--") for argument in visible[3:]]
    assert translated == old[2:]
    assert "--require-composited-frame" in translated
    assert str(count) in translated
