"""Tests for check_screenshot_paths: conforming and violating paths, grandfathering, one folder per title."""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_screenshot_paths as csp  # noqa: E402

REPO = HERE.parents[2]
GOOD = "assets/screenshots/PPSA04263-gta5/2026-10-06-prologue-bank-lobby-i2996.webp"


def build(root: Path, images: list[str], legacy: list[str]) -> Path:
    for rel in images:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
    listing = root / csp.LEGACY
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("# header\n" + "".join(f"{p}\n" for p in legacy), encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def problems(root: Path) -> list[str]:
    return csp.evaluate(root)[0]


@pytest.mark.parametrize(
    "path",
    [
        GOOD,
        "assets/screenshots/PPSA24651-messenger/2026-07-14-first-level.webp",
        "assets/screenshots/cross-title/2026-09-01-wave32-fix-i3407.webp",
        "assets/screenshots/app/2026-10-01-settings-panel.webp",
    ],
)
def test_conforming_paths_pass(tmp_path, path):
    assert problems(build(tmp_path, [path], [])) == []


@pytest.mark.parametrize(
    "path",
    [
        "assets/screenshots/gta5-prologue.webp",
        "assets/screenshots/PPSA04263-gta5/prologue.webp",
        "assets/screenshots/PPSA04263-gta5/2026-10-06-prologue.png",
        "assets/screenshots/PPSA04263-gta5/2026-13-06-prologue.webp",
        "assets/screenshots/PPSA04263-gta5/2026-10-06-Prologue_Bank.webp",
        "assets/screenshots/gta5/2026-10-06-prologue.webp",
        "assets/screenshots/PPSA04263/2026-10-06-prologue.webp",
        "assets/screenshots/PPSA04263-gta5/sub/2026-10-06-prologue.webp",
        "prosper/docs/screenshots/PPSA04263-gta5/2026-10-06-prologue.webp",
    ],
)
def test_violating_paths_fail(tmp_path, path):
    found = problems(build(tmp_path, [path], []))
    assert any(path in p and "not assets/screenshots" in p for p in found), found


def test_a_listed_legacy_file_passes(tmp_path):
    legacy = "assets/screenshots/3407-default-control-gta.webp"
    assert problems(build(tmp_path, [legacy], [legacy])) == []


def test_a_stale_legacy_line_fails(tmp_path):
    found = problems(build(tmp_path, [GOOD], ["assets/screenshots/gone.webp"]))
    assert any("gone.webp" in p and "no longer exists" in p for p in found), found


def test_a_conforming_file_must_leave_the_legacy_list(tmp_path):
    found = problems(build(tmp_path, [GOOD], [GOOD]))
    assert any("delete its line" in p for p in found), found


def test_one_folder_per_title_id(tmp_path):
    other = "assets/screenshots/PPSA04263-gtav/2026-10-06-title.webp"
    found = problems(build(tmp_path, [GOOD, other], []))
    assert any("PPSA04263 has more than one folder" in p for p in found), found


def test_missing_legacy_list_cannot_be_evaluated(tmp_path):
    root = build(tmp_path, [GOOD], [])
    (root / csp.LEGACY).unlink()
    with pytest.raises(csp.EvaluationError):
        csp.evaluate(root)


def test_the_repository_itself_is_clean():
    assert problems(REPO) == []
