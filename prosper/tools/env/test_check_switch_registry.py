"""Tests for check_switch_registry: a hand-built tree per check, plus a clean control.

Each check is shown firing on a tree built here, not drawn from the repository, so a scan that
quietly stops matching fails a test instead of reporting every switch registered forever.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_switch_registry as csr  # noqa: E402

REPO = HERE.parents[2]
REG = "prosper/tools/env/switch_registry.txt"

SOURCES = {
    "prosper/src/a.cpp": 'auto x = getenv("PROSPER_ALPHA"); PROSPER_ENV_ON("PROSPER_BETA");\n',
    "prosper/frontends/b.cpp": 'const char* k = "PROSPER_GAMMA";\n',
    "prosper/tests/fixtures/c.h": 'getenv("PROSPER_DELTA");\n',
}
REGISTRY = (
    "# header\n"
    "PROSPER_ALPHA  diagnostic\n"
    "PROSPER_BETA  selector  #42  # retired by the issue\n"
    "PROSPER_GAMMA  host-capability\n"
    "PROSPER_DELTA  unclassified\n"
)


def build(root: Path, registry: str | None = REGISTRY, **extra: str) -> Path:
    for rel, text in {**SOURCES, **extra}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if registry is not None:
        (root / REG).parent.mkdir(parents=True, exist_ok=True)
        (root / REG).write_text(registry, encoding="utf-8")
    return root


def problems(root: Path, **kw) -> list[str]:
    return csr.evaluate(root, min_names=1, **kw)[0]


def test_clean_tree_has_no_findings(tmp_path):
    assert problems(build(tmp_path)) == []


def test_the_repository_itself_is_clean():
    assert csr.evaluate(REPO)[0] == []


def test_a_switch_read_only_from_objective_c_is_seen(tmp_path):
    root = build(tmp_path, **{"prosper/frontends/v.mm": 'getenv("PROSPER_MAC_ONLY");'})
    assert "unregistered: PROSPER_MAC_ONLY -- add a row with its class" in problems(root)


def test_a_name_with_trailing_underscore_is_a_prefix_not_a_switch(tmp_path):
    root = build(tmp_path, **{"prosper/src/d.cpp": 'std::string p = "PROSPER_RENDER_";\n'})
    assert problems(root) == []


@pytest.mark.parametrize(
    "registry,expect",
    [
        (REGISTRY.replace("PROSPER_GAMMA  host-capability\n", ""), "unregistered: PROSPER_GAMMA"),
        (REGISTRY + "PROSPER_GONE  diagnostic\n", "stale: PROSPER_GONE"),
        (REGISTRY.replace("host-capability", "useful"), "unknown class `useful`"),
        (REGISTRY.replace("  #42", ""), "names the issue that retires it"),
        (REGISTRY + "PROSPER_ALPHA  diagnostic\n", "duplicates line"),
        (REGISTRY + "PROSPER_ALPHA\n", "not `NAME CLASS"),
    ],
)
def test_each_registry_violation_is_reported(tmp_path, registry, expect):
    found = problems(build(tmp_path, registry))
    assert any(expect in p for p in found), found


def test_too_few_names_cannot_be_evaluated(tmp_path):
    with pytest.raises(csr.EvaluationError):
        csr.evaluate(build(tmp_path))
    assert csr.main(["--root", str(tmp_path)]) == csr.EXIT_UNEVALUATED


def test_missing_registry_cannot_be_evaluated(tmp_path):
    with pytest.raises(csr.EvaluationError):
        csr.evaluate(build(tmp_path, None), min_names=1)


def test_update_adds_unclassified_drops_stale_and_keeps_classes_and_notes(tmp_path):
    root = build(
        tmp_path, REGISTRY + "PROSPER_GONE  diagnostic\n", **{"prosper/src/e.cpp": '"PROSPER_NEW"'}
    )
    csr.update(root, min_names=1)
    text = (root / REG).read_text(encoding="utf-8")
    assert "PROSPER_NEW  unclassified\n" in text
    assert "PROSPER_GONE" not in text
    assert "PROSPER_BETA  selector  #42  # retired by the issue\n" in text
    assert problems(root) == []


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def committed(tmp_path):
    root = build(tmp_path)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return root


def test_a_new_switch_registered_unclassified_is_rejected(committed):
    (committed / "prosper/src/e.cpp").write_text('"PROSPER_NEW"', encoding="utf-8")
    csr.update(committed, min_names=1)
    found = problems(committed, base="HEAD")
    assert any("new-unclassified: PROSPER_NEW" in p for p in found), found


def test_a_new_switch_registered_with_a_class_passes(committed):
    (committed / "prosper/src/e.cpp").write_text('"PROSPER_NEW"', encoding="utf-8")
    with (committed / REG).open("a", encoding="utf-8") as f:
        f.write("PROSPER_NEW  diagnostic\n")
    assert problems(committed, base="HEAD") == []


def test_grandfathered_unclassified_rows_stay_legal(committed):
    assert problems(committed, base="HEAD") == []


def test_a_base_without_the_registry_grandfathers_everything(tmp_path):
    root = build(tmp_path, None)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "before the registry")
    build(root)
    assert problems(root, base="HEAD") == []


def test_an_unknown_base_cannot_be_evaluated(committed):
    with pytest.raises(csr.EvaluationError):
        problems(committed, base="0000000000000000000000000000000000000000")
