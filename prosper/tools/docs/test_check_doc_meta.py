"""Tests for check_doc_meta: a hand-built repository per check, plus a clean control."""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_doc_meta as cdm  # noqa: E402

REPO = HERE.parents[2]
FM = "---\nkind: design\nstatus: current\n---\n\n"


def build(root: Path, files: dict[str, str]) -> Path:
    base = {
        "prosper/docs/gpu/A.md": FM
        + "# A\n\nSee [B](B.md) and [web](https://x.y) and [top](#a).\n",
        "prosper/docs/gpu/B.md": FM + "# B\n",
        "prosper/docs/gpu/AGENTS.md": "# map, no frontmatter needed\n",
        "prosper/docs/spec/rules.md": "---\nkind: spec\nstatus: accepted\n---\n",
        "prosper/docs/archive/OLD.md": "---\nkind: archive\nstatus: historical\n---\n",
        "README.md": "[docs](prosper/docs/gpu/A.md#section)\n",
    }
    base.update(files)
    for rel, text in base.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def test_clean_tree_has_no_findings(tmp_path):
    assert cdm.evaluate(build(tmp_path, {})) == []


def test_the_repository_itself_is_clean():
    assert cdm.evaluate(REPO) == []


@pytest.mark.parametrize(
    "files,expect",
    [
        ({"prosper/docs/gpu/B.md": "# B\n"}, "B.md: no frontmatter"),
        ({"prosper/docs/gpu/B.md": FM.replace("design", "essay")}, "kind `essay`"),
        ({"prosper/docs/gpu/B.md": FM.replace("current", "maybe")}, "status `maybe`"),
        ({"prosper/docs/gpu/B.md": FM.replace("design", "archive")}, "go together"),
        ({"prosper/docs/archive/OLD.md": FM}, "go together"),
        (
            {"prosper/docs/gpu/B.md": FM.replace("current", "superseded")},
            "names `superseded-by",
        ),
        (
            {
                "prosper/docs/gpu/B.md": FM.replace(
                    "status: current", "status: superseded\nsuperseded-by: prosper/docs/NO.md"
                )
            },
            "superseded-by prosper/docs/NO.md does not exist",
        ),
        (
            {"prosper/docs/gpu/B.md": FM.replace("---\n\n", "template: design\n---\n\n## Scope\n")},
            "sections missing: Current state",
        ),
        (
            {"prosper/docs/gpu/B.md": FM.replace("---\n\n", "template: fancy\n---\n\n")},
            "unknown template",
        ),
        (
            {"prosper/docs/gpu/B.md": FM + "![shot](../screenshots/a.png)\n"},
            "broken link `../screenshots/a.png`",
        ),
        ({"README.md": "[x](prosper/docs/missing.md)\n"}, "README.md: broken link"),
    ],
)
def test_each_violation_is_reported(tmp_path, files, expect):
    found = cdm.evaluate(build(tmp_path, files))
    assert any(expect in p for p in found), found


def test_code_is_not_a_link(tmp_path):
    body = FM + "`vtbl[0](this)` and\n```\n[x](nowhere.md)\n```\n"
    assert cdm.evaluate(build(tmp_path, {"prosper/docs/gpu/B.md": body})) == []


def test_a_complete_design_template_passes(tmp_path):
    heads = "".join(f"## {s}\n\nnone\n\n" for s in cdm.DESIGN_SECTIONS)
    body = FM.replace("---\n\n", "template: design\n---\n\n") + heads
    assert cdm.evaluate(build(tmp_path, {"prosper/docs/gpu/B.md": body})) == []


def test_the_generated_tracker_is_not_link_checked(tmp_path):
    assert cdm.evaluate(build(tmp_path, {"PROGRESS_TRACKER.md": "[x](gone.png)\n"})) == []


def test_crlf_frontmatter_is_read(tmp_path):
    root = build(tmp_path, {})
    (root / "prosper/docs/gpu/B.md").write_bytes(FM.replace("\n", "\r\n").encode())
    assert cdm.evaluate(root) == []


def test_a_tree_without_docs_cannot_be_evaluated(tmp_path):
    with pytest.raises(cdm.EvaluationError):
        cdm.evaluate(tmp_path)
