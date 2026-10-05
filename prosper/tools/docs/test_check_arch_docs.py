"""Tests for check_arch_docs: a hand-built tree per check, each violating exactly one rule.

Every check has a positive instance written here rather than drawn from the repository, so a matcher
that quietly stops matching fails a test instead of certifying every spec forever. The clean tree
is the negative control: it must produce no findings, or the positives prove nothing.
"""

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_arch_docs as cad  # noqa: E402

REPO = HERE.parents[2]

RATCHET_STUB = 'RULES = ("r1", "r2")\nLAYER_ORDER = ("low", "high")\n'
ALARMS_STUB = (
    "const std::vector<const char*>& rule_names() {\n"
    '    static const std::vector<const char*> names = {"a1", "a2", "a3", "a4", "gpu-sync-wait"};\n'
    "    return names;\n}\n"
)

SPEC_RULES = """---
kind: spec
status: accepted
owner: area:infra
last-verified: 2026-10-05 abcdef01
---

# Rules

### TST-1 -- first rule

Text.
Status: accepted
Enforcement: ratchet:r1, runtime:gpu-sync-wait

### TST-2 -- second rule

Text.
Status: proposed (adr:0001)
Enforcement: ratchet:r2, review: (a reason no tool can check it)
"""

ADR = """---
kind: adr
status: accepted
date: 2026-10-05
---

# ADR 0001: Decide a thing

## Decision

The decision.
"""


def layers_doc() -> str:
    return (
        "---\nkind: spec\nstatus: accepted\nowner: area:infra\n"
        "last-verified: 2026-10-05 abcdef01\n---\n\n# Layers\n\n"
        + cad.render_layer_block(("low", "high"))
        + "\n"
    )


def build(root: Path, **overrides: str | None) -> Path:
    files = {
        "prosper/tools/ci/check_arch_ratchet.py": RATCHET_STUB,
        "prosper/src/diagnostics/perf/perf_alarm_rules.cpp": ALARMS_STUB,
        "prosper/docs/spec/layers.md": layers_doc(),
        "prosper/docs/spec/rules.md": SPEC_RULES,
        "prosper/docs/spec/AGENTS.md": "# no frontmatter needed here\n",
        "prosper/docs/adr/0001-decide-a-thing.md": ADR,
        "prosper/docs/adr/AGENTS.md": "# map\n",
    }
    files.update(overrides)
    for rel, text in files.items():
        if text is None:
            continue
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def checks(root: Path, **kw) -> list[str]:
    return [f"{f.check}: {f.message}" for f in cad.evaluate(root, **kw)]


def test_clean_tree_has_no_findings(tmp_path):
    assert checks(build(tmp_path)) == []


def test_the_repository_itself_is_clean():
    assert checks(REPO) == []


@pytest.mark.parametrize(
    "old,new,expect",
    [
        ("---\nkind: spec\n", "", "no leading --- frontmatter"),
        ("kind: spec\n", "kind: adr\n", "kind `adr` in the spec/ folder"),
        ("status: accepted\nowner", "status: final\nowner", "status `final`"),
        ("owner: area:infra\n", "", "missing required key `owner`"),
        ("### TST-2 -- second rule", "### TST-1 -- second rule", "TST-1 also defined"),
        (
            "Status: accepted\nEnforcement: ratchet:r1",
            "Enforcement: ratchet:r1",
            "TST-1: no `Status:` line",
        ),
        (
            "Enforcement: ratchet:r1, runtime:gpu-sync-wait\n",
            "\n",
            "TST-1: no `Enforcement:` line",
        ),
        ("Status: proposed (adr:0001)", "Status: proposed", "names its ADR"),
        ("Status: proposed (adr:0001)", "Status: maybe", "Status is `accepted` or"),
        ("Status: proposed (adr:0001)", "Status: proposed (adr:0042)", "adr:0042 does not exist"),
        ("Enforcement: ratchet:r1", "Enforcement: ratchet:nope, ratchet:r1", "ratchet:nope is not"),
        (
            "Enforcement: ratchet:r1, runtime:gpu-sync-wait",
            "Enforcement: a careful reader",
            "names no ratchet:",
        ),
        ("runtime:gpu-sync-wait", "runtime:gpu-sync-wiat", "runtime:gpu-sync-wiat is not"),
        (
            "Enforcement: ratchet:r1",
            "Enforcement: adr:0042, ratchet:r1",
            "TST-1: adr:0042 does not",
        ),
        ("review: (a reason no tool can check it)", "review:", "must say in parentheses"),
        ("Enforcement: ratchet:r1", "Enforcement: ctest:", "ctest: needs a name"),
        ("### TST-1 -- first rule", "### TST-1 first rule", "heading must read"),
    ],
)
def test_each_spec_violation_is_reported(tmp_path, old, new, expect):
    assert old in SPEC_RULES
    root = build(tmp_path, **{"prosper/docs/spec/rules.md": SPEC_RULES.replace(old, new, 1)})
    found = checks(root)
    assert any(expect in f for f in found), found


@pytest.mark.parametrize("base", ["-oops", "--output=x", "HEAD;rm", ""])
def test_a_base_that_is_not_a_revision_cannot_be_evaluated(tmp_path, base):
    with pytest.raises(cad.EvaluationError):
        cad.evaluate(build(tmp_path), base=base)


def test_unreadable_alarm_rules_cannot_be_evaluated(tmp_path):
    root = build(tmp_path, **{"prosper/src/diagnostics/perf/perf_alarm_rules.cpp": "// moved\n"})
    with pytest.raises(cad.EvaluationError):
        cad.evaluate(root)


def test_an_uncited_ratchet_rule_is_reported(tmp_path):
    root = build(tmp_path, **{"prosper/docs/spec/rules.md": SPEC_RULES.replace("ratchet:r2, ", "")})
    assert "coverage: ratchet rule `r2` is cited by no spec rule" in checks(root)


@pytest.mark.parametrize(
    "name,text,expect",
    [
        ("0002-Bad_Name.md", ADR.replace("0001", "0002"), "not NNNN-slug.md"),
        ("0002-other.md", ADR, "heading says ADR 0001, file name says 0002"),
        ("0001-again.md", ADR, "ADR number 0001 also used"),
        (
            "0002-gone.md",
            ADR.replace("0001", "0002").replace("accepted", "superseded"),
            "needs `superseded-by: NNNN`",
        ),
        (
            "0002-gone.md",
            ADR.replace("0001", "0002").replace("accepted", "superseded\nsuperseded-by: 0009"),
            "superseded-by 0009 does not exist",
        ),
        (
            "0002-no-title.md",
            ADR.replace("# ADR 0001: Decide a thing", "# Decide"),
            "no `# ADR NNNN",
        ),
    ],
)
def test_each_adr_violation_is_reported(tmp_path, name, text, expect):
    root = build(tmp_path, **{f"prosper/docs/adr/{name}": text})
    found = checks(root)
    assert any(expect in f for f in found), found


def test_layer_table_drift_is_reported_and_write_repairs_it(tmp_path):
    root = build(
        tmp_path,
        **{
            "prosper/tools/ci/check_arch_ratchet.py": 'RULES = ("r1", "r2")\nLAYER_ORDER = ("low", "mid", "high")\n'
        },
    )
    assert any("layer-table" in f for f in checks(root))
    assert checks(root, write=True) == []
    text = (root / "prosper/docs/spec/layers.md").read_text(encoding="utf-8")
    assert "| 1 | `mid` | itself and `low` |" in text
    assert checks(root) == []


def test_missing_ratchet_cannot_be_evaluated(tmp_path):
    root = build(tmp_path, **{"prosper/tools/ci/check_arch_ratchet.py": None})
    with pytest.raises(cad.EvaluationError):
        cad.evaluate(root)
    assert cad.main(["--root", str(root)]) == cad.EXIT_UNEVALUATED


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def committed(tmp_path):
    root = build(tmp_path)
    git(root, "init", "-q")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return root


ADR_PATH = "prosper/docs/adr/0001-decide-a-thing.md"


def test_accepted_adr_body_is_frozen(committed):
    (committed / ADR_PATH).write_text(
        ADR.replace("The decision.", "A new decision."), encoding="utf-8"
    )
    assert any("adr-immutable" in f for f in checks(committed, base="HEAD"))


def test_accepted_adr_status_may_change(committed):
    (committed / "prosper/docs/adr/0002-next.md").write_text(
        ADR.replace("0001: Decide", "0002: Decide"), encoding="utf-8"
    )
    (committed / ADR_PATH).write_text(
        ADR.replace("status: accepted", "status: superseded\nsuperseded-by: 0002"), encoding="utf-8"
    )
    assert checks(committed, base="HEAD") == []


def test_accepted_adr_may_not_be_deleted(committed):
    (committed / ADR_PATH).unlink()
    assert any("was deleted" in f for f in checks(committed, base="HEAD"))


def test_proposed_adr_may_be_edited(committed):
    proposed = ADR.replace("status: accepted", "status: proposed")
    (committed / ADR_PATH).write_text(proposed, encoding="utf-8")
    git(committed, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "proposed")
    (committed / ADR_PATH).write_text(
        proposed.replace("The decision.", "Revised."), encoding="utf-8"
    )
    assert checks(committed, base="HEAD") == []
