"""Tests for confidence_ledger: markers found, non-markers ignored, filters and output modes."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import confidence_ledger as cl  # noqa: E402

REPO = HERE.parents[2]


def build(root: Path) -> Path:
    files = {
        "prosper/src/gpu/a.cpp": (
            "int x;  // CONFIDENCE: LOW -- one trace only\n"
            "/* CONFIDENCE: MED. two titles agree */\n"
            "// CONFIDENCE: HIGH published contract\n"
        ),
        "prosper/src/hle/kernel/b.hpp": "// confidence: low is not a marker\n// CONFIDENCE:LOW nor this\n",
        "prosper/frontends/c.mm": "// CONFIDENCE: LOW mac path\n",
        "prosper/src/notes.md": "CONFIDENCE: LOW in a doc is not counted\n",
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_markers_are_found_with_level_line_and_text(tmp_path):
    found = cl.scan(build(tmp_path))
    assert [(m.path, m.line, m.level, m.text) for m in found] == [
        ("prosper/src/gpu/a.cpp", 1, "LOW", "one trace only"),
        ("prosper/src/gpu/a.cpp", 2, "MED", "two titles agree"),
        ("prosper/src/gpu/a.cpp", 3, "HIGH", "published contract"),
        ("prosper/frontends/c.mm", 1, "LOW", "mac path"),
    ]


def test_default_lists_low_and_med_only(tmp_path, capsys):
    cl.main(["--root", str(build(tmp_path))])
    out = capsys.readouterr().out
    assert "LOW" in out and "MED" in out and "HIGH" not in out


def test_summary_counts_per_area(tmp_path, capsys):
    cl.main(["--root", str(build(tmp_path)), "--summary", "--level", "LOW"])
    out = capsys.readouterr().out.splitlines()
    assert any(line.startswith("src/gpu") and "LOW=1" in line for line in out)
    assert any(line.startswith("frontends") and "LOW=1" in line for line in out)


def test_json_round_trips(tmp_path, capsys):
    cl.main(["--root", str(build(tmp_path)), "--json", "--level", "HIGH"])
    records = json.loads(capsys.readouterr().out)
    assert records == [
        {"path": "prosper/src/gpu/a.cpp", "line": 3, "level": "HIGH", "text": "published contract"}
    ]


def test_the_repository_has_markers():
    assert len(cl.scan(REPO)) > 0
