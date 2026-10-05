"""Self-tests for the static HLE progress census.

Pins the done/todo contract: a real HLE body counts as done, a body calling
prosper_on_unimpl or a placeholder-only registration counts as todo, and a real
body anywhere beats a placeholder anywhere. The placeholder fixture doubles as the
positive control a clean zero must be checked against.
"""

from __future__ import annotations

import json
from pathlib import Path

from progress import collect, compare


def write_tree(root: Path) -> None:
    """Hand-built hle/ tree: one real handler, one stub, one placeholder-only."""
    area = root / "prosper" / "src" / "hle" / "svc"
    area.mkdir(parents=True)
    (area / "a.cpp").write_text(
        "uint64_t HLE(s_ok)(uint64_t a0) { return 0; }\n"
        'void reg() { Hle::register_fn("AAAAAAAAAAAAAAAAAAAA" + 11, (HleFn)s_ok, "s_ok"); }\n'
    )
    (area / "b.cpp").write_text(
        "uint64_t HLE(s_stub)(uint64_t a0) { return prosper_on_unimpl(1); }\n"
    )
    (area / "c.cpp").write_text(
        "uint64_t glog_thunk_0(uint64_t a0) { return 0; }\n"
        'void reg() { Hle::register_placeholder("BBBBBBBBBBBB", '
        '(HleFn)glog_thunk_0, "nid"); }\n'
    )


def test_counts_done_and_todo(tmp_path: Path) -> None:
    """One real handler is done; the stub body and placeholder thunk are todo."""
    write_tree(tmp_path)
    data = collect(tmp_path)
    assert data["done"] == 1, data
    assert data["total"] == 3, data
    (group,) = data["groups"]
    assert group["done_names"] == ["s_ok"], data
    assert sorted(group["todo_names"]) == ["glog_thunk_0", "s_stub"], data


def test_placeholder_detected() -> None:
    """Positive control: a placeholder-only tree never reports todo=0."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        area = root / "prosper" / "src" / "hle" / "svc"
        area.mkdir(parents=True)
        (area / "c.cpp").write_text(
            "uint64_t glog_thunk_0(uint64_t a0) { return 0; }\n"
            'void reg() { Hle::register_placeholder("BBBBBBBBBBBB", '
            '(HleFn)glog_thunk_0, "nid"); }\n'
        )
        data = collect(root)
    assert data["todo"] == 1, data
    assert data["done"] == 0, data


def test_real_body_beats_placeholder(tmp_path: Path) -> None:
    """Same handler placeholder-registered in one file, implemented in another."""
    area = tmp_path / "prosper" / "src" / "hle" / "svc"
    area.mkdir(parents=True)
    (area / "a.cpp").write_text("uint64_t HLE(s_both)(uint64_t a0) { return 1; }\n")
    (area / "b.cpp").write_text(
        'void reg() { Hle::register_placeholder("CCCCCCCCCCCC", (HleFn)s_both, "nid"); }\n'
    )
    data = collect(tmp_path)
    assert data["done"] == 1, data
    assert data["todo"] == 0, data


def test_compare_reports_implemented(tmp_path: Path) -> None:
    """Diffing two JSON files names the newly implemented handler."""
    write_tree(tmp_path)
    head = collect(tmp_path)
    base = json.loads(json.dumps(head))
    for group in base["groups"]:
        if "s_ok" in group["done_names"]:
            group["done_names"].remove("s_ok")
            group["todo_names"].append("s_ok")
            group["done"] -= 1
            group["todo"] += 1
    base["done"] -= 1
    base["percent"] = round(100 * base["done"] / base["total"], 2)
    report = compare(base, head)
    assert "svc::s_ok" in report, report
    assert "implemented" in report, report
