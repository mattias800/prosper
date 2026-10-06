"""Self-tests for the static HLE progress census.

Every fixture is hand-built in a registration shape that the first, `HLE()`-regex version of this
tool got wrong (#4532), so each one fails against that version for a reason that changes the headline
number, not on an API difference:

  * the `#define HLE(name)` line was itself counted as a done handler called `name`;
  * a handler registered through a lambda or a file-local wrapper macro was missing from both sides;
  * one `register_placeholder` fold over a `std::index_sequence` was counted as one handler, not as
    one NID per pack element, and a real handler in another file was not seen to override it;
  * an array-driven NID (`kUlt[kIdx].nid`) was invisible.

They do not draw from the parser's own shape list, so they test the domain, not the discriminator.
The last case runs the real tree and requires the literal NID count to equal
`hle_handler_map.py`'s, which is the independent second route to the same number.

Runs under pytest, or directly (`python3 test_progress.py`) as ctest `progress_census` does.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from progress import collect, compare  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

# A minimal stand-in for dispatch.hpp: the parser reads the registration API list from `class Hle`.
DISPATCH_HPP = """
namespace prosper {
using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
class Hle {
public:
    static void  register_fn(const std::string& nid, HleFn fn, const char* name);
    static void  register_placeholder(const std::string& nid, HleFn fn, const char* name);
};
}
"""


def tree(root: Path, sources: dict[str, str]) -> Path:
    """Write a synthetic `prosper/src/hle` of {"area/file.cpp": text} under `root`."""
    hle = root / "prosper" / "src" / "hle"
    (hle / "dispatch").mkdir(parents=True)
    (hle / "dispatch" / "dispatch.hpp").write_text(DISPATCH_HPP)
    for name, body in sources.items():
        (hle / name).parent.mkdir(parents=True, exist_ok=True)
        (hle / name).write_text(body)
    return root


def names(data: dict, state: str) -> list[str]:
    """Labels of every NID in one state, across areas."""
    return sorted(e["name"] for g in data["groups"] for e in g[state + "_nids"])


def test_hle_macro_definition_is_not_a_handler(tmp_path: Path) -> None:
    """`#define HLE(name)` is a macro, not a handler called `name`."""
    tree(
        tmp_path,
        {
            "svc/a.cpp": "#define HLE(name) static uint64_t name(uint64_t a0)\n"
            "HLE(s_ok) { return 0; }\n"
            'void register_svc() { Hle::register_fn("AAAAAAAAAAA", (HleFn)s_ok, "sceSvcOk"); }\n'
        },
    )
    data = collect(tmp_path, "linux")
    assert data["done"] == 1, data
    assert data["total"] == 1, data
    assert names(data, "done") == ["sceSvcOk"], data


def test_lambda_and_wrapper_macro_registrations_count(tmp_path: Path) -> None:
    """Plain functions registered through a lambda and through a wrapper macro are on both sides."""
    tree(
        tmp_path,
        {
            "sync/lam.cpp": "static uint64_t f_a(uint64_t) { return 0; }\n"
            "static uint64_t f_b(uint64_t) { return 0; }\n"
            "void register_lam() {\n"
            "    auto reg = [](const char* s, HleFn f) { Hle::register_fn(nid_hash(s), f, s); };\n"
            '    reg("sceLamA", (HleFn)f_a);\n'
            '    reg("sceLamB", (HleFn)f_b);\n'
            "}\n",
            "util/mac.cpp": "static uint64_t g_a(uint64_t) { return 0; }\n"
            "void register_mac() {\n"
            "    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)\n"
            '    R("sceMacA", g_a);\n'
            "    #undef R\n"
            "}\n",
        },
    )
    data = collect(tmp_path, "linux")
    assert data["done"] == 3, data
    assert data["total"] == 3, data
    assert {g["name"]: g["done"] for g in data["groups"]} == {"sync": 2, "util": 1}, data


INDEX_SEQUENCE_FOLD = """
const char* const kNids[] = {
    "AAAAAAAAAA0", "AAAAAAAAAA1", "AAAAAAAAAA2", "AAAAAAAAAA3", "AAAAAAAAAA4",
};
constexpr size_t kFirst = 3;
constexpr size_t kCount = sizeof(kNids) / sizeof(kNids[0]);
template <size_t I> uint64_t thunk(uint64_t a0) { return 0; }
template <size_t Offset, size_t... Is>
void register_tracers(std::index_sequence<Is...>) {
    (Hle::register_placeholder(kNids[Offset + Is], (HleFn)&thunk<Offset + Is>, kNids[Offset + Is]), ...);
}
void register_gfx() {
    register_tracers<0>(std::make_index_sequence<kFirst>{});
    register_tracers<kFirst>(std::make_index_sequence<kCount - kFirst>{});
}
"""


def test_index_sequence_fold_counts_one_nid_per_element(tmp_path: Path) -> None:
    """One placeholder fold over five NIDs is five NIDs; a real handler elsewhere overrides one."""
    tree(
        tmp_path,
        {
            "graphics/tracers.cpp": INDEX_SEQUENCE_FOLD,
            "graphics/real.cpp": "static uint64_t real(uint64_t) { return 7; }\n"
            'void register_real() { Hle::register_fn("AAAAAAAAAA3", (HleFn)real, "sceReal"); }\n',
        },
    )
    data = collect(tmp_path, "linux")
    assert data["total"] == 5, data
    assert data["done"] == 1, data
    assert data["todo"] == 4, data
    assert names(data, "done") == ["sceReal"], data
    handlers = sorted(e["handler"] for g in data["groups"] for e in g["todo_nids"])
    assert handlers == ["thunk<0>", "thunk<1>", "thunk<2>", "thunk<4>"], data
    assert data["complete"], data


def test_constant_table_struct_field_resolves(tmp_path: Path) -> None:
    """`kTab[kIdx].nid` resolves through the struct layout and the index constants."""
    tree(
        tmp_path,
        {
            "sync/ult.cpp": "struct Entry { const char* nid; const char* name; int ret; };\n"
            "constexpr Entry kTab[] = {\n"
            '    {"BBBBBBBBBB0", "sceTabZero", 0},\n'
            '    {"BBBBBBBBBB1", "sceTabOne", 1},\n'
            "};\n"
            "constexpr size_t kIdxZero = 0, kIdxOne = kIdxZero + 1;\n"
            "static uint64_t t0(uint64_t) { return 0; }\n"
            "static uint64_t t1(uint64_t) { return 0; }\n"
            "void register_tab() {\n"
            "    Hle::register_fn(kTab[kIdxZero].nid, (HleFn)t0, kTab[kIdxZero].name);\n"
            "    Hle::register_fn(kTab[kIdxOne].nid, (HleFn)t1, kTab[kIdxOne].name);\n"
            "}\n"
        },
    )
    data = collect(tmp_path, "linux")
    assert data["done"] == 2, data
    assert data["total"] == 2, data
    assert names(data, "done") == ["sceTabOne", "sceTabZero"], data
    assert data["complete"], data


def test_unresolvable_nid_marks_census_incomplete(tmp_path: Path) -> None:
    """A NID the tool cannot evaluate is reported and makes the census incomplete, not silent."""
    tree(
        tmp_path,
        {
            "svc/a.cpp": "static uint64_t h(uint64_t) { return 0; }\n"
            'void register_svc() { Hle::register_fn(runtime_nid(), (HleFn)h, "sceRuntime"); }\n'
        },
    )
    data = collect(tmp_path, "linux")
    assert data["total"] == 0, data
    assert not data["complete"], data
    assert len(data["coverage"]["unresolved_nid_exprs"]) == 1, data


def test_platform_arm_is_evaluated(tmp_path: Path) -> None:
    """Each platform counts only its own `#if` arm."""
    tree(
        tmp_path,
        {
            "kernel/k.cpp": "static uint64_t w(uint64_t) { return 0; }\n"
            "static uint64_t p(uint64_t) { return 0; }\n"
            "#if defined(_WIN32)\n"
            'void register_k() { Hle::register_fn(nid_hash("sceWinArm"), (HleFn)w, "sceWinArm"); }\n'
            "#else\n"
            'void register_k() { Hle::register_fn(nid_hash("scePosixArm"), (HleFn)p, "scePosixArm"); }\n'
            "#endif\n"
        },
    )
    assert names(collect(tmp_path, "linux"), "done") == ["scePosixArm"]
    assert names(collect(tmp_path, "windows"), "done") == ["sceWinArm"]


def test_compare_reports_implemented(tmp_path: Path) -> None:
    """A NID that moves from placeholder to real is reported as implemented, keyed by NID."""
    base_root, head_root = tmp_path / "base", tmp_path / "head"
    placeholder = (
        "static uint64_t t(uint64_t) { return 0; }\n"
        'void register_g() { Hle::register_placeholder("CCCCCCCCCC0", (HleFn)t, "sceLater"); }\n'
    )
    tree(base_root, {"graphics/g.cpp": placeholder})
    tree(
        head_root,
        {
            "graphics/g.cpp": placeholder,
            "graphics/h.cpp": "static uint64_t r(uint64_t) { return 1; }\n"
            'void register_h() { Hle::register_fn("CCCCCCCCCC0", (HleFn)r, "sceLater"); }\n',
        },
    )
    base, head = collect(base_root, "linux"), collect(head_root, "linux")
    assert (base["todo"], head["done"]) == (1, 1), (base, head)
    report = compare(json.loads(json.dumps(base)), head)
    assert "implemented (1):" in report, report
    assert "graphics::sceLater (CCCCCCCCCC0)" in report, report


def test_real_tree_agrees_with_hle_handler_map() -> None:
    """On the real tree the literal NID count is hle_handler_map's, and nothing is left unresolved."""
    src = REPO / "prosper" / "src" / "hle"
    if not src.is_dir():
        return
    import hle_handler_map as H

    sc = H.scan_tree(str(src), "linux")
    data = collect(REPO, "linux")
    c = data["coverage"]
    assert c["literal_nids"] == len({r.nid for r in sc.regs}), c
    assert data["total"] == c["literal_nids"] + c["table_resolved_new_nids"], c
    assert data["complete"], c
    assert data["done"] + data["todo"] == data["total"], data
    # sync registers sceUlt through a constant table; the HLE()-regex census reported it as 0/0.
    assert {g["name"]: g["done"] for g in data["groups"]}.get("sync", 0) > 0, data


def main() -> int:
    """Run every test without pytest, for ctest."""
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {str(exc)[:300]}")
    print(f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
