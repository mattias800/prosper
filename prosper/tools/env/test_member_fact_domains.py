"""Domain admission/refusal and real scanner visibility controls, run by --selftest."""
from __future__ import annotations

import tempfile
from pathlib import Path

from member_fact_domains import PROPERTY

DECLARATION = "add_executable(test_member_domain tests/local.cpp)\n"
TEST = "add_test(NAME member_domain COMMAND test_member_domain)\n"
MARKER = f"set_property(TARGET test_member_domain PROPERTY {PROPERTY} TEST_LOCAL)\n"
CMAKE = DECLARATION + TEST + MARKER

PRODUCTION = """
struct Context { const bool& detail; };
void production_report(const Context& ctx) {
    auto& detail = ctx.detail;
    if (detail) {
        if (getenv("PROSPER_DOMAIN_SECOND")) fprintf(stderr, "production\\n");
    }
}
void production_bind() {
    static const bool detail = getenv("PROSPER_DOMAIN_FIRST") != nullptr;
    production_report(Context{
        .detail = detail,
    });
}
"""
VALUE = "struct Owned { bool detail; };\n"
UNKNOWN = """
void preset() {
    Context ctx{
        .detail = false,
    };
}
"""
LOCAL = """
struct LocalContext { const bool& local_detail; };
void local_report(const LocalContext& ctx) {
    auto& local_detail = ctx.local_detail;
    if (local_detail) {
        if (getenv("PROSPER_LOCAL_SECOND")) fprintf(stderr, "local\\n");
    }
}
void local_bind() {
    static const bool local_detail = getenv("PROSPER_LOCAL_FIRST") != nullptr;
    local_report(LocalContext{
        .local_detail = local_detail,
    });
}
void local_direct() {
    unsigned hits = 0;
    if (getenv("PROSPER_LOCAL_PRODUCE")) {
        hits = 7;
    }
    fprintf(stderr, "hits=%u\\n", hits);
    if (getenv("PROSPER_LOCAL_MARKER")) fprintf(stderr, "inventory\\n");
}
"""
PRODUCTION_KEY = "TWO-GATE|src/production.cpp|report|PROSPER_DOMAIN_FIRST+PROSPER_DOMAIN_SECOND"
LOCAL_KEY = "TWO-GATE|tests/local.cpp|report|PROSPER_LOCAL_FIRST+PROSPER_LOCAL_SECOND"
DIRECT_KEY = "SPLIT-LOCAL|tests/local.cpp|hits|PROSPER_LOCAL_PRODUCE"


def run_tests(verbose: bool = False) -> int:
    # Import only after the scanner's definitions exist; the CLI also executes as __main__.
    import check_diag_gates as scanner

    bad, count = 0, 0

    def check(label, condition, detail=""):
        nonlocal bad, count
        count += 1
        if not condition:
            bad += 1
            print(f"  [FAIL] member-domain: {label}: {detail}")
        elif verbose:
            print(f"  [ok]   member-domain: {label}")

    def tree(root, cmake=CMAKE, local=LOCAL, additions=None):
        contents = {"CMakeLists.txt": cmake, "src/production.cpp": PRODUCTION,
                    "tests/local.cpp": local}
        contents.update(additions or {})
        for name, content in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def scan(root, enabled=True):
        roles = []
        files, _preds, findings = scanner.scan_tree(root, follow_test_domains=enabled,
                                                   domain_roles=roles)
        return files, {f.key() for f in findings}, roles

    # Independent collision arms require BOTH fact kinds to be partitioned. Bridge-off and
    # absent-marker arms calibrate the production observation rather than only the role parser.
    for label, collision in (("owning-value", VALUE), ("unknown-initializer", UNKNOWN)):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree(root, local=LOCAL + collision)
            files, keys, roles = scan(root)
            check(f"{label} preserves exact production report", PRODUCTION_KEY in keys, sorted(keys))
            check(f"{label} local context and producer/printer remain visible",
                  LOCAL_KEY in keys and DIRECT_KEY in keys, sorted(keys))
            check(f"{label} inventory and file population retained",
                  len(files) == 2 and "PROSPER_LOCAL_MARKER" in scanner.inventory(files))
            check(f"{label} one accepted role is auditable",
                  len(roles) == 1 and roles[0].accepted and roles[0].source == "tests/local.cpp")
            _f, disabled, _r = scan(root, False)
            check(f"{label} domain-disabled control loses production report",
                  PRODUCTION_KEY not in disabled and LOCAL_KEY in disabled)
            (root / "CMakeLists.txt").write_text(DECLARATION + TEST, encoding="utf-8")
            _f, absent, absent_roles = scan(root)
            check(f"{label} marker-absent control retains shared collision",
                  PRODUCTION_KEY not in absent and not absent_roles)

    # A shared production unknown/value still retires the member; headers are never isolated.
    for where in ("src/collision.cpp", "tests/fixtures/render_runner.h"):
        for label, collision in (("value", VALUE), ("unknown", UNKNOWN)):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                tree(root, additions={where: collision})
                files, keys, roles = scan(root)
                check(f"shared {where} {label} retires production member",
                      PRODUCTION_KEY not in keys and LOCAL_KEY in keys and len(files) == 3
                      and roles[0].accepted, sorted(keys))

    refusal_cases = [
        ("duplicate-marker", CMAKE + MARKER, {}, "duplicate marker"),
        ("unsupported-value", CMAKE.replace("TEST_LOCAL", "OTHER"), {}, "unsupported marker"),
        ("quoted-marker", CMAKE.replace("TEST_LOCAL", '"TEST_LOCAL"'), {}, "unsupported marker"),
        ("unterminated-marker", CMAKE.rstrip()[:-1], {}, "unsupported marker"),
        ("unsupported-command", DECLARATION + TEST +
         f"set_target_properties(test_member_domain PROPERTIES {PROPERTY} TEST_LOCAL)\n",
         {}, "unsupported marker"),
        ("stale-target", TEST + MARKER, {}, "missing or duplicate target"),
        ("missing-source", CMAKE.replace("tests/local.cpp", "tests/missing.cpp"), {}, "existing collected"),
        ("missing-test", DECLARATION + MARKER, {}, "direct literal add_test"),
        ("duplicate-test", CMAKE + TEST, {}, "direct literal add_test"),
        ("multiple-sources", CMAKE.replace("tests/local.cpp)", "tests/local.cpp src/production.cpp)"),
         {}, "one literal executable source"),
        ("variable-source", CMAKE.replace("tests/local.cpp", "${SOURCES}"), {}, "existing collected"),
        ("glob-source", CMAKE.replace("tests/local.cpp", "tests/*.cpp"), {}, "existing collected"),
        ("header-source", CMAKE.replace("tests/local.cpp", "tests/fixtures/render_runner.h"),
         {"tests/fixtures/render_runner.h": VALUE}, "tests/*.cpp"),
        ("extra-target-sources", CMAKE + "target_sources(test_member_domain PRIVATE ${EXTRA})\n",
         {}, "target_sources"),
        ("literal-source-reuse", CMAKE + "add_library(shipping STATIC tests/local.cpp)\n",
         {}, "add_library"),
        ("canonical-source-reuse", CMAKE + "add_library(shipping STATIC tests/../tests/local.cpp)\n",
         {}, "add_library"),
        ("relative-scope-source-reuse", CMAKE,
         {"cmake/reuse.cmake": "add_library(shipping STATIC ../tests/local.cpp)\n"}, "add_library"),
        ("genex-source-reuse", CMAKE + 'target_sources(shipping PRIVATE "$<$<BOOL:X>:tests/local.cpp>")\n',
         {}, "target_sources"),
        ("target-alias", CMAKE + "add_executable(alias ALIAS test_member_domain)\n", {}, "add_executable"),
        ("inbound-link", CMAKE + "target_link_libraries(shipping PRIVATE test_member_domain)\n",
         {}, "target_link_libraries"),
        ("export", CMAKE + "export(TARGETS test_member_domain FILE targets.cmake)\n", {}, "export"),
        ("source-variable", CMAKE + "set(SOURCES tests/local.cpp)\n", {}, "source construction"),
        ("source-glob", CMAKE + "file(GLOB_RECURSE SOURCES tests/*.cpp)\n", {}, "literal glob"),
        ("source-glob-partial-name", CMAKE + "file(GLOB SOURCES tests/loc?l.cpp)\n", {}, "literal glob"),
        ("source-glob-list", CMAKE + 'file(GLOB SOURCES "src/*.cpp;tests/local*.cpp")\n', {}, "literal glob"),
        ("unknown-role-command", CMAKE + "opaque_build(test_member_domain)\n",
         {}, "unsupported role-affecting command"),
        ("cpp-source-include", CMAKE, {"src/include.cpp": '#include "../tests/local.cpp"\n'}, "CPP inclusion"),
        ("cpp-header-include", CMAKE,
         {"tests/fixtures/bridge.h": '#include "../local.cpp"\n'}, "CPP inclusion"),
        ("cpp-inc-include", CMAKE,
         {"src/bridge.inc": '#include "../tests/local.cpp"\n'}, "CPP inclusion"),
        ("definition-marker", DECLARATION + TEST + "function(register_role)\n" + MARKER + "endfunction()\n",
         {}, "definition scope"),
        ("opaque-role-eval", CMAKE + 'cmake_language(EVAL CODE "target_sources(test_member_domain PRIVATE x.cpp)")\n',
         {}, "opaque role modification"),
    ]
    for label, cmake, additions, reason in refusal_cases:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree(root, cmake, LOCAL + VALUE, additions)
            files, keys, roles = scan(root)
            check(f"refuse {label} with explicit reason and retain local file/report/inventory",
                  bool(roles) and all(not role.accepted for role in roles)
                  and any(reason in role.reason for role in roles)
                  and root / "tests/local.cpp" in files and LOCAL_KEY in keys and DIRECT_KEY in keys
                  and PRODUCTION_KEY not in keys
                  and "PROSPER_LOCAL_MARKER" in scanner.inventory(files), [r.report() for r in roles])

    for label, hidden in (("line-comment", "# " + MARKER),
                          ("bracket-comment", "#[=[\n" + MARKER + "]=]\n"),
                          ("bracket-string", "set(payload [=[\n" + MARKER + "]=])\n"),
                          ("quoted-payload", 'message("' + MARKER.strip().replace('"', '\\"') + '")\n')):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree(root, DECLARATION + TEST + hidden, LOCAL + VALUE)
            files, keys, roles = scan(root)
            check(f"hidden {label} cannot declare a role; file/report/inventory retained",
                  not roles and len(files) == 2 and PRODUCTION_KEY not in keys
                  and LOCAL_KEY in keys and "PROSPER_LOCAL_MARKER" in scanner.inventory(files))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        tree(root, CMAKE.replace("tests/local.cpp", "tests/alias.cpp"), LOCAL + VALUE)
        (root / "tests/alias.cpp").symlink_to("local.cpp")
        files, _keys, roles = scan(root)
        check("symlink source refuses isolation while both file identities are collected",
              len(files) == 3 and roles and not roles[0].accepted and "without symlinks" in roles[0].reason)

    # A root reached through a symlink must behave exactly like its canonical spelling (#4043).
    # macOS gets this for free -- every temporary directory is /var -> /private/var -- but the arm
    # builds the symlink BY HAND so it fails on any host, not only on the one that exposed it.
    with tempfile.TemporaryDirectory() as td:
        real, link = Path(td) / "real", Path(td) / "link"
        real.mkdir()
        link.symlink_to(real, target_is_directory=True)
        tree(real, local=LOCAL + VALUE)
        _f, canonical_keys, _r = scan(real.resolve())
        files, keys, roles = scan(link)
        check("symlinked root: role accepted and production report identical to canonical root",
              len(roles) == 1 and roles[0].accepted and roles[0].source == "tests/local.cpp"
              and PRODUCTION_KEY in keys and keys == canonical_keys
              and link / "tests/local.cpp" in files, sorted(keys))
        (real / "src/include.cpp").write_text('#include "../tests/local.cpp"\n', encoding="utf-8")
        try:
            _f, _k, roles = scan(link)
            refused = (bool(roles) and not roles[0].accepted
                       and "literal CPP inclusion: src/include.cpp" in roles[0].reason)
        except ValueError as exc:  # the pre-fix shape: relative_to across the two spellings
            refused, roles = False, [exc]
        check("symlinked root: CPP-inclusion refusal names the file instead of raising", refused,
              roles)

    if not bad:
        print(f"  [ok]   member-fact domains: {count} controls; declarations/writes, "
              "bridge-off, shared retirement, local visibility and literal refusal arms")
    return bad


if __name__ == "__main__":
    raise SystemExit(run_tests(True))
