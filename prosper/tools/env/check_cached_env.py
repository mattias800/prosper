#!/usr/bin/env python3
"""Refuse to cache a PROSPER_* variable that something arms at RUNTIME.

PROSPER_ENV_ON / PROSPER_ENV_VALUE (src/hle/dispatch.hpp) sample getenv ONCE, at first use, and
return that value forever. That is correct for a boot-time diagnostic switch and wrong for any
variable written after the process starts -- the later write is simply never observed.

The failure mode is what makes this worth a gate rather than a comment. A test that arms a
diagnostic with setenv/_putenv_s and then asserts on the behaviour does not fail when the read is
cached; it goes VACUOUS and keeps printing [ok], because the assertion still holds for the stale
value. #2214 hit this: three variables that test_gpu_capture_render.cpp toggles between phases were
cached, and the only symptom was one unrelated-looking Vulkan binding error on a single CI job.

So this compares two sets over the whole tree:

    cached  -- names appearing in PROSPER_ENV_ON("X") / PROSPER_ENV_VALUE("X")
    armed   -- names appearing in setenv("X" / _putenv_s("X" / unsetenv("X" / putenv("X"

and fails if any name is in both. The check is deliberately CONSERVATIVE: it does not try to prove
the write happens before the read. Ordering arguments are exactly what went wrong in #2214, where
the reasoning was correct for the site being examined and there was a second site nobody looked at.
If a name lands here, make it a live getenv or hoist it -- do not add an exception.

Run standalone against a checkout, or via ctest as cached_env_arming_logic.
"""
import re
import sys
from pathlib import Path

CACHED_RE = re.compile(r'PROSPER_ENV_(?:ON|VALUE)\(\s*"(PROSPER_[A-Z_0-9]+)"\s*\)')

# Any call whose FIRST argument is a PROSPER_* literal. Which of those count as "arming" is decided
# by armed_callee() rather than by a fixed list of libc names -- see why below.
CALL_RE = re.compile(r'\b(\w+)\s*\(\s*"(PROSPER_[A-Z_0-9]+)"')


def armed_callee(name: str) -> bool:
    """True if `name(...)` WRITES the environment.

    Matching only setenv/_putenv_s/unsetenv/putenv is not enough, and this is not hypothetical: it
    is how the first version of this checker passed a tree that still contained the #2214 defect.
    test_gpu_capture_render.cpp arms diagnostics through its own two-line wrappers --

        static void set_env(const char* name, const std::string& v)  { setenv(name, v.c_str(), 1); }
        static void unset_env(const char* name)                      { unsetenv(name); }

    -- so at the CALL site the name is a literal and at the setenv site it is a variable. A pattern
    anchored on the libc name sees neither, reports an empty intersection, and the gate is decorative.

    So: treat any callee containing "env" as a writer, minus the reads. That accepts set_env,
    unset_env, set_environment, setenv, _putenv_s, putenv without enumerating them, and stays
    conservative in the safe direction -- a false positive here costs one live getenv, while a false
    negative costs a silently vacuous test.
    """
    low = name.lower()
    if "env" not in low:
        return False
    if low in ("getenv", "std_getenv"):          # a READ is the fix, not the defect
        return False
    if name.startswith("PROSPER_ENV_"):          # the caching macros themselves
        return False
    return True

SCAN_DIRS = ("src", "frontends", "tools", "tests")
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp")


def scan(root: Path):
    """Return (cached, armed) name -> sorted list of 'relpath:line'."""
    cached, armed = {}, {}
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SCAN_EXT or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.split("\n"), 1):
                where = f"{path.relative_to(root)}:{lineno}"
                for m in CACHED_RE.finditer(line):
                    cached.setdefault(m.group(1), []).append(where)
                for m in CALL_RE.finditer(line):
                    if armed_callee(m.group(1)):
                        armed.setdefault(m.group(2), []).append(where)
    return cached, armed


def scan_line(line: str):
    """The per-line half of scan(), exposed so the self-test exercises the real code path."""
    return ([m.group(1) for m in CACHED_RE.finditer(line)],
            [m.group(2) for m in CALL_RE.finditer(line) if armed_callee(m.group(1))])


# --- self-test -------------------------------------------------------------------------------
# Without these, a regex that stops matching (a macro rename, a reformat that splits the call over
# two lines) makes this gate report a clean tree forever -- the one way it can fail silently, and
# the same failure the vkval scanner's registration comment warns about. Half the cases assert on
# shapes that must NOT match, because a checker that fires on valid code gets deleted, not heeded.
SELF_TESTS = [
    # (snippet, expected cached names, expected armed names)
    ('if (PROSPER_ENV_ON("PROSPER_FOO")) {', ["PROSPER_FOO"], []),
    ('const char* v = PROSPER_ENV_VALUE("PROSPER_BAR");', ["PROSPER_BAR"], []),
    ('PROSPER_ENV_VALUE( "PROSPER_SPACED" )', ["PROSPER_SPACED"], []),
    ('setenv("PROSPER_BAZ", "1", 1);', [], ["PROSPER_BAZ"]),
    ('_putenv_s("PROSPER_QUX", "1");', [], ["PROSPER_QUX"]),
    ('unsetenv("PROSPER_QUUX");', [], ["PROSPER_QUUX"]),
    # The WRAPPER forms. These are the cases the first version of this checker missed, which let it
    # pass a tree that still had the #2214 defect -- test_gpu_capture_render.cpp arms every one of
    # its diagnostics this way. If these three stop matching, the gate is decorative again.
    ('set_env("PROSPER_WRAPPED", dump_dir.string());', [], ["PROSPER_WRAPPED"]),
    ('unset_env("PROSPER_UNWRAPPED");', [], ["PROSPER_UNWRAPPED"]),
    ('set_environment("PROSPER_ENVIRON", "1", false);', [], ["PROSPER_ENVIRON"]),
    # Two on one line must both be seen -- the real tree has these.
    ('a = PROSPER_ENV_VALUE("PROSPER_A") ? PROSPER_ENV_VALUE("PROSPER_B") : 0;',
     ["PROSPER_A", "PROSPER_B"], []),
    # MUST NOT match: a live read is the fix, not the defect.
    ('const char* v = getenv("PROSPER_LIVE");', [], []),
    # MUST NOT match: a non-PROSPER variable is not ours to police.
    ('setenv("SDL_AUDIO_DRIVER", "dummy", 1);', [], []),
    # MUST NOT match: naming the macro in prose does not read a variable.
    ('// PROSPER_ENV_VALUE caches, so do not use it for a runtime-armed name', [], []),
]


def self_test() -> int:
    bad = 0
    for snippet, want_cached, want_armed in SELF_TESTS:
        got_cached, got_armed = scan_line(snippet)
        if got_cached != want_cached or got_armed != want_armed:
            print(f"  [FAIL] self-test: {snippet!r}")
            print(f"         cached want={want_cached} got={got_cached}")
            print(f"         armed  want={want_armed} got={got_armed}")
            bad += 1
    if bad:
        print(f"  the scanner's own patterns are broken -- a tree scan would report a false CLEAN")
    else:
        print(f"  [ok]   scanner self-test: {len(SELF_TESTS)} cases")
    return bad


def main() -> int:
    here = Path(__file__).resolve()
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here.parents[2]
    print("== check_cached_env ==")
    if self_test():
        return 1

    cached, armed = scan(root)
    if not cached:
        # An empty cached set would make the intersection trivially empty and the gate meaningless.
        print(f"  [FAIL] no PROSPER_ENV_ON/VALUE uses found under {root} -- scan is not seeing the tree")
        return 1
    print(f"  [ok]   scanned {root}: {len(cached)} cached name(s), {len(armed)} armed at runtime")

    # Two tiers, because "cached and written somewhere" is not by itself a defect and a gate with
    # standing false positives is one people learn to skip.
    #
    #   FAIL  -- armed from tests/. A test process arms a diagnostic and then reads it back in the
    #            same process, often toggling it BETWEEN phases. The cached read latches phase one's
    #            value, the assertion still holds for it, and the test goes vacuous while printing
    #            [ok]. That is #2214, and it is provably wrong without needing an ordering argument.
    #
    #   NOTE  -- armed anywhere else. Tools set these in main() before the renderer exists, so the
    #            cached read happens afterwards and observes the write. That is fine, but it is fine
    #            BY ORDERING, and ordering is what nobody re-checks when code moves. Listed, not
    #            failed, so a reader can confirm the write still precedes first use.
    def from_tests(wheres):
        return [w for w in wheres if w.startswith("tests/") or "/tests/" in w]

    fails, notes = {}, {}
    for name in sorted(set(cached) & set(armed)):
        (fails if from_tests(armed[name]) else notes)[name] = armed[name]

    for name, where in notes.items():
        print(f"  [note] {name}: cached, and armed outside tests/ -- safe only if the write precedes")
        for w in sorted(set(where))[:2]:
            print(f"           armed at {w}")

    if not fails:
        print(f"  [ok]   no cached variable is armed by a test ({len(notes)} ordering note(s) above)")
        print("== all checks passed ==")
        return 0

    print(f"  [FAIL] {len(fails)} variable(s) are cached AND armed by a test:")
    for name, where in fails.items():
        print(f"    {name}")
        for w in sorted(set(cached[name]))[:4]:
            print(f"        cached at {w}")
        for w in sorted(set(where))[:4]:
            print(f"        armed  at {w}")
    print("  A cached read never observes the test's later write, so the arm becomes vacuous and the")
    print("  assertion still passes. Use a live getenv, or hoist the read out of the hot loop.")
    print(f"== {len(fails)} failure(s) ==")
    return 1


if __name__ == "__main__":
    sys.exit(main())
