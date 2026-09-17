#!/usr/bin/env python3
"""Refuse to cache a PROSPER_* variable that something arms at RUNTIME.

PROSPER_ENV_ON / PROSPER_ENV_VALUE (src/hle/dispatch/dispatch.hpp) sample getenv ONCE, at first use, and
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

# PROSPER_ENV_ON_PER_SUBMIT (src/diagnostics/env_submit.hpp) is a THIRD visibility contract, and it
# must not be confused with either of the other two. It re-samples once per submit inside a
# SubmitEnvScope and reads live outside one, so unlike PROSPER_ENV_ON it is perfectly legal for a
# per-submit name to be armed at runtime -- a test that arms between two submits still sees its
# arm. CACHED_RE above deliberately does not match it (the `(` it requires after ON is a `_` here),
# and armed_callee() deliberately does not count it as a write; both are pinned by self-tests
# below, because the accident that makes this work today is a regex boundary rather than a
# decision, and widening either pattern would silently start failing valid code.
PER_SUBMIT_RE = re.compile(r'PROSPER_ENV_ON_PER_SUBMIT\(\s*"(PROSPER_[A-Z_0-9]+)"\s*\)')

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
    if low in ENV_READERS:                       # a READ is the fix, not the defect
        return False
    if name.startswith("PROSPER_ENV_"):          # the caching macros themselves
        return False
    return True


# Callees that contain "env" and READ rather than write. Deliberately an explicit list rather than a
# looser pattern: the heuristic above is conservative on purpose, and the cost of widening it wrongly
# is a silently vacuous test, which is the entire defect class this gate exists for.
#
# `env_u64_or_default` (diagnostics/env_numeric.hpp) takes the variable's NAME as a literal first
# argument purely so a refusal can name it, and the value as the second -- so a caller passing a
# literal name is reading, not arming, and `test_write_watch_policy.cpp` does exactly that with three
# names production caches (#3253).
ENV_READERS = frozenset((
    "getenv", "std_getenv",
    "env_u64_or_default", "env_u64_or_default_capped",
    # The rest of env_numeric.hpp's entry points, on the same argument: each takes the variable's
    # NAME as a literal first argument purely so a refusal can name it. The first four were listed
    # when they were the only ones; the _auto family, env_u64_or_report and env_tristate_or_unset
    # (#3304) were not, and the heuristic above counts any callee containing "env" as a WRITE.
    "env_u64_or_default_auto", "env_u64_or_default_auto_capped",
    "env_u64_or_report", "env_tristate_or_unset",
))

SCAN_DIRS = ("src", "frontends", "tools", "tests")
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp")


def scan(root: Path):
    """Return (cached, armed, per_submit) name -> sorted list of 'relpath:line'."""
    cached, armed, per_submit = {}, {}, {}
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
                for m in PER_SUBMIT_RE.finditer(line):
                    per_submit.setdefault(m.group(1), []).append(where)
                for m in CALL_RE.finditer(line):
                    if armed_callee(m.group(1)):
                        armed.setdefault(m.group(2), []).append(where)
    return cached, armed, per_submit


# --- tier 3: the hot sites must not REGROW a live getenv ----------------------------------------
#
# #3094 took getenv off the per-draw and per-resource paths of the three files below. Nothing about
# a bare getenv looks wrong at the call site, so the only thing that noticed the cost in the first
# place was a perf profile -- which means the next edit to any of these guards can silently put it
# back and no build, test or review would go red. This table is the standing statement that these
# particular reads are hot, so that reintroducing getenv for one of them fails here instead.
#
# Adding a name is cheap; removing one needs a reason. A name here must ALSO survive the tier-1 and
# tier-2 checks above -- being hot is never a licence to cache something that is armed at runtime.
HOT_SITES = {
    "tests/fixtures/render_runner.h": [
        "PROSPER_BACKEND_LOAD_LOG", "PROSPER_GEOM_PROBE", "PROSPER_DSLOG",
        "PROSPER_RENDER_DROP_UNPROVEN_DRAW", "PROSPER_DRAW_STATS", "PROSPER_DEPTH_CLEAR",
        "PROSPER_DEPTH_CLEAR_WHY", "PROSPER_PIPELOG", "PROSPER_BACKEND_TRACE",
        "PROSPER_NO_BACKEND_BUFFER_ARENA", "PROSPER_NO_BACKEND_PERSISTENT_TEXTURE_BINDINGS",
        "PROSPER_DS_SLICE_CENSUS", "PROSPER_STENCIL_MIRROR", "PROSPER_STENCILLOG",
        "PROSPER_SHADER_TAP", "PROSPER_GEOM_PROBE_DUMP", "PROSPER_NO_DEPTH_BIAS",
        "PROSPER_NO_CULL", "PROSPER_FLIP_FRONT_FACE",
        "PROSPER_NO_BLEND", "PROSPER_STENCIL_REPLACE", "PROSPER_NO_SWIZZLE",
    ],
    "src/gpu/execute/gpu_execute.hpp": [
        "PROSPER_FORCE_COLORWRITE", "PROSPER_NO_EARLY_NO_EFFECT", "PROSPER_DRAWDIAG",
        "PROSPER_DRAWLOG", "PROSPER_EXECLOG", "PROSPER_RECTLOG", "PROSPER_RTLOG",
        "PROSPER_INTERPLOG", "PROSPER_GEOM_PROBE", "PROSPER_ONLY_ATLAS", "PROSPER_ONLY_IC",
        "PROSPER_CAPTION_DIAG", "PROSPER_PROLOGLOG", "PROSPER_VS_DUMP", "PROSPER_SHADER_DUMP",
        "PROSPER_DRAWMAP",
    ],
    "frontends/shared/live/live_renderer.cpp": [
        "PROSPER_DCCLOG", "PROSPER_SLICESTRIDE", "PROSPER_UNIFORMLOG",
    ],
}


def live_getenv_names(line: str):
    """PROSPER_* names this line reads with a LIVE getenv.

    A line carrying `static` is excluded: `static const bool x = getenv("N") != nullptr;` is a
    one-shot read already -- the older hand-rolled spelling of the same fix, still present in the
    tree and not something this gate should push people off.
    """
    if "static" in line:
        return []
    return LIVE_GETENV_RE.findall(line)


LIVE_GETENV_RE = re.compile(r'(?:std::)?getenv\s*\(\s*"(PROSPER_[A-Z_0-9]+)"\s*\)')
CACHED_USE_RE = re.compile(r'PROSPER_ENV_(?:ON|VALUE)\(\s*"(PROSPER_[A-Z_0-9]+)"\s*\)')


def check_hot_sites(root) -> int:
    """Every HOT_SITES name must be read through PROSPER_ENV_* and never through a live getenv."""
    bad = 0
    for relpath, names in sorted(HOT_SITES.items()):
        path = root / relpath
        if not path.is_file():
            print(f"  [FAIL] hot-site file missing: {relpath}")
            print( "         the guard cannot see the sites it is meant to protect -- if the file")
            print( "         moved, move this entry with it rather than deleting it")
            bad += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        cached_here = set(CACHED_USE_RE.findall(text))
        live_here = {}
        for lineno, line in enumerate(text.split("\n"), 1):
            for nm in live_getenv_names(line):
                live_here.setdefault(nm, []).append(lineno)
        for name in sorted(names):
            if name in live_here:
                print(f"  [FAIL] {relpath}: {name} is read with a live getenv at "
                      f"line(s) {', '.join(str(n) for n in live_here[name][:4])}")
                bad += 1
            elif name not in cached_here:
                print(f"  [FAIL] {relpath}: {name} is in HOT_SITES but the file reads it neither")
                print( "         cached nor live -- the guard is stale, so fix or drop the entry")
                bad += 1
    if not bad:
        total = sum(len(v) for v in HOT_SITES.values())
        print(f"  [ok]   {total} hot site(s) across {len(HOT_SITES)} file(s) still read cached")
    return bad


# --- tier 4: one name, one visibility contract --------------------------------------------------
#
# A switch read through PROSPER_ENV_ON at one site and PROSPER_ENV_ON_PER_SUBMIT at another answers
# the same question two different ways, and which answer you get depends on which call site the
# guest happened to reach. That is the shape of the #1873 SKU divergence, in a diagnostic: arm the
# switch mid-run and half the renderer notices. It is also the likeliest way this macro gets
# misused, because converting one hot site and leaving the others alone looks like exactly the
# right incremental move.
#
# Note what is deliberately NOT checked. A per-submit name read by a live getenv elsewhere is fine:
# live is strictly more eager than per-submit, so the two cannot disagree about an arm that
# precedes a submit, and the per-submit macro itself falls back to a live read outside every scope.
# Only the process-lifetime cache is a genuinely different answer.
def check_per_submit(cached, per_submit) -> int:
    if not per_submit:
        # Not a failure on its own: a tree with no per-submit sites is a legal tree. But say so,
        # because "0 names" and "the regex stopped matching" print identically otherwise, and the
        # self-tests above are what separate them.
        print("  [ok]   no PROSPER_ENV_ON_PER_SUBMIT uses (self-tests above pin the pattern)")
        return 0
    clash = sorted(set(cached) & set(per_submit))
    for name in clash:
        print(f"  [FAIL] {name} is read BOTH process-cached and per-submit:")
        for w in sorted(set(cached[name]))[:2]:
            print(f"           PROSPER_ENV_ON/VALUE at {w}")
        for w in sorted(set(per_submit[name]))[:2]:
            print(f"           PROSPER_ENV_ON_PER_SUBMIT at {w}")
    if clash:
        print("  Pick one contract for the name and use it at every site. A process-cached read")
        print("  never observes a runtime arm; a per-submit read observes it at the next submit.")
        return len(clash)
    total = sum(len(v) for v in per_submit.values())
    print(f"  [ok]   {len(per_submit)} per-submit name(s) over {total} site(s), none also cached")
    return 0


def scan_line(line: str):
    """The per-line half of scan(), exposed so the self-test exercises the real code path."""
    return ([m.group(1) for m in CACHED_RE.finditer(line)],
            [m.group(2) for m in CALL_RE.finditer(line) if armed_callee(m.group(1))],
            [m.group(1) for m in PER_SUBMIT_RE.finditer(line)])


# --- self-test -------------------------------------------------------------------------------
# Without these, a regex that stops matching (a macro rename, a reformat that splits the call over
# two lines) makes this gate report a clean tree forever -- the one way it can fail silently, and
# the same failure the vkval scanner's registration comment warns about. Half the cases assert on
# shapes that must NOT match, because a checker that fires on valid code gets deleted, not heeded.
SELF_TESTS = [
    # (snippet, expected cached names, expected armed names, expected per-submit names)
    ('if (PROSPER_ENV_ON("PROSPER_FOO")) {', ["PROSPER_FOO"], [], []),
    ('const char* v = PROSPER_ENV_VALUE("PROSPER_BAR");', ["PROSPER_BAR"], [], []),
    ('PROSPER_ENV_VALUE( "PROSPER_SPACED" )', ["PROSPER_SPACED"], [], []),
    ('setenv("PROSPER_BAZ", "1", 1);', [], ["PROSPER_BAZ"], []),
    # A reader that names the variable so it can report a refusal is NOT an arming (#3253).
    ('env_u64_or_default("PROSPER_READ_ONLY", text, 8192);', [], [], []),
    ('prosper::diag::env_u64_or_default_capped("PROSPER_READ_ONLY2", v, 8, 99);', [], [], []),
    ('_putenv_s("PROSPER_QUX", "1");', [], ["PROSPER_QUX"], []),
    ('unsetenv("PROSPER_QUUX");', [], ["PROSPER_QUUX"], []),
    # The WRAPPER forms. These are the cases the first version of this checker missed, which let it
    # pass a tree that still had the #2214 defect -- test_gpu_capture_render.cpp arms every one of
    # its diagnostics this way. If these three stop matching, the gate is decorative again.
    ('set_env("PROSPER_WRAPPED", dump_dir.string());', [], ["PROSPER_WRAPPED"], []),
    ('unset_env("PROSPER_UNWRAPPED");', [], ["PROSPER_UNWRAPPED"], []),
    ('set_environment("PROSPER_ENVIRON", "1", false);', [], ["PROSPER_ENVIRON"], []),
    # Two on one line must both be seen -- the real tree has these.
    ('a = PROSPER_ENV_VALUE("PROSPER_A") ? PROSPER_ENV_VALUE("PROSPER_B") : 0;',
     ["PROSPER_A", "PROSPER_B"], [], []),
    # MUST NOT match: a live read is the fix, not the defect.
    ('const char* v = getenv("PROSPER_LIVE");', [], [], []),
    # MUST NOT match: a non-PROSPER variable is not ours to police.
    ('setenv("SDL_AUDIO_DRIVER", "dummy", 1);', [], [], []),
    # MUST NOT match: naming the macro in prose does not read a variable.
    ('// PROSPER_ENV_VALUE caches, so do not use it for a runtime-armed name', [], [], []),
    # PROSPER_ENV_ON_PER_SUBMIT is its OWN contract: neither process-cached nor an arming. Both of
    # those exclusions are currently accidents of pattern boundaries rather than decisions -- the
    # `(` CACHED_RE wants after ON is a `_` here, and armed_callee() bails on the PROSPER_ENV_
    # prefix -- so they are pinned here. Without these three, widening either pattern would start
    # refusing a perfectly legal per-submit name and nothing would say why.
    ('const bool log = PROSPER_ENV_ON_PER_SUBMIT("PROSPER_SUB");', [], [], ["PROSPER_SUB"]),
    ('PROSPER_ENV_ON_PER_SUBMIT( "PROSPER_SUB_SPACED" )', [], [], ["PROSPER_SUB_SPACED"]),
    # A per-submit name MAY be armed at runtime -- that is the whole point of the contract, so the
    # two sets landing on one line is not a defect and must not be reported as one.
    ('setenv("PROSPER_SUB2", "1", 1); if (PROSPER_ENV_ON_PER_SUBMIT("PROSPER_SUB2")) {',
     [], ["PROSPER_SUB2"], ["PROSPER_SUB2"]),
]


# Shapes the hot-site scanner must and must not treat as a LIVE read. Without these, a regex that
# stops matching turns tier 3 into a permanent [ok] -- the same silent-clean failure the tier-1
# self-tests above exist to prevent.
HOT_SELF_TESTS = [
    ('if (getenv("PROSPER_HOT")) {', ["PROSPER_HOT"]),
    ('const char* v = std::getenv("PROSPER_HOT");', ["PROSPER_HOT"]),
    ('x = getenv("PROSPER_HOT") != nullptr;', ["PROSPER_HOT"]),
    # MUST NOT match: already a one-shot read, in the older hand-rolled spelling.
    ('static const bool on = getenv("PROSPER_HOT") != nullptr;', []),
    # MUST NOT match: the cached macro is the fix.
    ('if (PROSPER_ENV_ON("PROSPER_HOT")) {', []),
    # MUST NOT match: prose naming the variable reads nothing.
    ('// PROSPER_HOT is evaluated per draw', []),
]


# The tier-4 predicate, which is a set intersection and therefore looks too simple to test. It is
# not the arithmetic that can break -- it is the DIRECTION. A check written the other way round
# ("per-submit names must also be cached", or an intersection taken against the armed set instead of
# the cached one) passes a clean tree exactly as loudly as the right one, and the clean run below is
# what a reader would take as proof it works. Same shape as the positive-control rule in CLAUDE.md:
# a check that has never been shown to FIRE has not been shown to do anything.
#
# (cached, per_submit, expected number of clashes reported)
PER_SUBMIT_SELF_TESTS = [
    ({"PROSPER_X": ["a.cpp:1"]}, {"PROSPER_X": ["b.cpp:2"]}, 1),      # must FIRE
    ({"PROSPER_X": ["a.cpp:1"]}, {"PROSPER_Y": ["b.cpp:2"]}, 0),      # disjoint: legal
    ({"PROSPER_X": ["a.cpp:1"]}, {}, 0),                              # no per-submit sites at all
    ({}, {"PROSPER_Y": ["b.cpp:2"]}, 0),                              # per-submit only: legal
    # Armed is NOT an input here, deliberately: a per-submit name may be armed at runtime, and an
    # intersection taken against the armed set instead of the cached one would refuse every
    # conversion this mechanism exists to allow. Two clashes, so a check that stops at the first
    # also fails.
    ({"PROSPER_X": ["a.cpp:1"], "PROSPER_Z": ["c.cpp:3"]},
     {"PROSPER_X": ["b.cpp:2"], "PROSPER_Z": ["d.cpp:4"]}, 2),
]


def self_test() -> int:
    bad = 0
    import io, contextlib
    for cached, per_submit, want in PER_SUBMIT_SELF_TESTS:
        with contextlib.redirect_stdout(io.StringIO()):
            got = check_per_submit(cached, per_submit)
        if got != want:
            print(f"  [FAIL] per-submit self-test: cached={sorted(cached)} "
                  f"per_submit={sorted(per_submit)} want={want} got={got}")
            bad += 1
    for snippet, want in HOT_SELF_TESTS:
        got = live_getenv_names(snippet)
        if got != want:
            print(f"  [FAIL] hot-site self-test: {snippet!r} want={want} got={got}")
            bad += 1
    for snippet, want_cached, want_armed, want_submit in SELF_TESTS:
        got_cached, got_armed, got_submit = scan_line(snippet)
        if got_cached != want_cached or got_armed != want_armed or got_submit != want_submit:
            print(f"  [FAIL] self-test: {snippet!r}")
            print(f"         cached     want={want_cached} got={got_cached}")
            print(f"         armed      want={want_armed} got={got_armed}")
            print(f"         per-submit want={want_submit} got={got_submit}")
            bad += 1
    if bad:
        print(f"  the scanner's own patterns are broken -- a tree scan would report a false CLEAN")
    else:
        print(f"  [ok]   scanner self-test: "
              f"{len(SELF_TESTS) + len(HOT_SELF_TESTS) + len(PER_SUBMIT_SELF_TESTS)} cases")
    return bad


def main() -> int:
    here = Path(__file__).resolve()
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here.parents[2]
    print("== check_cached_env ==")
    if self_test():
        return 1

    cached, armed, per_submit = scan(root)
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
        if check_per_submit(cached, per_submit):
            print("== 1 failure(s) ==")
            return 1
        if check_hot_sites(root):
            print("  A hot site regrew a live getenv. #3094 measured these on per-draw and")
            print("  per-resource paths; restore the PROSPER_ENV_ON/VALUE read, or if the site is")
            print("  genuinely no longer hot, remove its entry from HOT_SITES in this file.")
            print("== 1 failure(s) ==")
            return 1
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
