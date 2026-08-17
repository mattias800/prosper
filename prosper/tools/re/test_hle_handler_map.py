#!/usr/bin/env python3
"""test_hle_handler_map — unit tests for the registration-table parser behind hle_handler_map.py.

The tool answers "how many Sony entry points collapse onto ONE prosper handler?", and every way it
can be wrong produces a plausible-looking smaller (or larger) table rather than a visible failure.
Two of the cases below are regressions for defects that had already reached a published measurement
(#2070):

  * **Both arms of one `#if` counted as two Sony names.** `hle_kernel_mem.cpp` defines
    `register_kernel_mem_hle()` twice, ~3,200 lines apart, in the two arms of one
    `#if defined(__linux__) || defined(__APPLE__)`. A line-based extraction sees every kernel-memory
    handler registered "twice" and promotes it to a shared handler — five handlers that answer
    exactly one Sony function each were counted as collapses this way.
  * **Registration counted per SITE instead of per distinct Sony name.** `scePthreadAttrSetaffinity`
    is registered twice to `k_attr_noop` (hle_kernel.cpp:4539 and :4588). It is one entry point, and
    counting the lines rather than the names inflates every handler that has a duplicate.

The rest pin the properties that make a zero from this tool readable at all: that the shape list is
DISCOVERED rather than hardcoded, that a shape it cannot parse is reported instead of dropped, and
that a registration BLOCK is never mistaken for a forwarding wrapper (which would empty the census).

Run directly, or via ctest as `re_hle_handler_map`.
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hle_handler_map as H                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "hle_handler_map.py")
SRC_HLE = os.path.normpath(os.path.join(HERE, "..", "..", "src", "hle"))

fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    print("  [%s] %-58s got=%r" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        fails += 1
        print("         wanted %r" % (want,))


# A minimal stand-in for the real dispatch.hpp: the tool reads the registration API list out of the
# `class Hle` declaration rather than hardcoding it, and these fixtures exercise that.
DISPATCH_HPP = """
namespace prosper {
using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
class Hle {
public:
    static void  register_fn(const std::string& nid, HleFn fn, const char* name,
                             HleReturnHook return_hook = nullptr);
    static void  register_placeholder(const std::string& nid, HleFn fn, const char* name);
    static HleFn lookup(const std::string& nid);
};
}
"""


def scan_fixture(sources, platform="linux", dispatch=DISPATCH_HPP):
    """Parse a synthetic `src/hle` made of {filename: text}. Returns the Scan."""
    d = tempfile.mkdtemp(prefix="hle_handler_map_t_")
    try:
        with open(os.path.join(d, "dispatch.hpp"), "w") as f:
            f.write(dispatch)
        for name, text in sources.items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        return H.scan_tree(d, platform)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def serves(sc):
    """handler -> number of DISTINCT Sony NIDs it answers."""
    return {h: len(d) for h, d in H.handler_index(sc.regs).items()}


# ---------------------------------------------------------------- 1. nid_hash

def test_nid_hash():
    """`nid_hash` must agree with Sony's own hash, checked against an INDEPENDENT oracle.

    These four pairs were read out of the PS5 3.20 firmware library dump
    (`sprx_dlsym(__handle, "<NID>", &__ptr_<name>)`), which was produced by neither this file nor
    `src/hle/nid.cpp`. Pinning them here keeps the test self-contained — a port that agreed with
    nid.cpp but with nothing else would satisfy a round-trip and still be wrong.
    """
    print("nid_hash against the PS5 3.20 firmware dump:")
    for name, nid in [("sceKernelMapFlexibleMemory", "IWIBBdTHit4"),
                      ("sceUserServiceInitialize", "j3YMu1MVNNo"),
                      ("sceCommonDialogIsUsed", "BQ3tey0JmQM"),
                      ("scePthreadJoin", "onNY9Byn-W8")]:
        check("nid_hash(%s)" % name, H.nid_hash(name), nid)


# ---------------------------------------------------------------- 2. every shape

SHAPES = r"""
namespace prosper {
uint64_t h_a(uint64_t) { return 0; }
uint64_t h_b(uint64_t) { return 0; }

// A registration BLOCK: zero parameters, so it must NOT be mistaken for a forwarding wrapper.
// If it were, every call below would be dismissed as "the wrapper's template" and the census
// would come back empty -- the single most destructive way this parser can be wrong.
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)
    #define RN_NAMED(nid, fn, name) Hle::register_fn(nid, (HleFn)(fn), name, &hook)
    R("sceDemoAlpha", h_a);
    RN("AAAAAAAAAAA", h_b);
    RN_NAMED("BBBBBBBBBBB", h_b, "sceDemoBeta");
    Hle::register_fn("CCCCCCCCCCC", (HleFn)h_a, "sceDemoGamma");
    Hle::register_placeholder("DDDDDDDDDDD", (HleFn)h_b, "sceDemoDelta");
    auto reg = [](const char* nid, HleFn fn, const char* name) { Hle::register_fn(nid, fn, name); };
    reg("EEEEEEEEEEE", (HleFn)h_a, "sceDemoEpsilon");
    #undef R
    #undef RN
    #undef RN_NAMED
}
}
"""


def test_every_shape():
    print("every registration shape is parsed:")
    sc = scan_fixture({"demo.cpp": SHAPES})
    check("no unclaimed sites", sc.unclaimed, [])
    check("registrations found", len(sc.regs), 6)
    check("h_a serves 3", serves(sc).get("h_a"), 3)
    check("h_b serves 3", serves(sc).get("h_b"), 3)
    check("nid_hash form resolved", H.nid_hash("sceDemoAlpha") in {r.nid for r in sc.regs}, True)
    check("RN_NAMED display name kept",
          next(r.name for r in sc.regs if r.nid == "BBBBBBBBBBB"), "sceDemoBeta")
    check("register_placeholder counted",
          next(r.api for r in sc.regs if r.nid == "DDDDDDDDDDD"), "register_placeholder")
    kinds = {w.kind for w in sc.wrappers}
    check("macro + lambda wrappers discovered", kinds, {"macro", "lambda"})


# ---------------------------------------------------------------- 3. the cross-shape collapse

CROSS_SHAPE = r"""
namespace prosper {
uint64_t k_map_flex(uint64_t) { return 0; }
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelMapFlexibleMemory", k_map_flex);
    Hle::register_fn("4h6F1LLbTiw", (HleFn)k_map_flex, "sceKernelMapFlexibleMemoryInternal");
    #undef R
}
}
"""


def test_cross_shape_collapse():
    """A collapse spanning TWO shapes — the positive control constructed by hand.

    Modelled on a real pair read out of `src/hle/hle_kernel_mem.cpp` (Linux arm, lines 2615-2617):
    `k_map_flexible_noname` answers `sceKernelMapFlexibleMemory` through the `R` macro AND
    `4h6F1LLbTiw` (`sceKernelMapFlexibleMemoryInternal`) through a direct raw-NID call. It can only
    be found if BOTH shapes are parsed, so a parser that handles either one alone reports "no
    collapse" — which is exactly the false clean answer this tool exists to prevent.
    """
    print("a collapse spanning two different registration shapes:")
    sc = scan_fixture({"demo.cpp": CROSS_SHAPE})
    check("no unclaimed sites", sc.unclaimed, [])
    check("k_map_flex serves 2 distinct NIDs", serves(sc).get("k_map_flex"), 2)
    nids = {r.nid for r in sc.regs}
    check("macro side resolved", H.nid_hash("sceKernelMapFlexibleMemory") in nids, True)
    check("direct side resolved", "4h6F1LLbTiw" in nids, True)


# ---------------------------------------------------------------- 4. per-NAME, not per-SITE

DUPLICATE_SITE = r"""
namespace prosper {
uint64_t k_attr_noop(uint64_t) { return 0; }
uint64_t k_other(uint64_t) { return 0; }
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("scePthreadAttrSetaffinity", k_attr_noop);
    R("scePthreadAttrSetaffinity", k_attr_noop);   // the real tree does this at :4539 and :4588
    R("scePthreadSetaffinity", k_other);
    #undef R
}
}
"""


def test_distinct_names_not_sites():
    """One Sony name registered twice to one handler is ONE entry point, not two.

    Counting sites instead of names is half of what turned the tool's 36 shared gated rows into a
    published 41 (#2070); the other half is the `#if` case below.
    """
    print("a name registered twice counts once:")
    sc = scan_fixture({"demo.cpp": DUPLICATE_SITE})
    check("two registration SITES parsed",
          sum(1 for r in sc.regs if r.handler == "k_attr_noop"), 2)
    check("but k_attr_noop serves ONE distinct name", serves(sc).get("k_attr_noop"), 1)
    check("k_other serves one", serves(sc).get("k_other"), 1)


# ---------------------------------------------------------------- 5. the #if arms

PLATFORM_ARMS = r"""
namespace prosper {
uint64_t k_dmem_size(uint64_t) { return 0; }
uint64_t s_ok(uint64_t) { return 0; }
uint64_t s_real(uint64_t) { return 0; }
#if defined(__linux__) || defined(__APPLE__)
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelGetDirectMemorySize", k_dmem_size);
    #undef R
}
#else
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceKernelGetDirectMemorySize", k_dmem_size);
    #undef R
}
#endif
void register_more_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceCommonDialogIsUsed", s_ok);
#ifndef _WIN32
    Hle::register_fn("3Zl8BePTh9Y", (HleFn)s_real, "sceNpCheckCallback");
#else
    R("sceNpCheckCallback", s_ok);
#endif
    #undef R
}
}
"""


def test_platform_arms():
    """Both arms of one `#if` are ONE registration, not two — and the arms really do differ.

    This is the regression for the defect behind the published 41: on the real tree, five
    `hle_kernel_mem.cpp` handlers that answer exactly one Sony function each were promoted to
    "shared" purely because the same `R(...)` line appears once per arm.
    """
    print("mutually exclusive #if arms are one registration:")
    for platform in ("linux", "windows", "macos"):
        sc = scan_fixture({"demo.cpp": PLATFORM_ARMS}, platform=platform)
        check("[%s] no unclaimed" % platform, sc.unclaimed, [])
        check("[%s] k_dmem_size serves ONE name" % platform, serves(sc).get("k_dmem_size"), 1)
    lin = scan_fixture({"demo.cpp": PLATFORM_ARMS}, platform="linux")
    win = scan_fixture({"demo.cpp": PLATFORM_ARMS}, platform="windows")
    # The raw NID written in the fixture and the hash of the name are the SAME symbol — that is why
    # the two arms are interchangeable in the real source, and asserting it here also pins the two
    # literal forms against each other on a name taken from prosper's own tree.
    check("the two literal forms name one symbol",
          H.nid_hash("sceNpCheckCallback"), "3Zl8BePTh9Y")
    check("linux: sceNpCheckCallback answered by the REAL handler",
          {r.handler for r in lin.regs if r.nid == "3Zl8BePTh9Y"}, {"s_real"})
    check("linux: s_ok serves only sceCommonDialogIsUsed", serves(lin).get("s_ok"), 1)
    check("windows: sceNpCheckCallback collapses onto s_ok",
          {r.handler for r in win.regs if r.nid == "3Zl8BePTh9Y"}, {"s_ok"})
    check("windows: s_ok therefore serves 2", serves(win).get("s_ok"), 2)
    check("windows: the real handler is registered for nothing",
          "s_real" in serves(win), False)


# ---------------------------------------------------------------- 6. macro scope

UNDEF_SCOPE = r"""
namespace prosper {
uint64_t h_a(uint64_t) { return 0; }
int R(int x) { return x; }              // an unrelated R AFTER the macro is undefined
void register_demo_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("sceDemoAlpha", h_a);
    #undef R
}
int later() { return R(3) + R(4); }     // must NOT be read as two registrations
}
"""


def test_undef_scope():
    """`R(` after `#undef R` is not a registration.

    Without scoping, the two `R(3)` / `R(4)` calls below become arity-matched "registrations" with
    unresolvable NIDs, and — worse — a second `#define R` elsewhere in the same file would claim
    this one's call sites too, doubling every count it touches.
    """
    print("a macro's call sites end at its #undef:")
    sc = scan_fixture({"demo.cpp": UNDEF_SCOPE})
    check("exactly one registration", len(sc.regs), 1)
    check("no unclaimed", sc.unclaimed, [])
    check("out-of-scope R( calls counted and reported", sc.out_of_scope, 2)


# ---------------------------------------------------------------- 7. discovered, not hardcoded

NEW_API_HPP = DISPATCH_HPP.replace(
    "    static HleFn lookup",
    "    static void  register_alias(const std::string& nid, HleFn fn, const char* name);\n"
    "    static HleFn lookup")

NEW_SHAPES = r"""
namespace prosper {
uint64_t h_a(uint64_t) { return 0; }
// A macro that wraps ANOTHER macro: a one-level scan misses this silently, because nothing at the
// R2(...) call sites textually mentions a registration API.
// And a free-function forwarder, which is neither a macro nor a lambda.
static void reg3(const char* nid, HleFn fn, const char* name) { Hle::register_fn(nid, fn, name); }
void register_demo_hle() {
    // R2 is defined BEFORE the macro it expands to, which is legal (a macro body is expanded at the
    // use site, not at definition) and is what makes the fixpoint load-bearing: a single discovery
    // pass sees R2 forwarding to a name it does not yet know is a wrapper, and drops it silently.
    #define R2(str, fn) R(str, fn)
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R2("sceDemoAlpha", h_a);
    reg3("AAAAAAAAAAA", (HleFn)h_a, "sceDemoBeta");
    Hle::register_alias("BBBBBBBBBBB", (HleFn)h_a, "sceDemoGamma");
    #undef R2
    #undef R
}
}
"""


EXTRA_IN_WRAPPER = r"""
namespace prosper {
uint64_t h_a(uint64_t) { return 0; }
uint64_t h_b(uint64_t) { return 0; }
// A forwarder that ALSO makes one fixed registration of its own. Only the forwarded call is the
// wrapper's template; the fixed one beside it is an ordinary registration site.
static void reg_and_seed(const char* nid, HleFn fn, const char* name) {
    Hle::register_fn(nid, fn, name);
    Hle::register_fn("ZZZZZZZZZZZ", (HleFn)h_b, "sceDemoSeed");
}
void register_demo_hle() {
    reg_and_seed("AAAAAAAAAAA", (HleFn)h_a, "sceDemoAlpha");
    reg_and_seed("BBBBBBBBBBB", (HleFn)h_a, "sceDemoBeta");
}
}
"""


def test_registration_beside_a_forward_is_not_swallowed():
    """A second registration inside a wrapper body must still be counted.

    Skipping a wrapper's WHOLE body as "the template" also discards any fixed registration written
    beside the forwarded one — and it vanishes with nothing in the residual, because a mention
    inside a wrapper body is exactly what the residual is told to ignore. Only the forwarded call's
    own span is the template.
    """
    print("a fixed registration beside a forward is still counted:")
    sc = scan_fixture({"demo.cpp": EXTRA_IN_WRAPPER})
    check("no unclaimed", sc.unclaimed, [])
    check("the forwarder is discovered",
          {w.name for w in sc.wrappers}, {"reg_and_seed"})
    check("h_a serves the two forwarded NIDs", serves(sc).get("h_a"), 2)
    check("the fixed registration beside it survives", serves(sc).get("h_b"), 1)
    check("and it is the right NID",
          {r.nid for r in sc.regs if r.handler == "h_b"}, {"ZZZZZZZZZZZ"})


def test_shapes_are_discovered():
    """A registration API, a chained macro and a free-function forwarder that nobody hardcoded.

    None of these three exist in `src/hle` today. They are here because "grep found no other shapes"
    is not evidence that no other shape can appear — it only says the shapes already thought of were
    searched for. What makes the claim checkable is that the parser derives its API list from
    `dispatch.hpp` and its wrapper list from any body that forwards a parameter into a registration
    call, so a shape written tomorrow is found without touching this tool.
    """
    print("APIs and wrappers are discovered, not hardcoded:")
    sc = scan_fixture({"demo.cpp": NEW_SHAPES}, dispatch=NEW_API_HPP)
    check("register_alias discovered from dispatch.hpp",
          "register_alias" in sc.apis, True)
    check("no unclaimed", sc.unclaimed, [])
    check("all three registrations found", len(sc.regs), 3)
    check("h_a serves 3 distinct NIDs", serves(sc).get("h_a"), 3)
    check("chained macro resolved via R2->R",
          H.nid_hash("sceDemoAlpha") in {r.nid for r in sc.regs}, True)
    check("free-function forwarder discovered",
          {w.name for w in sc.wrappers if w.kind == "function"}, {"reg3"})


# ---------------------------------------------------------------- 8. a missed shape is LOUD

BROKEN_SHAPE = r"""
namespace prosper {
uint64_t h_a(uint64_t) { return 0; }
void register_demo_hle() {
    Hle::register_fn("AAAAAAAAAAA", (HleFn)h_a, "sceDemoAlpha");
    Hle::register_fn("BBBBBBBBBBB");                       // a shape this parser cannot read
}
}
"""


def test_unparsable_shape_is_reported():
    """A registration the parser cannot read must be REPORTED, never dropped.

    This is the whole coverage contract (#2149): "no collapses" and "I parsed nothing" must not be
    the same output. An unparsable site keeps the run at a non-zero exit so a saved transcript
    carries its own validity.
    """
    print("an unparsable registration is reported and changes the exit code:")
    sc = scan_fixture({"demo.cpp": BROKEN_SHAPE})
    check("the readable one still parsed", len(sc.regs), 1)
    check("the unreadable one is unclaimed", len(sc.unclaimed), 1)
    d = tempfile.mkdtemp(prefix="hle_handler_map_t_")
    try:
        open(os.path.join(d, "dispatch.hpp"), "w").write(DISPATCH_HPP)
        open(os.path.join(d, "demo.cpp"), "w").write(BROKEN_SHAPE)
        r = subprocess.run([sys.executable, TOOL, "--src", d], capture_output=True, text=True)
        check("exit 3 (incomplete), not 0", r.returncode, 3)
        check("says so on stdout", "UNCLAIMED" in r.stdout, True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_empty_tree_refuses():
    """Zero registrations is a REFUSAL, not an answer of zero."""
    print("an empty tree refuses instead of reporting a clean zero:")
    d = tempfile.mkdtemp(prefix="hle_handler_map_t_")
    try:
        open(os.path.join(d, "dispatch.hpp"), "w").write(DISPATCH_HPP)
        open(os.path.join(d, "empty.cpp"), "w").write("namespace prosper { int x = 1; }\n")
        r = subprocess.run([sys.executable, TOOL, "--src", d], capture_output=True, text=True)
        check("exit 2 (refused)", r.returncode, 2)
        check("stderr explains", "refused" in r.stderr, True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 9. the real tree

def test_real_tree():
    """Structural invariants on prosper's ACTUAL `src/hle`, on every platform.

    Deliberately NOT an exact census pin: several lanes add registrations concurrently, and a test
    that fails whenever someone registers a new NID would be noise rather than a guard. What is
    pinned is what cannot drift without the measurement becoming unreadable — that nothing is
    unclaimed, that the well-known shared handlers are still found, and that the specific
    double-count behind the published 41 does not come back.
    """
    if not os.path.isdir(SRC_HLE):
        print("real tree: SKIP (%s not found)" % SRC_HLE)
        return
    print("prosper's real src/hle:")
    for platform in ("linux", "windows", "macos"):
        sc = H.scan_tree(SRC_HLE, platform)
        s = serves(sc)
        check("[%s] nothing unclaimed" % platform, sc.unclaimed, [])
        check("[%s] parsed a real number of sites" % platform, len(sc.regs) > 900, True)
        check("[%s] k_attr_noop serves 20" % platform, s.get("k_attr_noop"), 20)
        check("[%s] s_ok serves 14" % platform, s.get("s_ok"), 14)
        # The five handlers the platform-blind extraction promoted to "shared" (#2070). Each is
        # registered once per `#if` arm of hle_kernel_mem.cpp and answers exactly ONE Sony function.
        for h in ("k_dmem_size", "k_virtual_query", "k_alloc_dmem", "k_mtypeprotect", "k_mprotect"):
            check("[%s] %s answers exactly one Sony name" % (platform, h), s.get(h), 1)


def main():
    test_nid_hash()
    test_every_shape()
    test_cross_shape_collapse()
    test_distinct_names_not_sites()
    test_platform_arms()
    test_undef_scope()
    test_registration_beside_a_forward_is_not_swallowed()
    test_shapes_are_discovered()
    test_unparsable_shape_is_reported()
    test_empty_tree_refuses()
    test_real_tree()
    print("\n%s" % ("all checks passed" if not fails else "%d CHECK(S) FAILED" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
