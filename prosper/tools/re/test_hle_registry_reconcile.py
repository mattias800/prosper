#!/usr/bin/env python3
"""test_hle_registry_reconcile — is `hle_handler_map.py`'s static parse the table prosper BUILDS?

`hle_handler_map.py` reads `src/hle/**` and reports which Sony NIDs collapse onto one prosper
handler. Its two structural weaknesses cannot be tested from inside itself:

  * A registration SHAPE it does not recognise is invisible to it. It does not error — the census
    simply comes back smaller, and a smaller collapse census reads as good news.
  * It reimplements the C preprocessor to pick a `#if` arm, and `hle_kernel_mem.cpp` alone has two
    ~3,200-line arms of `register_kernel_mem_hle()`. Only the compiler knows which one won.

So this reconciles the parse against `hle_registry_dump`, which runs the real
`register_builtin_hle()` and prints the resulting registry. That is ground truth by construction —
it is the same map the guest's imports resolve through — and it makes the shape list's completeness
a CHECKED property rather than an assertion that grep found nothing else.

Reading the two directions of disagreement:

    registered at runtime, absent from the parse   -> a missed shape. Always a failure.
    parsed, absent at runtime                      -> a registrar this build does not call, or an
                                                      inactive arm. Reported, and bounded.

The handler-ADDRESS column is used only as an upper bound on collapse: an identical-code-folding
linker can give two distinct `{ return 0; }` handlers one address, so equal addresses do not prove
a shared handler — but unequal addresses do prove distinct ones. The test therefore checks the one
sound direction (everything the parser calls shared must share an address) and reports folding.

Usage:  test_hle_registry_reconcile.py <path to hle_registry_dump>
Run via ctest as `re_hle_handler_map_reconcile`.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hle_handler_map as H                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_HLE = os.path.normpath(os.path.join(HERE, "..", "..", "src", "hle"))

fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    print("  [%s] %-58s got=%r" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        fails += 1
        print("         wanted %r" % (want,))


def main():
    if len(sys.argv) < 2:
        print("refused: needs the path to hle_registry_dump", file=sys.stderr)
        return 2
    dump = sys.argv[1]
    if not os.path.isfile(dump):
        print("refused: %s does not exist (build target `hle_registry_dump`)" % dump,
              file=sys.stderr)
        return 2

    print("== test_hle_registry_reconcile ==")
    tmp = tempfile.mkdtemp(prefix="hle_registry_")
    path = os.path.join(tmp, "registry.tsv")
    r = subprocess.run([dump, path], capture_output=True, text=True)
    check("hle_registry_dump exits 0", r.returncode, 0)
    if r.returncode != 0:
        print(r.stderr[:2000])
        return 1

    registry = H.read_registry(path)
    # The host this runs on IS the platform whose arms the binary compiled, so that is the arm the
    # parser must be asked for. Reconciling a Linux binary against the Windows parse would report a
    # pile of differences that mean nothing.
    platform = ("windows" if sys.platform.startswith("win")
                else "macos" if sys.platform == "darwin" else "linux")
    sc = H.scan_tree(SRC_HLE, platform)
    hard_missing, declared_missing, extra, groups = H.reconcile(sc, registry, SRC_HLE)
    missing = hard_missing + declared_missing

    print("  platform=%s  runtime=%d NIDs  parsed=%d NIDs  (parser also saw %d sites whose NID is "
          "not a literal)" % (platform, len(registry), len({x.nid for x in sc.regs}),
                              len(sc.unresolved)))

    check("the dump is not empty", len(registry) > 900, True)
    check("the parse is not empty", len(sc.regs) > 900, True)
    check("nothing unclaimed in the parse", sc.unclaimed, [])

    # THE assertion. A NID the binary registers and the parser never produced is one of two things,
    # and they are told apart by where the NID's TEXT lives — not by a count, which would be a
    # budget the next table growth silently exhausts:
    #
    #   * it is a literal inside one of the arrays the parser NAMED as unresolved (`kUlt[...].nid`,
    #     `kAgcNids[Offset + Is]`) -> a declared limit, reported in the coverage block on every run.
    #   * it is anywhere else -> the parser cannot see how it got registered at all. That is a
    #     missed shape, and it is the failure this test exists for.
    check("no NID is registered by a shape the parser cannot read", hard_missing, [])
    print("  (%d of the %d runtime-only NIDs come from the %d array-driven sites the parser names "
          "as unresolved)" % (len(declared_missing), len(missing), len(sc.unresolved)))
    check("every runtime-only NID is accounted for", len(declared_missing), len(missing))

    # The reverse direction is bounded, not zero: a registrar the dump's build does not call would
    # show up here, and so would a genuine parser over-report. Either is worth seeing.
    check("the parser invents no registrations beyond a small, explainable set", len(extra) <= 8,
          True)
    if extra:
        print("  parsed but not registered at runtime: %s" % " ".join(extra[:20]))

    # Sound direction only: shared-by-the-parser must be shared-by-address at runtime.
    check("every shared handler the parser found shares one runtime address", groups, [])

    # And report folding, so a reader knows why an address-keyed grouping would over-count.
    by_addr = {}
    for nid, (_kind, addr, _name) in registry.items():
        by_addr.setdefault(addr, []).append(nid)
    idx = H.handler_index(sc.regs)
    static_groups = sum(1 for d in idx.values() if len(d) >= 2)
    addr_groups = sum(1 for v in by_addr.values() if len(v) >= 2)
    print("  handlers serving >1 NID: %d by source, %d by runtime address "
          "(address >= source; the excess is identical-code folding)" % (static_groups, addr_groups))
    check("runtime address grouping is an upper bound on the source grouping",
          addr_groups >= static_groups, True)

    # Reported, not asserted. When the linker folds nothing, the two partitions come out IDENTICAL,
    # and that is the strongest statement available here: the compiler and the parser independently
    # agree on exactly which Sony entry points collapse onto one answer. It is not a gate because
    # enabling identical-code folding in a future build would merge distinct handlers and break it
    # legitimately — the sound direction is the check above.
    src_parts = {frozenset(d) for d in idx.values() if len(d) >= 2}
    addr_parts = {frozenset(v) for v in by_addr.values() if len(v) >= 2}
    print("  source-derived and address-derived shared-handler partitions identical: %s%s"
          % (src_parts == addr_parts,
             "" if src_parts == addr_parts
             else "  (%d groups only the binary sees — identical-code folding)"
                  % len(addr_parts - src_parts)))

    print("\n%s" % ("all checks passed" if not fails else "%d CHECK(S) FAILED" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
