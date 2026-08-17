#!/usr/bin/env python3
"""test_stub_nid_map — unit tests for the stub -> NID inversion behind stub_nid_map.py.

The tool answers "which import does the stub at address X resolve to?", and both ways it can be
wrong are silent. A dropped entry point makes a real import report `NOT A STUB`, which reads as
"this call does not go through an import" — the reader then concludes the call chain does something
it does not. A dropped *fail-visible* line is worse: the address simply vanishes from the output,
and "not an import" and "you named the wrong address" become the same empty result.

Each arm below names the mutation it kills, because an assertion over a mapping is easy to write so
that it passes on a mapping built the wrong way round.

Run directly, or via ctest as `re_stub_nid_map`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nid_gate_scan as G                                            # noqa: E402
import stub_nid_map as S                                             # noqa: E402

fails = 0


def check(name, got, want):
    global fails
    if got != want:
        fails += 1
        print("[FAIL] %s\n         got  %r\n         want %r" % (name, got, want))
    else:
        print("[ok]   %s" % name)


class FakeImg(G.Image):
    """Enough Image for `invert`: a byte window (for the CET prologue peek), a dynsym and JMPREL.

    `imports` is [(nid, sym_index)]; `slots` is {sym_index: [slot VA]}.
    """

    def __init__(self, base, blob, imports, slots):
        self.base, self.raw = base, blob
        self._imports, self._slots = imports, slots

    def foff(self, va):
        off = va - self.base
        return off if 0 <= off < len(self.raw) else None

    def imported_nids(self):
        return self._imports

    def jump_slots(self, sym_index):
        return self._slots.get(sym_index, [])


def main():
    # A stub with a CET prologue at 0x1000 whose `jmp *[rip+d]` is at 0x1004, targeting slot 0x9000.
    cet_blob = b"\xf3\x0f\x1e\xfa" + b"\xff\x25\x00\x00\x00\x00"
    img = FakeImg(0x1000, cet_blob, [("fMP5NHUOaMk", 3)], {3: [0x9000]})
    xref = {0x9000: [(0x1004, "jmp*")]}
    smap = S.invert(img, xref)

    # BOTH entry points must resolve. Mutation killed: keying the map by NID (or keeping only the
    # `jmp` address) — a caller targeting the CET prologue at 0x1000 would then get NOT A STUB for
    # an address that is unambiguously an import stub.
    check("cet-stub-both-entry-points-resolve",
          sorted(smap), [0x1000, 0x1004])
    # `.get` rather than `[]` so a mutation that drops an entry point fails this arm too, with a
    # readable diff, instead of raising KeyError and aborting the remaining arms.
    check("cet-stub-entry-points-name-the-same-nid",
          [smap.get(0x1000, (None,))[0], smap.get(0x1004, (None,))[0]],
          ["fMP5NHUOaMk", "fMP5NHUOaMk"])

    # A plain (non-CET) stub contributes exactly one entry point.
    plain = FakeImg(0x2000, b"\x90\x90\x90\x90" + b"\xff\x25\x00\x00\x00\x00",
                    [("Oad3rvY-NJQ", 1)], {1: [0xA000]})
    check("plain-stub-is-jmp-address-only",
          sorted(S.invert(plain, {0xA000: [(0x2004, "jmp*")]})), [0x2004])

    # An import whose slot is reached only by a direct `call *[rip+d]` has NO stub, so it has no
    # address to name and must be absent. Mutation killed: treating every xref kind as a stub,
    # which would invent a stub address at the call site itself.
    check("call-star-site-is-not-a-stub",
          S.invert(plain, {0xA000: [(0x2004, "call*")]}), {})

    # An import with a jump slot that nothing references at all is likewise absent.
    check("unreferenced-slot-yields-no-stub", S.invert(plain, {}), {})

    # --- format_rows(): the fail-visible branch --------------------------------------------------
    names = {"fMP5NHUOaMk": ("sceSysmoduleIsLoaded", "libSceSysmodule")}
    rows = S.invert(img, xref)

    # A requested address that is not a stub must produce a LINE, not an omission. Mutation killed:
    # `continue`-ing silently, after which a reader cannot distinguish "not an import call" from
    # "wrong address" — both print nothing.
    check("requested-non-stub-is-reported",
          S.format_rows(rows, [0x4444], "eboot.bin", names),
          ["0x4444\t-\tNOT A STUB in eboot.bin\t-"])

    # ...and the same address must NOT be invented when the user asked for the full listing.
    check("full-listing-invents-no-rows",
          S.format_rows(rows, [], "eboot.bin", names),
          ["0x1000\tfMP5NHUOaMk\tsceSysmoduleIsLoaded\tlibSceSysmodule",
           "0x1004\tfMP5NHUOaMk\tsceSysmoduleIsLoaded\tlibSceSysmodule"])

    # An unsymbolicated NID still gets a row — --names is labels, never coverage.
    check("unknown-nid-still-listed",
          S.format_rows(rows, [0x1004], "eboot.bin", {}),
          ["0x1004\tfMP5NHUOaMk\t?\t?"])

    print("\n%s (%d failure(s))" % ("FAILED" if fails else "all passed", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
