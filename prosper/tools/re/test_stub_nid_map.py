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

    # --- patched_import_stubs(): imports overwritten in place -------------------------------------
    # The detector tells a reader their dump has been modified, so a false positive is expensive and
    # a false negative wastes the hour it exists to save. Both directions get arms.
    import struct as _st

    BASE = 0x10000
    RESOLVER = BASE + 0x400

    ELSEWHERE = BASE + 0x600

    def entry(k, reloc, head=b"\xff\x25\x00\x00\x00\x00", target=None):
        """One 16-byte PLT entry at blob index k: head, `push reloc`, `jmp target` (default PLT0)."""
        rel = (RESOLVER if target is None else target) - (BASE + k * 16 + 16)
        return head[:6].ljust(6, b"\x90") + b"\x68" + _st.pack("<I", reloc) + \
            b"\xe9" + _st.pack("<i", rel)

    JUNK6 = b"\x83\xfe\x45\x77\x06\x90"          # a hand-written stub's first 6 bytes
    WHOLE = b"\x31\xc0\xc3" + b"\x90" * 13          # xor eax,eax; ret; padding -- no tail at all

    class FakePlt(G.Image):
        def __init__(self, blob):
            self.base, self.raw = BASE, blob
            self.segs = [(BASE, 0, len(blob), 1)]      # (va, foff, filesz, PF_X)
            self.tags = {}                             # no JMPREL: format_patched must still work

        def foff(self, va):
            off = va - self.base
            return off if 0 <= off < len(self.raw) else None

        def imported_nids(self):
            return [("AAAAAAAAAAA", 10), ("BBBBBBBBBBB", 11), ("CCCCCCCCCCC", 12),
                    ("DDDDDDDDDDD", 13)]

    syms = {0: 10, 1: 11, 2: 12, 3: 13}

    # CONTROL, and it comes first: a module whose entries are all intact must report NOTHING. An
    # arm that only ever checks for detections passes just as well on a detector that fires always.
    clean = FakePlt(entry(0, 0) + entry(1, 1) + entry(2, 2) + entry(3, 3))
    check("clean-plt-reports-no-patching", S.patched_import_stubs(clean, syms), [])

    # A patched head with the lazy-binding tail left behind: the `tail` signal.
    tailed = FakePlt(entry(0, 0) + entry(1, 1, JUNK6) + entry(2, 2) + entry(3, 3))
    check("tail-left-behind-is-detected",
          S.patched_import_stubs(tailed, syms), [(BASE + 16, 1, 11, "tail")])

    # A patched entry with the tail ALSO gone leaves no `push` to find, so only the geometry can
    # reach it -- and on the dump this was written for, that was the entry that broke the title
    # while five noisier ones were caught. Mutation killed: dropping the geometry pass.
    gone = FakePlt(entry(0, 0) + WHOLE + entry(2, 2) + entry(3, 3))
    check("whole-entry-overwritten-is-detected",
          S.patched_import_stubs(gone, syms), [(BASE + 16, 1, 11, "whole-entry")])

    # ...and the geometry must not INVENT slots outside the span it actually observed, which is the
    # over-report the docstring promises not to make: an import may legitimately have no PLT entry.
    # Relocs 0 and 1 are intact and nothing else is, so nothing may be reported.
    #
    # The padding is load-bearing and was added after a mutation test: with the blob ending right
    # after the two entries, widening the loop bound reddened NOTHING, because every extrapolated
    # address fell outside the segment and was dropped by the `foff is None` guard instead. The arm
    # was passing for a reason that had nothing to do with the span bound it names. The padding puts
    # the invented slots inside the segment, so only the bound can refuse them.
    short = FakePlt(entry(0, 0) + entry(1, 1) + b"\x90" * 64)
    check("geometry-does-not-extrapolate-past-observed-entries",
          S.patched_import_stubs(short, syms), [])

    # A `push imm32; jmp rel32` that jumps somewhere OTHER than the module's resolver is not a PLT
    # entry. Mutation killed: accepting every tail instead of taking the majority vote.
    #
    # The stray sits ON the grid, at the slot where entry 4 belongs and with reloc 4, so the grid
    # guard cannot refuse it and only the resolver can. An earlier version put it off-grid, and the
    # grid guard therefore shadowed this arm completely -- the vote could be deleted with all 21 arms
    # still green. That is the third arm in this file to have passed for a reason other than the one
    # it names; the shape to watch for is a fixture that trips an EARLIER guard than the one under
    # test.
    # syms5 (not syms) is load-bearing: with reloc 4 absent from the symbol table the TAIL SYMBOL
    # GUARD drops the stray first and the vote is never consulted -- which is how this arm failed to
    # discriminate on its first rewrite too.
    syms5 = {**syms, 4: 14}
    on_grid_stray = FakePlt(entry(0, 0) + entry(1, 1) + entry(2, 2) + entry(3, 3) +
                            entry(4, 4, JUNK6, ELSEWHERE))
    check("on-grid-stray-jumping-elsewhere-is-not-a-plt-entry",
          S.patched_import_stubs(on_grid_stray, syms5), [])

    # ...and the twin that proves the arm above discriminates on the RESOLVER and nothing else: the
    # same slot, same reloc, same patched head, tail jumping to the real PLT0 -- which IS reportable.
    # Without this an arm expecting [] could pass because the slot is unreportable for some unrelated
    # reason.
    on_grid_real = FakePlt(entry(0, 0) + entry(1, 1) + entry(2, 2) + entry(3, 3) +
                           entry(4, 4, JUNK6))
    check("...and the same slot WITH the module's own resolver is reported",
          S.patched_import_stubs(on_grid_real, syms5),
          [(BASE + 64, 4, 14, "tail")])

    # A reloc index the JMPREL table does not know is not reported by the geometry pass -- it would
    # be a row naming no import, which is worse than silence.
    check("geometry-skips-relocs-with-no-symbol",
          S.patched_import_stubs(gone, {0: 10, 2: 12, 3: 13}), [])

    # A patched entry whose reloc the JMPREL table does not know must not be emitted as a row naming
    # no import -- it renders as `reloc=4  -  ?  ?`, an accusation about an import that does not
    # exist. The entry has to sit ON the grid, or the grid guard suppresses it first and this arm
    # tests that guard instead (it did, until a mutation sweep showed the symbol guard could be
    # deleted with nothing going red).
    unknown_reloc = FakePlt(entry(0, 0) + entry(1, 1) + entry(2, 2) + entry(3, 3) +
                            entry(4, 4, JUNK6))
    check("tail-row-naming-no-import-is-not-emitted",
          S.patched_import_stubs(unknown_reloc, syms), [])

    # An off-grid candidate whose reloc IS a real index must be refused rather than reported at the
    # address it happens to sit at. This is the grid guard's actual job: the entries are a fixed
    # stride apart, so anything not on that stride is not one of them, however well-formed it looks.
    # Built three bytes off the stride, with reloc 2 -- a reloc the table knows, and one whose real
    # entry is intact and elsewhere, so a fabricated row here would contradict a row we can see.
    off_grid = entry(0, 0) + entry(1, 1) + entry(2, 2) + entry(3, 3) + b"\x90" * 3 + \
        JUNK6 + b"\x68" + _st.pack("<I", 2) + b"\xe9" + \
        _st.pack("<i", RESOLVER - (BASE + 4 * 16 + 3 + 16))
    check("off-grid-candidate-is-not-reported",
          S.patched_import_stubs(FakePlt(off_grid), syms), [])

    # CALIBRATION. A PLT whose entries have a shape this code does not know -- here an `endbr64`
    # prologue that shifts the `push` off byte 6 -- contains no intact entry to compare against, so
    # every entry looks patched. Reporting it would be the loudest possible false accusation, so the
    # answer must be silence. Mutation killed: dropping the "at least one intact entry" guard.
    ENDBR = b"\xf3\x0f\x1e\xfa"

    def cet_entry(k, reloc):
        rel = RESOLVER - (BASE + k * 16 + 16)
        return ENDBR + b"\x68" + _st.pack("<I", reloc) + b"\xe9" + _st.pack("<i", rel) + b"\x90\x90"

    cet = FakePlt(b"".join(cet_entry(k, k) for k in range(4)))
    check("unrecognised-plt-shape-is-silence-not-an-accusation",
          S.patched_import_stubs(cet, syms), [])

    # TWO PLT REGIONS with different resolvers. The losing region's entries are real PLT entries, not
    # overwritten slots, so the geometry must not report the gaps they occupy. Mutation killed:
    # omitting the other-region check.
    OTHER = BASE + 0x800

    def other_entry(k, reloc):
        rel = OTHER - (BASE + k * 16 + 16)
        return b"\xff\x25\x00\x00\x00\x00" + b"\x68" + _st.pack("<I", reloc) + \
            b"\xe9" + _st.pack("<i", rel)

    two_plt = FakePlt(entry(0, 0) + other_entry(1, 1) + entry(2, 2) + other_entry(3, 3))
    check("second-plt-region-is-not-reported-as-overwritten",
          S.patched_import_stubs(two_plt, syms), [])

    # A module with no lazy PLT at all (`-z now`: no `push; jmp` tails anywhere) yields no candidates
    # and must report nothing rather than dividing by an empty population. No local dump is built
    # this way, so this arm is the only thing exercising that path.
    now_bound = FakePlt(b"\xff\x25\x00\x00\x00\x00" * 8)
    check("non-lazy-module-with-no-plt-tails-reports-nothing",
          S.patched_import_stubs(now_bound, syms), [])

    # format_patched(): a clean module says so out loud rather than printing nothing, so "clean" and
    # "the tool did not run" are distinguishable.
    check("clean-module-says-so",
          S.format_patched(clean, {}, "eboot.bin"),
          ["eboot.bin: no import stub has been overwritten in place"])

    # A calibration refusal must SAY it did not judge, rather than printing the clean line. Mutation
    # killed: collapsing "uncalibrated" into "clean" in format_patched.
    cet.tags = {}
    check("not-judged-is-distinguishable-from-clean",
          S.format_patched(cet, {}, "eboot.bin", sym_of=syms)[0].split(":")[1].strip()[:10],
          "NOT JUDGED")

    # ...and the DETECTION row is pinned too, because that row is what a reader pastes into an issue.
    # Only the clean line was covered before, so the NID/name lookup and the [tail]/[whole-entry]
    # label could have changed shape unnoticed.
    detected = FakePlt(entry(0, 0) + entry(1, 1, JUNK6) + entry(2, 2) + entry(3, 3))
    detected.tags = {}
    check("detection-row-format-is-pinned",
          S.format_patched(detected, {"BBBBBBBBBBB": ("sceThing", "libSceThing")}, "eboot.bin",
                           sym_of=syms),
          ["eboot.bin: 1 import stub(s) OVERWRITTEN IN PLACE -- calls to these never reach the "
           "HLE layer",
           "0x10010\treloc=1\tBBBBBBBBBBB\tsceThing\tlibSceThing\t[tail]"])

    print("\n%s (%d failure(s))" % ("FAILED" if fails else "all passed", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
