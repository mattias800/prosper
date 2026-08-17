#!/usr/bin/env python3
"""test_nid_gate_scan — unit tests for the classifier behind nid_gate_scan.py.

This exists because a claim of the form "N of the corpus branches on this value, M ignore it" is
only as trustworthy as `classify_window`, and every way that function can be wrong produces a
plausible-looking table rather than an obvious failure. Three of the cases below are regressions
for bugs an independent review found after the first table had already been published:

  * `mov ecx,eax; xor eax,eax; test ecx,ecx` — the compiler parks the result elsewhere and zeroes
    eax as the ENCLOSING function's return value. An eax-only reader calls this live gate `ignored`.
  * `jmp` used to fall through, so the scan walked past an unconditional jump and could classify a
    `cmp` belonging to the next basic block as this call's gate.
  * `and eax,mask; jne` used to land in `ignored`, i.e. in the one bucket whose whole meaning is
    "this site cannot be affected by the answer".

The arm-following cases at the bottom are the same idea one level up. A walk that stops at the first
branch reports a site that gates generically and const-compares *inside its error arm* as a plain
`nonzero` — as insensitive — and re-running it at greater effort cannot change that, because the
case is structurally inexpressible for that walk. So the positive instance is CONSTRUCTED BY HAND
here, outside any corpus, and each one is paired with the LEGACY verdict on the very same bytes:
`const` with arms followed, `nonzero` without. A control drawn from the same instrument as the null
it validates can only ever test the discriminator.

Run directly, or via ctest as `re_nid_gate_classifier`.
"""
import os
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nid_gate_scan as G                                            # noqa: E402

UNLOADED = 0x805A1001
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    print("  [%s] %-42s got=%s" % ("ok  " if ok else "FAIL", name, got))
    if not ok:
        fails += 1
        print("         wanted %s" % (want,))


def bucket(asm_bytes, base=0x1000):
    """Classify a literal byte string as the window after the call. Needs GNU objdump."""
    return G.classify_window(G.disasm(asm_bytes, base), UNLOADED)[0]


class FakeImg:
    """The minimum `block_fetcher` and `stub_entry_points` need: a flat blob at a base VA."""

    def __init__(self, base, blob):
        self.base, self.raw = base, blob

    def foff(self, va):
        off = va - self.base
        return off if 0 <= off < len(self.raw) else None


def follow(asm_bytes, base=0x1000, window=48):
    """Classify literal bytes as a call site WITH arms followed. Needs GNU objdump."""
    fetch = G.block_fetcher(FakeImg(base, asm_bytes), window)
    return G.classify_site_following(fetch, base, UNLOADED)[0]


def twalk(*insns):
    """One follow-mode block over pre-decoded instructions: (bucket, edges, saw_gate)."""
    ins = [(0x1000 + i * 4, m, o) for i, (m, o) in enumerate(insns)]
    bucket, _ev, st = G._walk_block(ins, UNLOADED, {"a"}, False, True, None)
    return bucket, st["edges"], bool(st["gated"])


def tbucket(*insns):
    """Classify pre-decoded instructions — the same classifier, no objdump involved.

    These carry the classifier coverage on a host with no GNU objdump. Without them the skip guard
    below would take the whole file with it and the ctest would report green having tested nothing,
    which is the trap this file's own header is about, one level up (instrument-trap 94)."""
    return G.classify_window([(0x1000 + i * 4, m, o) for i, (m, o) in enumerate(insns)], UNLOADED)[0]


def objdump_free_checks():
    """Everything that does not need a disassembler. ALWAYS runs."""
    # --- canon(): the register normaliser the taint set depends on -------------------------------
    check("canon-eax", G.canon("eax"), "a")
    check("canon-rax", G.canon("rax"), "a")
    check("canon-al", G.canon("al"), "a")
    check("canon-r14d", G.canon("r14d"), "r14")
    check("canon-r8", G.canon("r8"), "r8")
    check("canon-esi", G.canon("esi"), "si")
    check("canon-memory-operand", G.canon("QWORD PTR [rbp-0x30]"), None)
    check("canon-immediate", G.canon("0x805a1001"), None)

    # --- classify_window() over pre-decoded instructions (no disassembler needed) -----------------
    check("t-const-gate", tbucket(("cmp", "eax,0x805a1001")), "const")
    check("t-nonzero-gate", tbucket(("test", "eax,eax")), "nonzero")
    check("t-other-cmp", tbucket(("cmp", "eax,0x5")), "other-cmp")
    check("t-taint-follows-move",
          tbucket(("mov", "ecx,eax"), ("xor", "eax,eax"), ("test", "ecx,ecx")), "nonzero")
    check("t-jmp-ends-window",
          tbucket(("jmp", "0x2000"), ("cmp", "eax,0x805a1001")), "forward")
    check("t-alu-mask-is-a-gate", tbucket(("and", "eax,0x80000000")), "alu-gate")
    check("t-self-xor-is-ignored", tbucket(("xor", "eax,eax"), ("ret", "")), "ignored")
    check("t-call-clobbers", tbucket(("call", "0x2000"), ("ret", "")), "ignored")
    check("t-ret-forwards", tbucket(("ret", "")), "forward")
    check("t-callee-saved-survives-call",
          tbucket(("mov", "r14d,eax"), ("call", "0x2000"), ("test", "r14d,r14d")), "nonzero")
    check("t-undecodable", tbucket(("(bad)", "")), "undecodable")

    # --- classify_window() must NOT follow anything: the legacy walk is the reproducibility path --
    # `--no-follow-arms` exists so every number this tool published before arm following can be
    # re-derived. If the legacy walk quietly gained the new behaviour, that guarantee is gone and
    # nothing above would notice, because the follow-mode answers are the ones people now want.
    check("legacy-stops-at-the-gate",
          tbucket(("test", "eax,eax"), ("jne", "0x2000"), ("cmp", "eax,0x805a1001")), "nonzero")
    check("legacy-stops-at-jmp",
          tbucket(("jmp", "0x2000"), ("cmp", "eax,0x805a1001")), "forward")

    # --- _walk_block(follow=True): the block reports its SUCCESSORS instead of stopping ----------
    check("follow-gate-forks-into-both-arms",
          twalk(("test", "eax,eax"), ("jne", "0x2000"), ("ret", "")), ("edges", [0x2000, 0x1008], True))
    check("follow-cmp-zero-forks-too",
          twalk(("cmp", "eax,0x0"), ("je", "0x2000"), ("ret", "")), ("edges", [0x2000, 0x1008], True))
    check("follow-jmp-is-an-edge", twalk(("jmp", "0x2000")), ("edges", [0x2000], False))
    check("follow-indirect-jmp-is-unresolved", twalk(("jmp", "rax")), ("forward", [], False))
    #  A branch whose fall-through fell off the end of the window with no `fall_va` to continue from
    #  is UNRESOLVED, never a silent "the arm held nothing".
    check("follow-truncated-branch-is-unresolved",
          twalk(("test", "eax,eax"), ("jne", "0x2000")), ("forward", [], True))
    #  A conditional branch on flags this walk did not set still forks: the value is live in both
    #  successors, and exploring only the fall-through is how a const compare hides in the taken arm.
    check("follow-forks-on-an-unrelated-branch",
          twalk(("cmp", "ecx,0x4"), ("jne", "0x2000"), ("ret", "")), ("edges", [0x2000, 0x1008], False))

    # --- stub_entry_points(): the CET prologue case ----------------------------------------------
    # No dump in the local corpus uses a CET-enabled stub, so the re-sweep that left every other
    # number unchanged does NOT constrain this path — it only ever adds call sites, and it never
    # fired. These assertions are the only thing testing it.
    #  stub at 0x1000: f3 0f 1e fa (endbr64) then ff 25 ... at 0x1004
    cet = FakeImg(0x1000, b"\xf3\x0f\x1e\xfa" + b"\xff\x25\x00\x00\x00\x00")
    check("stub-entry-cet-adds-prologue", sorted(G.stub_entry_points(cet, 0x1004)), [0x1000, 0x1004])
    #  a plain stub: the four bytes before the jmp are not endbr64
    plain = FakeImg(0x1000, b"\x90\x90\x90\x90" + b"\xff\x25\x00\x00\x00\x00")
    check("stub-entry-plain-is-jmp-only", G.stub_entry_points(plain, 0x1004), [0x1004])
    #  a stub at the very start of a segment: site-4 is unmapped, must not raise
    edge = FakeImg(0x1000, b"\xff\x25\x00\x00\x00\x00")
    check("stub-entry-unmapped-prologue", G.stub_entry_points(edge, 0x1000), [0x1000])

    # --- arg0_before(): the module-id recovery, a pure byte scan ---------------------------------
    #  bf b4 00 00 00          mov edi,0xb4   immediately before the call at offset 5
    check("arg0-exact", G.arg0_before(b"\xbf\xb4\x00\x00\x00", 5), (0xB4, True))
    #  31 ff                   xor edi,edi
    check("arg0-xor-edi", G.arg0_before(b"\x31\xff", 2), (0, True))
    #  bf b4 00 00 00 90       mov edi,0xb4 ; nop  -> recovered but not exact
    check("arg0-not-exact", G.arg0_before(b"\xbf\xb4\x00\x00\x00\x90", 6), (0xB4, False))
    check("arg0-absent", G.arg0_before(b"\x90\x90\x90\x90", 4), None)

    # --- imported_nids(): the --all-nids enumerator -----------------------------------------------
    # This decides the whole sweep's coverage, and both ways it can be wrong are silent: too strict
    # drops real imports from a census whose entire purpose is to be exhaustive, and too loose pulls
    # ordinary defined symbols in as phantom "imports" that then report zero call sites and read as
    # a clean negative. The `#` alone is not the discriminator — the 11-character NID before it is.
    class FakeSyms(G.Image):
        """A dynsym of `syment`-byte entries whose only field is the u32 st_name, then a strtab."""

        def __init__(self, names):
            syment = 4
            strtab = syment * len(names)
            blob, ents = bytearray(), bytearray()
            for n in names:
                ents += struct.pack("<I", len(blob))
                blob += n.encode() + b"\0"
            self.raw = bytes(ents + blob)
            self._d = (0, strtab, len(names), syment)

        def dynsym(self):
            return self._d

    syms = FakeSyms(["fMP5NHUOaMk#h#h",                      # a real import: 11-char NID then '#'
                     "_ZNK3sce4Json6Object5emptyEv",         # C++ mangled, no '#' at all
                     "short#h#h",                            # '#' present, NID too short
                     "twelvecharss#h#h",                     # '#' present, NID too long
                     "Oad3rvY-NJQ#libSceNpManager#libSceNpManager"])
    check("imported-nids-picks-11-char-nids",
          syms.imported_nids(), [("fMP5NHUOaMk", 0), ("Oad3rvY-NJQ", 4)])
    #  The `#`-less name above is rejected in this fixture because the NEXT string's `#` is far
    #  away, which does not pin the NUL clamp at all. Put a `#`-less name immediately before one
    #  whose `#` lands at exactly distance 11 from it: an unclamped search finds that borrowed `#`
    #  and admits a symbol that has no NID. Both entries must be rejected.
    #  "nohash" is 6 bytes + NUL, so "abcd#efgh" starts at 7 and its '#' sits at absolute 11 —
    #  exactly 11 from "nohash"'s first byte. An unclamped search therefore admits the phantom NID
    #  "nohash\0abcd"; the clamped one rejects both entries. Verified to discriminate: with the
    #  clamp removed this assertion returns [('nohash\0abcd', 0)].
    borrow = FakeSyms(["nohash", "abcd#efgh"])
    check("imported-nids-clamps-hash-search-to-nul", borrow.imported_nids(), [])


def objdump_checks():
    """End-to-end cases: real bytes through objdump, the output regex, and --adjust-vma."""
    # --- the gate idioms -------------------------------------------------------------------------
    #  3d 01 10 5a 80          cmp eax,0x805a1001
    check("const-gate", bucket(b"\x3d\x01\x10\x5a\x80\x75\x10"), "const")
    #  85 c0                   test eax,eax
    check("nonzero-gate", bucket(b"\x85\xc0\x75\x10"), "nonzero")
    #  83 f8 05                cmp eax,0x5   (not the errno, not zero)
    check("other-cmp-gate", bucket(b"\x83\xf8\x05\x75\x10"), "other-cmp")

    # REGRESSION: result moved to ecx, eax then zeroed as the caller's own return value.
    #  89 c1                   mov ecx,eax
    #  31 c0                   xor eax,eax
    #  85 c9                   test ecx,ecx
    check("taint-follows-move", bucket(b"\x89\xc1\x31\xc0\x85\xc9\x75\x10"), "nonzero")

    # REGRESSION: an unconditional jump ends the block. The `cmp` after it is unrelated code.
    #  eb 05                   jmp .+5
    #  3d 01 10 5a 80          cmp eax,0x805a1001
    check("jmp-ends-the-window", bucket(b"\xeb\x05\x3d\x01\x10\x5a\x80"), "forward")

    # REGRESSION: a mask test is a gate, not an ignored result.
    #  25 00 00 00 80          and eax,0x80000000
    check("alu-mask-is-a-gate", bucket(b"\x25\x00\x00\x00\x80\x75\x10"), "alu-gate")

    # --- genuinely dead results ------------------------------------------------------------------
    #  31 c0                   xor eax,eax    (self-xor kills the taint)
    #  c3                      ret
    check("self-xor-then-ret-is-ignored", bucket(b"\x31\xc0\xc3"), "ignored")
    #  b8 07 00 00 00          mov eax,0x7    (overwritten from a non-result source)
    check("overwritten-is-ignored", bucket(b"\xb8\x07\x00\x00\x00\xc3"), "ignored")
    #  e8 00 00 00 00          call .+0       (clobbers every caller-saved register)
    check("call-clobbers-is-ignored", bucket(b"\xe8\x00\x00\x00\x00\xc3"), "ignored")

    # --- results that leave the window still live ------------------------------------------------
    #  c3                      ret            (returned to the caller: the gate is one level up)
    check("ret-forwards", bucket(b"\xc3"), "forward")
    #  48 89 45 d0             mov QWORD PTR [rbp-0x30],rax   (spilled)
    #  e8 00 00 00 00          call .+0
    check("spill-then-call-forwards", bucket(b"\x48\x89\x45\xd0\xe8\x00\x00\x00\x00"), "forward")
    #  41 89 c6                mov r14d,eax   (parked in a callee-saved register across a call)
    #  e8 00 00 00 00          call .+0
    check("callee-saved-survives-call", bucket(b"\x41\x89\xc6\xe8\x00\x00\x00\x00\x85\xc0"),
          "forward")

    # --- a window that cannot be decoded is VOID, not negative ------------------------------------
    check("undecodable-is-its-own-bucket", bucket(b"\x06\x07\x0e\x16"), "undecodable")

    # --- THE POSITIVE INSTANCE, built by hand --------------------------------------------------
    # A site that gates generically and const-compares INSIDE ITS ERROR ARM. This is the class the
    # legacy walk cannot express at all, so it is written out here rather than sampled from a dump:
    # a control drawn from the instrument under test can only confirm that the instrument runs.
    # Each case asserts BOTH verdicts on the same bytes, which is what makes it a discriminator.
    #  1000: 85 c0             test eax,eax
    #  1002: 74 05             je   0x1009          <- success arm
    #  1004: 3d 01 10 5a 80    cmp  eax,0x805a1001  <- ERROR ARM, past the branch
    #  1009: 31 c0             xor  eax,eax
    #  100b: c3                ret
    ARM_CONST = b"\x85\xc0\x74\x05\x3d\x01\x10\x5a\x80\x31\xc0\xc3"
    check("arm-const-found-when-followed", follow(ARM_CONST), "const")
    check("arm-const-INVISIBLE-to-the-legacy-walk", bucket(ARM_CONST), "nonzero")
    #  the same shape comparing a DIFFERENT immediate must not answer `const` — otherwise the case
    #  above would pass for any compare at all, and prove nothing about the value being searched for
    ARM_OTHER = b"\x85\xc0\x74\x05\x3d\x05\x00\x00\x00\x31\xc0\xc3"
    check("arm-other-immediate-is-not-const", follow(ARM_OTHER), "other-cmp")
    #  a `jmp` into a shared tail that const-compares: the 3 hand-read windows of #1640's census had
    #  exactly this shape (`call <import>; jmp <back>`), and the legacy walk stopped at the jmp
    #  1000: eb 02             jmp 0x1004
    #  1002: 90 90             (padding)
    #  1004: 3d 01 10 5a 80    cmp eax,0x805a1001
    TAIL_CONST = b"\xeb\x02\x90\x90\x3d\x01\x10\x5a\x80"
    check("jmp-tail-const-found-when-followed", follow(TAIL_CONST), "const")
    check("jmp-tail-const-INVISIBLE-to-the-legacy-walk", bucket(TAIL_CONST), "forward")

    # --- and the negatives, so `nonzero` is not simply what the follow walk always says ----------
    #  both arms drop the value -> gated and provably insensitive to WHICH non-zero answer
    #  1000: 85 c0  test eax,eax / 1002: 74 03  je 0x1007 / 1004: 31 c0 c3 / 1007: 31 c0 c3
    check("both-arms-dead-is-nonzero", follow(b"\x85\xc0\x74\x03\x31\xc0\xc3\x31\xc0\xc3"), "nonzero")
    #  one arm hands the value back to the caller: NOT cleared, and it must not read as `nonzero`
    #  1000: 85 c0  test eax,eax / 1002: 74 01  je 0x1005 / 1004: c3 (eax live) / 1005: 31 c0 c3
    check("an-arm-that-returns-the-value-is-gate-open",
          follow(b"\x85\xc0\x74\x01\xc3\x31\xc0\xc3"), "gate-open")
    #  an indirect tail-jump cannot be followed at all
    check("indirect-jmp-site-is-forward", follow(b"\xff\xe0"), "forward")
    #  a backward branch must not walk forever — the visited set is the only thing stopping it
    #  1000: 85 c0  test eax,eax / 1002: 75 fc  jne 0x1000 / 1004: c3
    check("self-loop-terminates", follow(b"\x85\xc0\x75\xfc\xc3"), "gate-open")

    # --- disasm(): objdump's wrapped byte column is not an instruction ---------------------------
    # objdump breaks the raw-byte listing after seven bytes and the continuation line carries the
    # same `<addr>:` shape, so an 11-byte instruction used to yield a phantom second "instruction"
    # at a VA INSIDE it. Inert as a mnemonic, poisonous as an address: the walk continues a block
    # from an instruction VA, and continuing from mid-instruction decodes garbage.
    #  c7 84 87 e0 06 00 00 03 00 00 00   mov DWORD PTR [rdi+rax*4+0x6e0],0x3   (11 bytes)
    #  90                                 nop
    long_insn = b"\xc7\x84\x87\xe0\x06\x00\x00\x03\x00\x00\x00\x90"
    check("disasm-no-phantom-from-a-wrapped-byte-column",
          [va for va, _m, _o in G.disasm(long_insn, 0x1000)], [0x1000, 0x100B])

    # --- block_fetcher(): every returned instruction is whole, and next_va is exact ---------------
    # Cutting the blob at the window instead would hand the walk a truncated final instruction,
    # which decodes as `(bad)`/`.byte` — a truncation artifact reported as the `undecodable` bucket,
    # i.e. as a finding. Over-reading one instruction length and discarding what starts past the
    # boundary removes the whole class and yields the continuation VA for free.
    #  1000: 90 90 90                      nop nop nop
    #  1003: 3d 01 10 5a 80                cmp eax,0x805a1001   (straddles a 4-byte window)
    #  1008: 90                            nop
    fetched, nxt = G.block_fetcher(FakeImg(0x1000, b"\x90\x90\x90\x3d\x01\x10\x5a\x80\x90"), 4)(0x1000)
    check("fetch-keeps-only-whole-instructions",
          [(va, mn) for va, mn, _o in fetched],
          [(0x1000, "nop"), (0x1001, "nop"), (0x1002, "nop"), (0x1003, "cmp")])
    check("fetch-next-va-is-the-first-instruction-past-the-window", nxt, 0x1008)

    # --- disasm() itself: addresses, mnemonics, and the output regex ------------------------------
    # Asserting only "it did not raise" is tautological once the capability probe has passed — the
    # probe already ran a decode. Assert the DECODE: two nops at the requested base, which covers
    # --adjust-vma, the objdump output regex, and the toolchain in one line.
    check("disasm-decodes-two-nops-at-base", G.disasm(b"\x90\x90", 0x4000),
          [(0x4000, "nop", ""), (0x4001, "nop", "")])


def main():
    print("== test_nid_gate_scan ==")
    objdump_free_checks()                       # always — never gated on the toolchain
    # Capability, not existence. The first version of this guard ran `objdump --version` and treated
    # success as "usable" — but macOS ships LLVM's objdump as plain `objdump`, and it answers
    # --version happily while rejecting `-b binary`, so the guard passed on a machine where every
    # single decode raises. The macOS CI job caught it. objdump_binary() probes with a real decode.
    #
    # And the guard is scoped to the cases that actually need the decoder. Skipping the whole file
    # would leave macOS reporting this test green while running nothing at all — the same trap one
    # level up, and the reason objdump_free_checks() carries the classifier coverage in tuple form.
    if G.objdump_binary() is None:
        print("  [skip] no GNU objdump able to disassemble a raw binary (LLVM's cannot) — "
              "end-to-end decode cases skipped; classifier cases above still ran")
    else:
        objdump_checks()
    print("== FAILURES: %d ==" % fails if fails else "== ALL CHECKS PASSED ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
