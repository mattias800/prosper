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

Run directly, or via ctest as `re_nid_gate_classifier`.
"""
import os
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
    """Assemble-free: classify a literal byte string as the window after the call."""
    return G.classify_window(G.disasm(asm_bytes, base), UNLOADED)[0]


def main():
    print("== test_nid_gate_scan ==")

    # --- canon(): the register normaliser the taint set depends on -------------------------------
    check("canon-eax", G.canon("eax"), "a")
    check("canon-rax", G.canon("rax"), "a")
    check("canon-al", G.canon("al"), "a")
    check("canon-r14d", G.canon("r14d"), "r14")
    check("canon-r8", G.canon("r8"), "r8")
    check("canon-esi", G.canon("esi"), "si")
    check("canon-memory-operand", G.canon("QWORD PTR [rbp-0x30]"), None)
    check("canon-immediate", G.canon("0x805a1001"), None)

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

    # --- arg0_before(): the module-id recovery ---------------------------------------------------
    #  bf b4 00 00 00          mov edi,0xb4   immediately before the call at offset 5
    check("arg0-exact", G.arg0_before(b"\xbf\xb4\x00\x00\x00", 5), (0xB4, True))
    #  31 ff                   xor edi,edi
    check("arg0-xor-edi", G.arg0_before(b"\x31\xff", 2), (0, True))
    #  bf b4 00 00 00 90       mov edi,0xb4 ; nop  -> recovered but not exact
    check("arg0-not-exact", G.arg0_before(b"\xbf\xb4\x00\x00\x00\x90", 6), (0xB4, False))
    check("arg0-absent", G.arg0_before(b"\x90\x90\x90\x90", 4), None)

    # --- disasm() must not swallow an objdump failure --------------------------------------------
    try:
        G.disasm(b"\x90", 0)
        objdump_ok = True
    except Exception:                                                # noqa: BLE001
        objdump_ok = False
    check("disasm-works-on-a-nop", objdump_ok, True)

    print("== FAILURES: %d ==" % fails if fails else "== all checks passed ==")
    return 1 if fails else 0


if __name__ == "__main__":
    # Capability, not existence. The first version of this guard ran `objdump --version` and treated
    # success as "usable" — but macOS ships LLVM's objdump as plain `objdump`, and it answers
    # --version happily while rejecting `-b binary`, so the guard passed on a machine where every
    # single decode raises. The macOS CI job caught it. objdump_binary() probes with a real decode.
    if G.objdump_binary() is None:
        print("no GNU objdump able to disassemble a raw binary (LLVM's cannot) — skipping")
        sys.exit(0)
    sys.exit(main())
