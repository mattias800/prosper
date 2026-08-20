// x86_read_decode.hpp — minimal x86-64 decoder for "memory READ into a general register" instructions.
//
// Purpose: the Windows PROSPER_NULL_PAGE path (exec_image_win.cpp) cannot map the reserved bottom-64 KiB
// null dead-zone, so when the guest faults reading a low null-field address it must EMULATE the read as
// returning zero — parity with the Linux SIGSEGV handler's mmap'd zero page. To do that it needs the
// faulting instruction's destination register and total length. This header isolates that decode as a
// pure, host-independent function so it can be unit-tested on any platform (the Windows VEH is not).
//
// Scope is deliberately narrow and conservative: it ONLY recognizes forms where writing the FULL 64-bit
// destination register to zero is provably correct for a zero source operand:
//   * mov r32/r64, r/m   (opcode 0x8B) — 64-bit needs REX.W; 32-bit zero-extends to 64. NOT 16-bit (0x66).
//   * movzx r, r/m8|r/m16 (0F B6 / 0F B7) — always zero-extends into the full dest.
//   * movsx r, r/m8|r/m16 (0F BE / 0F BF) — sign-extends; sign of 0 is 0, so full-dest zero is correct.
// Everything else — 8-bit (0x8A) and 16-bit (0x66-prefixed) partial-register writes, RIP-relative-only
// oddities aside, any other opcode, and register-source (ModRM.mod==3) forms — returns false so the
// caller falls through to the normal fault path and never silently mishandles an instruction.
//
// %fs/%gs-overridden accesses are declined for a reason that is about MEANING, not encoding (#2112). For
// every other prefix the faulting linear address IS the address the guest computed, so "it faulted below
// 64 KiB" really does mean "the guest read a null field" and zeroing the destination reproduces the Linux
// zero page. With an FS/GS override the linear address is segbase + offset, so a fault below 64 KiB means
// OUR segment base is wrong — on Windows the kernel zeroes the user FS base at every kernel transition —
// and the guest's own offset was small, not null. Emulating that read as zero manufactures a value the
// guest never asked for AND consumes the fault that exec_image_win.cpp's guest_fs_reapply() needs to
// restore the base and retry. Measured on Dragon Quest VII (PPSA17942) before this change: the guest's
// `mov rax, %fs:0x0` TCB-self-pointer load was emulated to rax=0 (one `[nullpage] addr=0x0` line), and
// the next instruction, `cmp BYTE PTR [rax-0x10]`, took a fatal access violation at 0xfffffffffffffff0.
// CS/DS/ES/SS overrides (0x2E/0x36/0x3E/0x26) stay ignorable: x86-64 forces those bases to zero, so they
// are genuinely equivalent to no prefix. CONFIDENCE: HIGH.
#pragma once
#include <cstdint>
#include <cstddef>

namespace prosper {

// Decode the instruction at `p` (which must point to at least `avail` readable bytes). On success,
// sets *dest_reg to the x86 register encoding (0=rax,1=rcx,2=rdx,3=rbx,4=rsp,5=rbp,6=rsi,7=rdi,8..15=r8..r15)
// and *insn_len to the instruction's byte length, then returns true. Returns false for any unsupported or
// non-memory-read form, or if the instruction would need more than `avail` bytes.
inline bool decode_low_read_dest(const uint8_t* p, size_t avail, int* dest_reg, int* insn_len) {
    size_t i = 0;
    auto need = [&](size_t n) { return i + n <= avail; };
    bool opsize16 = false;
    // Legacy prefixes. 0x66 selects 16-bit operand size (a partial-register write — declined below).
    for (;;) {
        if (!need(1)) return false;
        uint8_t b = p[i];
        if (b == 0x66) { opsize16 = true; i++; continue; }
        if (b == 0x64 || b == 0x65) return false;   // %fs/%gs: a low fault means a wrong base, not a null read
        if (b == 0x67 || b == 0xF0 || b == 0xF2 || b == 0xF3 ||
            b == 0x2E || b == 0x36 || b == 0x3E || b == 0x26) { i++; continue; }
        break;
    }
    if (!need(1)) return false;
    uint8_t rex = 0;
    if ((p[i] & 0xF0) == 0x40) { rex = p[i]; i++; }
    if (!need(1)) return false;
    uint8_t op = p[i++];
    if (op == 0x0F) {
        if (!need(1)) return false;
        uint8_t op2 = p[i++];
        if (op2 != 0xB6 && op2 != 0xB7 && op2 != 0xBE && op2 != 0xBF) return false;  // not movzx/movsx
    } else if (op != 0x8B) {
        return false;                                            // unsupported (incl. 0x8A 8-bit)
    }
    // A 0x66 operand-size prefix makes the destination a 16-bit partial-register write (which zeroing the
    // full 64-bit register would corrupt) — for both `mov` and movzx/movsx. REX.W (64-bit) overrides 0x66,
    // so decline only the genuine 16-bit-dest case. (0x66 without REX.W: reject; 0x66 + REX.W: 64-bit, ok.)
    if (opsize16 && !(rex & 8)) return false;
    if (!need(1)) return false;
    uint8_t modrm = p[i++];
    int mod = modrm >> 6;
    if (mod == 3) return false;                                  // register source, not a memory read
    int reg = ((modrm >> 3) & 7) | ((rex & 4) ? 8 : 0);
    int rm  = modrm & 7;
    if (rm == 4) {                                               // SIB present
        if (!need(1)) return false;
        uint8_t sib = p[i++];
        if (mod == 0 && (sib & 7) == 5) { i += 4; }              // base==5, mod==0 -> disp32
    } else if (mod == 0 && rm == 5) {
        i += 4;                                                  // RIP-relative disp32
    }
    if (mod == 1) i += 1;                                        // disp8
    else if (mod == 2) i += 4;                                   // disp32
    if (i > avail) return false;
    *dest_reg = reg;
    *insn_len = (int)i;
    return true;
}

} // namespace prosper
