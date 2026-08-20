#pragma once
#include <cstdint>
#include <cstring>
#include <cstddef>

// %fs-prefixed instruction decode + execute core for the macOS/Rosetta guest-TLS trap emulator
// (try_emulate_fs_access in exec_image_linux.cpp). Rosetta keeps the CPU fs base at 0 and rejects
// wrfsbase, so guest `%fs:disp` accesses fault at linear address == the raw offset; the signal
// handler redirects the access to guest_TP + offset. Extracted as a pure function (same pattern as
// sse4a.hpp) so the decode/register semantics are unit-testable on Linux (tests/test_fs_emu.cpp),
// where the __APPLE__ handler itself never compiles.
//
// Handles the mov family observed in real boots: reg loads/stores (8A/8B/88/89), imm stores
// (C6/C7), and movzx/movsx (0F B6/B7/BE/BF). Anything else reports Unhandled so the caller can
// log the raw form and take the normal (fatal, visible) fault path.
//
// Register-operand encoding (Intel SDM Vol. 2, Table 2-2 + REX rules): for 8-bit operands WITHOUT
// a REX prefix, reg-field encodings 4-7 name AH/CH/DH/BH — bits 15:8 of rAX..rBX — NOT the low
// byte of RSP/RBP/RSI/RDI. Only when any REX prefix is present do 4-7 mean SPL/BPL/SIL/DIL. The
// pre-extraction emulator got this wrong (issue #727): `mov ah, fs:[x]` clobbered RSP's low byte.
// CONFIDENCE: HIGH (SDM register-encoding rules; covered by golden vectors in the test).
namespace prosper {

enum class FsEmuStatus {
    NotFs,        // no %fs prefix (or register-form operand): not a guest-TLS memory access
    Unhandled,    // fs memory access, but an instruction form the emulator does not know
    Handled,      // executed against `mem`; regs updated; insn_len valid
};

struct FsEmuResult {
    FsEmuStatus status;
    size_t insn_len = 0;   // total instruction length (prefixes..imm), valid when Handled
};

// Decode the instruction at `p` and, if it is a recognized %fs-relative memory access, execute it
// with `mem` as the resolved target address and `regs` as the x86-numbered register file
// (rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7, r8..r15). `regs` is mutated in place
// for loads. The caller owns address resolution and range policy for `mem`.
inline FsEmuResult fs_emulate_access(const uint8_t* p, volatile uint8_t* mem, uint64_t regs[16]) {
    size_t i = 0;
    bool fs = false, opsz16 = false;
    for (;; i++) {              // legacy prefixes
        uint8_t b = p[i];
        if (b == 0x64) { fs = true; continue; }                 // %fs override — the one we want
        if (b == 0x66) { opsz16 = true; continue; }
        if (b==0xF2||b==0xF3||b==0x2E||b==0x36||b==0x3E||b==0x26||b==0x65||b==0x67) continue;
        break;
    }
    if (!fs) return { FsEmuStatus::NotFs };
    uint8_t rex = 0;
    if ((p[i] & 0xF0) == 0x40) rex = p[i++];
    bool w = rex & 8; bool rexR = rex & 4;
    uint8_t op = p[i++];
    bool two = (op == 0x0F);
    if (two) op = p[i++];
    uint8_t modrm = p[i++];
    int mod = modrm >> 6;
    int reg = ((modrm >> 3) & 7) | (rexR ? 8 : 0);
    int rm  = modrm & 7;
    if (mod == 3) return { FsEmuStatus::NotFs };   // register operand: not an fs memory access
    size_t sib_off = i;
    if (rm == 4) i++;                            // SIB byte present
    if (mod == 0) {
        if (rm == 5) i += 4;                     // disp32 (no base)
        else if (rm == 4 && (p[sib_off] & 7) == 5) i += 4;
    } else if (mod == 1) i += 1;
    else i += 4;                                 // mod == 2

    // Read a register as an x86 operand of width sz. The no-REX byte forms 4-7 are AH/CH/DH/BH.
    auto rd_reg = [&](int xr, int sz) -> uint64_t {
        if (sz == 1 && !rex && xr >= 4 && xr <= 7) return (regs[xr - 4] >> 8) & 0xff;   // AH/CH/DH/BH
        return regs[xr];
    };
    auto wr_reg = [&](int xr, uint64_t v, int sz) {   // write a value into a reg with x86 width rules
        if (sz == 1 && !rex && xr >= 4 && xr <= 7) {                                    // AH/CH/DH/BH
            uint64_t& hr = regs[xr - 4];
            hr = (hr & ~0xff00ull) | ((v & 0xff) << 8);
            return;
        }
        uint64_t& r = regs[xr];
        if (sz == 8)      r = v;
        else if (sz == 4) r = (uint32_t)v;                    // 32-bit dest zero-extends to 64
        else if (sz == 2) r = (r & ~0xffffull) | (v & 0xffff);
        else              r = (r & ~0xffull)   | (v & 0xff);
    };
    auto rd_mem = [&](int sz) -> uint64_t {
        uint64_t v = 0; for (int k = 0; k < sz; k++) v |= (uint64_t)mem[k] << (8*k); return v;
    };
    auto wr_mem = [&](uint64_t v, int sz) { for (int k = 0; k < sz; k++) mem[k] = (uint8_t)(v >> (8*k)); };

    int sz = w ? 8 : (opsz16 ? 2 : 4);
    bool handled = false;
    if (!two) {
        switch (op) {
            case 0x8B: wr_reg(reg, rd_mem(sz), sz); handled = true; break;            // mov r, [fs:m]
            case 0x8A: wr_reg(reg, rd_mem(1), 1);  handled = true; break;            // mov r8, [fs:m8]
            case 0x89: wr_mem(rd_reg(reg, sz), sz); handled = true; break;           // mov [fs:m], r
            case 0x88: wr_mem(rd_reg(reg, 1), 1);  handled = true; break;            // mov [fs:m8], r8
            case 0xC7: { int isz = opsz16 ? 2 : 4; int32_t im = 0; memcpy(&im, p + i, isz); i += isz;
                         wr_mem(w ? (uint64_t)(int64_t)im : (uint64_t)(uint32_t)im, sz); handled = true; break; } // mov [fs:m], imm
            case 0xC6: { uint8_t imm = p[i++]; wr_mem(imm, 1); handled = true; break; }  // mov [fs:m8], imm8
            default: break;
        }
    } else {
        switch (op) {
            // Destination width is the effective operand size `sz` (= 8 under REX.W, 2 under a 0x66
            // operand-size prefix, else 4). The source width is fixed by the opcode (m8/m16). Using a
            // bare w?8:4 here ignored 0x66 and wrote a 32-bit result into a 16-bit destination, clobbering
            // all of bits 63:16 (zeroing 63:32 and overwriting 31:16) instead of merging into the low 16
            // and preserving 63:16 (#1326).
            case 0xB6: wr_reg(reg, rd_mem(1), sz); handled = true; break;          // movzx r, m8
            case 0xB7: wr_reg(reg, rd_mem(2), sz); handled = true; break;          // movzx r, m16
            case 0xBE: wr_reg(reg, (uint64_t)(int64_t)(int8_t)rd_mem(1), sz);  handled = true; break;  // movsx r, m8
            case 0xBF: wr_reg(reg, (uint64_t)(int64_t)(int16_t)rd_mem(2), sz); handled = true; break;  // movsx r, m16
            default: break;
        }
    }
    if (!handled) return { FsEmuStatus::Unhandled };
    return { FsEmuStatus::Handled, i };
}

} // namespace prosper
