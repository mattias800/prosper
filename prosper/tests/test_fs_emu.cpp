// Golden vectors for the %fs trap-emulation decode core (src/host/fs_emu.hpp) — the macOS/Rosetta
// guest-TLS emulator's instruction semantics, unit-tested on Linux where the handler itself never
// compiles. Includes the issue #727 regression: no-REX byte forms with reg 4-7 are AH/CH/DH/BH,
// not SPL/BPL/SIL/DIL (`mov ah, fs:[x]` must set AH, not clobber RSP's low byte).
#include "../src/host/fs_emu.hpp"
#include <cstdio>
#include <cstring>

using prosper::fs_emulate_access;
using prosper::FsEmuStatus;

static int g_fail = 0;
#define CHECK(cond, name) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s (%s:%d)\n", name, __FILE__, __LINE__); g_fail = 1; } \
} while (0)

enum { RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15 };

struct Ctx {
    uint64_t regs[16];
    uint8_t mem[16];
    Ctx() { for (int k = 0; k < 16; k++) regs[k] = 0x1111111111111111ull * (k + 1); memset(mem, 0, sizeof mem); }
};

int main() {
    // --- mov eax, fs:[disp32]  (64 8b 04 25 imm32): 4-byte load, zero-extends to 64 ---
    {
        Ctx c; c.mem[0]=0x78; c.mem[1]=0x56; c.mem[2]=0x34; c.mem[3]=0x12; c.mem[4]=0xff;
        const uint8_t insn[] = { 0x64, 0x8b, 0x04, 0x25, 0x30, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 8, "mov32 handled+len");
        CHECK(c.regs[RAX] == 0x12345678ull, "mov32 zero-extends");
    }
    // --- REX.W mov rax, fs:[disp32]  (64 48 8b 04 25 imm32): 8-byte load ---
    {
        Ctx c; for (int k = 0; k < 8; k++) c.mem[k] = (uint8_t)(0xA0 + k);
        const uint8_t insn[] = { 0x64, 0x48, 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 9, "mov64 handled+len");
        CHECK(c.regs[RAX] == 0xA7A6A5A4A3A2A1A0ull, "mov64 value");
    }
    // --- #727 regression: mov ah, fs:[disp]  (64 8a 24 25 imm32, reg=4, NO REX) ---
    // Must set AH (bits 15:8 of RAX); RSP must be untouched.
    {
        Ctx c; c.mem[0] = 0x5A;
        const uint8_t insn[] = { 0x64, 0x8a, 0x24, 0x25, 0x10, 0x00, 0x00, 0x00 };
        uint64_t rsp_before = c.regs[RSP], rax_before = c.regs[RAX];
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled, "mov ah handled");
        CHECK(c.regs[RSP] == rsp_before, "mov ah leaves RSP alone");
        CHECK(c.regs[RAX] == ((rax_before & ~0xff00ull) | 0x5A00ull), "mov ah sets AH");
    }
    // --- #727 regression: mov fs:[disp], bh  (64 88 3c 25 imm32, reg=7, NO REX) ---
    // Must store BH (bits 15:8 of RBX), not RDI's low byte.
    {
        Ctx c; c.regs[RBX] = 0x0000000000BEEF00ull; c.regs[RDI] = 0x77;
        const uint8_t insn[] = { 0x64, 0x88, 0x3c, 0x25, 0x10, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled, "mov ,bh handled");
        CHECK(c.mem[0] == 0xEF, "mov ,bh stores BH");
    }
    // --- WITH a REX prefix, reg 4 is SPL: 64 40 8a 24 25 imm32 loads into RSP's low byte ---
    {
        Ctx c; c.mem[0] = 0xCD;
        const uint8_t insn[] = { 0x64, 0x40, 0x8a, 0x24, 0x25, 0x10, 0x00, 0x00, 0x00 };
        uint64_t rsp_before = c.regs[RSP];
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled, "mov spl handled");
        CHECK(c.regs[RSP] == ((rsp_before & ~0xffull) | 0xCD), "REX mov spl writes RSP low byte");
    }
    // --- mov fs:[disp], ecx  (64 89 0c 25 imm32): 4-byte store ---
    {
        Ctx c; c.regs[RCX] = 0xDDCCBBAA99887766ull;
        const uint8_t insn[] = { 0x64, 0x89, 0x0c, 0x25, 0x00, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled, "store32 handled");
        CHECK(c.mem[0]==0x66 && c.mem[1]==0x77 && c.mem[2]==0x88 && c.mem[3]==0x99 && c.mem[4]==0, "store32 bytes");
    }
    // --- REX.W mov fs:[disp], imm32 (64 48 c7 04 25 imm32 imm32): sign-extended 8-byte store ---
    {
        Ctx c;
        const uint8_t insn[] = { 0x64, 0x48, 0xc7, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00,
                                 0xfe, 0xff, 0xff, 0xff };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 13, "imm64 handled+len");
        uint64_t v; memcpy(&v, c.mem, 8);
        CHECK(v == 0xFFFFFFFFFFFFFFFEull, "imm sign-extends under REX.W");
    }
    // --- 66-prefixed imm16 store: 64 66 c7 04 25 imm32 imm16 ---
    {
        Ctx c;
        const uint8_t insn[] = { 0x64, 0x66, 0xc7, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0x34, 0x12 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 11, "imm16 handled+len");
        CHECK(c.mem[0] == 0x34 && c.mem[1] == 0x12 && c.mem[2] == 0, "imm16 bytes");
    }
    // --- movzx / movsx ---
    {
        Ctx c; c.mem[0] = 0x80;
        const uint8_t zx[] = { 0x64, 0x0f, 0xb6, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(zx, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && c.regs[RAX] == 0x80ull, "movzx m8 zero-extends");
        const uint8_t sx[] = { 0x64, 0x48, 0x0f, 0xbe, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        r = fs_emulate_access(sx, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && c.regs[RAX] == 0xFFFFFFFFFFFFFF80ull, "movsx m8 sign-extends");
    }
    // --- #1326: 0x66-prefixed movzx/movsx have a 16-bit destination — must MERGE into the low 16 bits
    //     and preserve register bits 63:16 (the bug wrote a 32-bit result, zeroing 31:16 and 63:32). ---
    {
        Ctx c; c.mem[0] = 0x80;   // RAX starts at 0x1111111111111111
        const uint8_t zx16[] = { 0x64, 0x66, 0x0f, 0xb6, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(zx16, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 10, "movzx16 handled+len");
        CHECK(c.regs[RAX] == 0x1111111111110080ull, "movzx ax zero-extends into 16-bit dest, preserving bits 63:16");
        const uint8_t sx16[] = { 0x64, 0x66, 0x0f, 0xbe, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        r = fs_emulate_access(sx16, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && c.regs[RAX] == 0x111111111111FF80ull,
              "movsx ax sign-extends into 16-bit dest, preserving bits 63:16");
    }
    // --- disp8 form length: mov eax, fs:[rax+0x10] is mod=1 rm=0 (64 8b 40 10) ---
    {
        Ctx c; c.mem[0] = 0x42;
        const uint8_t insn[] = { 0x64, 0x8b, 0x40, 0x10 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Handled && r.insn_len == 4, "disp8 length");
        CHECK(c.regs[RAX] == 0x42ull, "disp8 load");
    }
    // --- non-fs instruction bails as NotFs; fs register-form (mod=3) bails as NotFs ---
    {
        Ctx c;
        const uint8_t plain[] = { 0x8b, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        CHECK(fs_emulate_access(plain, c.mem, c.regs).status == FsEmuStatus::NotFs, "no prefix -> NotFs");
        const uint8_t regform[] = { 0x64, 0x8b, 0xc1 };
        CHECK(fs_emulate_access(regform, c.mem, c.regs).status == FsEmuStatus::NotFs, "mod3 -> NotFs");
    }
    // --- unknown fs form (add [fs:m], eax = 64 01 04 25 ...) reports Unhandled, touches nothing ---
    {
        Ctx c;
        const uint8_t insn[] = { 0x64, 0x01, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00 };
        auto r = fs_emulate_access(insn, c.mem, c.regs);
        CHECK(r.status == FsEmuStatus::Unhandled, "add -> Unhandled");
        CHECK(c.mem[0] == 0, "unhandled leaves memory alone");
    }

    if (!g_fail) printf("fs_emu: all golden vectors passed\n");
    return g_fail;
}
