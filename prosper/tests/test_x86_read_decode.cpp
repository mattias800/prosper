// test_x86_read_decode - the Windows PROSPER_NULL_PAGE path emulates a faulting low-address READ by
// zeroing the instruction's destination register and advancing past it. decode_low_read_dest() must
// therefore (a) recognize exactly the forms where zeroing the FULL 64-bit dest is correct, (b) report the
// right destination register and instruction length, and (c) REJECT every other form so the caller falls
// through to the real fault path. A wrong length or register would silently corrupt guest state.
#include "../src/host/x86_read_decode.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace prosper;

namespace {
int g_fail = 0;

void expect_ok(const char* name, std::vector<uint8_t> bytes, int want_reg, int want_len) {
    int reg = -1, len = -1;
    bool ok = decode_low_read_dest(bytes.data(), bytes.size(), &reg, &len);
    if (!ok || reg != want_reg || len != want_len) {
        fprintf(stderr, "FAIL %-28s: ok=%d reg=%d(want %d) len=%d(want %d)\n",
                name, (int)ok, reg, want_reg, len, want_len);
        g_fail++;
    }
}

void expect_reject(const char* name, std::vector<uint8_t> bytes) {
    int reg = -1, len = -1;
    if (decode_low_read_dest(bytes.data(), bytes.size(), &reg, &len)) {
        fprintf(stderr, "FAIL %-28s: expected reject but decoded reg=%d len=%d\n", name, reg, len);
        g_fail++;
    }
}
} // namespace

int main() {
    // --- Accepted: the exact instructions observed faulting in DOLL's init walk. ---
    expect_ok("mov (%rdx),%rdx",        {0x48,0x8b,0x12},              2, 3);  // reg=rdx
    expect_ok("mov 0x8(%rdx),%rax",     {0x48,0x8b,0x42,0x08},         0, 4);  // reg=rax, disp8
    // Exact low-address read reached by Beneath when Linux cannot map the diagnostic zero page.
    expect_ok("mov 0x8(%rdx),%rdx",     {0x48,0x8b,0x52,0x08},         2, 4);
    expect_ok("mov disp32(%rdx),%rax",  {0x48,0x8b,0x82,0x34,0x12,0,0},0, 7);  // disp32
    // --- Accepted: register-extension and size variants where full-dest zero is correct. ---
    expect_ok("mov r8,(%rsi) [REX.R]",  {0x4c,0x8b,0x06},              8, 3);  // REX.WR -> r8
    expect_ok("mov ecx,(%rax) [32-bit]",{0x8b,0x08},                   1, 2);  // 32-bit zero-extends
    expect_ok("movzx eax,BYTE(%rsi)",   {0x0f,0xb6,0x06},              0, 3);
    expect_ok("movzx eax,WORD(%rsi)",   {0x0f,0xb7,0x06},              0, 3);
    expect_ok("movsx rax,BYTE(%rsi)",   {0x48,0x0f,0xbe,0x06},         0, 4);
    // --- Accepted: SIB / RIP-relative addressing (length must include the extra bytes). ---
    expect_ok("mov rax,[disp32] (SIB)", {0x48,0x8b,0x04,0x25,0,0,0,0}, 0, 8);  // SIB base=5,mod=0 disp32
    expect_ok("mov rax,[rax+rcx+8]",    {0x48,0x8b,0x44,0x08,0x10},    0, 5);  // SIB + disp8
    expect_ok("mov rax,[rip+0]",        {0x48,0x8b,0x05,0,0,0,0},      0, 7);  // RIP-relative disp32
    // SIB with base==5 at mod==1/mod==2 must still consume its disp8/disp32 (the mod==0 disp32 special
    // case must NOT fire here). rbp/r13-relative addressing is exactly this encoding.
    expect_ok("mov rax,[rbp+0x10] (SIB)",  {0x48,0x8b,0x44,0x25,0x10},          0, 5); // SIB base=5,mod=1
    expect_ok("mov rax,[rbp+disp32] (SIB)",{0x48,0x8b,0x84,0x25,0,0,0,0},        0, 8); // SIB base=5,mod=2
    // REX.W overrides a 0x66 prefix -> 64-bit dest, so full-dest zero is correct and these are accepted.
    expect_ok("mov rax,(%rsi) [66+REX.W]", {0x66,0x48,0x8b,0x06},               0, 4);
    expect_ok("movzx rax,WORD [66+REX.W]", {0x66,0x48,0x0f,0xb7,0x06},          0, 5);

    // --- Rejected: partial-register writes (zeroing the full dest would be wrong). ---
    expect_reject("mov cx,(%rax) [16-bit]", {0x66,0x8b,0x08});
    expect_reject("mov cl,(%rax) [8-bit]",  {0x8a,0x08});
    // 0x66-prefixed movzx/movsx have a 16-bit DEST too — must be rejected (regression guard: the 0F
    // branch originally skipped the opsize16 check, which would have zeroed the full 64-bit register).
    expect_reject("movzx ax,BYTE(%rsi) [66]", {0x66,0x0f,0xb6,0x06});
    expect_reject("movzx ax,WORD(%rsi) [66]", {0x66,0x0f,0xb7,0x06});
    expect_reject("movsx ax,BYTE(%rsi) [66]", {0x66,0x0f,0xbe,0x06});
    // --- Rejected: not a memory read. ---
    expect_reject("mov rax,rcx [reg src]",  {0x48,0x8b,0xc1});         // ModRM.mod==3
    expect_reject("mov (%rsi),%rax [WRITE]",{0x48,0x89,0x06});         // store opcode 0x89
    expect_reject("lea rax,(%rdx)",         {0x48,0x8d,0x02});         // 0x8D not handled
    expect_reject("add rax,(%rdx)",         {0x48,0x03,0x02});         // arithmetic read not handled
    // --- Rejected: truncated instruction (fewer bytes than the encoding needs). ---
    expect_reject("truncated disp8",        {0x48,0x8b,0x42});         // needs a disp8 byte

    if (g_fail) { fprintf(stderr, "test_x86_read_decode: %d failure(s)\n", g_fail); return 1; }
    printf("test_x86_read_decode: all cases passed\n");
    return 0;
}
