// test_module_entry_abi -- a module entry runs with System V arguments and leaves the host's
// callee-saved registers intact.
//
// THE DEFECT. Deferred module init (and runtime module load) called the guest entry through a plain C
// function pointer. Guest code is System V; on Windows the host ABI is Microsoft x64, so the plain call
// delivered `args`/`argp` in RCX/RDX instead of RDI/RSI and let a SysV guest clobber RBX/RSI/RDI/R12-R15
// that Microsoft x64 requires preserved. The established bridge (`prosper_call_guest_sysv`) does both.
//
// WHAT EACH TEST KILLS:
//   DeliversSysvArguments        a plain-pointer call: the stub would read RDI/RSI as garbage on Windows
//   PreservesHostCalleeSavedRegs the stub clobbers RSI/RDI/XMM6/7; a missing save/restore corrupts the loop
#include "host/image/runtime_module_load.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

struct Slots { uint64_t arg0, arg1; };

void* make_exec(const uint8_t* code, size_t n) {
#ifdef _WIN32
    void* p = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) p = nullptr;
#endif
    if (p) std::memcpy(p, code, n);
    return p;
}

// System V stub: store rdi/rsi to *slots, clobber RSI/RDI/XMM6/XMM7 (volatile in SysV, nonvolatile in Microsoft x64), return 0x77.
std::vector<uint8_t> stub_for(Slots* slots) {
    std::vector<uint8_t> c = { 0x48, 0xB8 };                       // mov rax, imm64
    const uint64_t a = reinterpret_cast<uint64_t>(slots);
    for (int i = 0; i < 8; ++i) c.push_back(static_cast<uint8_t>(a >> (8 * i)));
    const uint8_t rest[] = {
        0x48, 0x89, 0x38,                                          // mov [rax], rdi
        0x48, 0x89, 0x70, 0x08,                                    // mov [rax+8], rsi
        0x66, 0x0F, 0x76, 0xF6,                                    // pcmpeqd xmm6, xmm6 (all ones)
        0x66, 0x0F, 0x76, 0xFF,                                    // pcmpeqd xmm7, xmm7
        0x48, 0xC7, 0xC6, 0xFF, 0xFF, 0xFF, 0xFF,                  // mov rsi, -1
        0x48, 0xC7, 0xC7, 0xFF, 0xFF, 0xFF, 0xFF,                  // mov rdi, -1
        0xB8, 0x77, 0x00, 0x00, 0x00,                              // mov eax, 0x77
        0xC3,                                                      // ret
    };
    c.insert(c.end(), rest, rest + sizeof rest);
    return c;
}

} // namespace

TEST(ModuleEntryAbi, DeliversSysvArguments) {
    Slots slots{};
    const auto code = stub_for(&slots);
    void* fn = make_exec(code.data(), code.size());
    ASSERT_NE(fn, nullptr);
    const uint64_t r = prosper::call_guest_module_entry(reinterpret_cast<uint64_t>(fn),
                                                        0x1122334455667788ull, 0x99AABBCCDDEEFF00ull, 0);
    EXPECT_EQ(r, 0x77u);
    EXPECT_EQ(slots.arg0, 0x1122334455667788ull) << "args must arrive in RDI";
    EXPECT_EQ(slots.arg1, 0x99AABBCCDDEEFF00ull) << "argp must arrive in RSI";
}

TEST(ModuleEntryAbi, PreservesHostCalleeSavedRegs) {
    Slots slots{};
    const auto code = stub_for(&slots);
    void* fn = make_exec(code.data(), code.size());
    ASSERT_NE(fn, nullptr);
    // Values live across the calls in registers the stub clobbers; a missing restore changes them.
    volatile uint64_t a = 1, b = 2, c = 3, d = 4, e = 5, f = 6;
    uint64_t sum = 0;
    double x = 1.0, y = 2.0;   // the compiler keeps these in XMM6+ across the call
    for (uint64_t i = 0; i < 64; ++i) {
        x = x * 1.25 + 0.5; y = y * 0.5 + 1.0;
        sum += prosper::call_guest_module_entry(reinterpret_cast<uint64_t>(fn), i, i + 1, 0);
        a = a + 1; b = b + 2; c = c + 3; d = d + 4; e = e + 5; f = f + 6;
    }
    EXPECT_EQ(sum, 64u * 0x77u);
    EXPECT_EQ(a, 65u); EXPECT_EQ(b, 130u); EXPECT_EQ(c, 195u);
    double ex = 1.0, ey = 2.0;
    for (int i = 0; i < 64; ++i) { ex = ex * 1.25 + 0.5; ey = ey * 0.5 + 1.0; }
    EXPECT_EQ(x, ex); EXPECT_EQ(y, ey);
    EXPECT_EQ(d, 260u); EXPECT_EQ(e, 325u); EXPECT_EQ(f, 390u);
}
