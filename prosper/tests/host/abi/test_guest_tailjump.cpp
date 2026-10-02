// Bare SysV tail transfers must preserve AL, argument registers and the original stack (#4084).
// This register witness does not implement va_start: the compiler-owned variadic readers remain
// covered by guest_varargs/printf_varargs. A zero-low-byte target makes the former RAX jump red.
#include "host/abi/sysv_ms_bridge.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32)
#include <windows.h>
#define GUEST_ABI __attribute__((sysv_abi))
#else
#include <sys/mman.h>
#define GUEST_ABI
#endif

namespace {
using namespace prosper::abi;
struct Capture {
    uint64_t rax{}, r10{}, rsp{}, gp[6]{}, stack[7]{};
    double fp[8]{};
};
static_assert(sizeof(Capture) == 192);
struct Bytes {
    uint8_t* p;
    size_t n{};
    void byte(uint8_t v) { p[n++] = v; }
    void word(uint32_t v) { std::memcpy(p + n, &v, 4); n += 4; }
    void quad(uint64_t v) { std::memcpy(p + n, &v, 8); n += 8; }
    void save(unsigned reg, uint32_t offset) {
        byte(reg >= 8 ? 0x4c : 0x48); byte(0x89);
        byte(uint8_t(0x87 | ((reg & 7) << 3))); word(offset); // mov [rdi+disp32], reg
    }
};
struct Page {
    uint8_t* p{};
    Page() {
#if defined(_WIN32)
        p = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT,
                                               PAGE_EXECUTE_READWRITE));
#else
        void* allocation = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (allocation != MAP_FAILED) p = static_cast<uint8_t*>(allocation);
#endif
    }
    ~Page() {
        if (!p) return;
#if defined(_WIN32)
        VirtualFree(p, 0, MEM_RELEASE);
#else
        munmap(p, 4096);
#endif
    }
};
constexpr uint64_t kReturn = 0x6ab7c8d9e0f12345ull;
constexpr uint64_t kStaticChain = 0x8192a3b4c5d6e7f8ull;
int failures{}, checks{};
void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", name); }
}

// Independent instrumentation handler: record the incoming frame before using RAX as scratch.
// Its address is page+0x200 (low byte zero), not a compiler/linker alignment assumption.
void emit_witness(uint8_t* destination) {
    Bytes b{destination};
    b.save(kRax, 0); b.save(kR10, 8); b.save(kRsp, 16);
    constexpr unsigned regs[] = {kRdi, kRsi, kRdx, kRcx, kR8, kR9};
    for (unsigned i = 0; i < 6; ++i) b.save(regs[i], 24 + 8 * i);
    for (unsigned i = 0; i < 8; ++i) {
        b.byte(0xf2); b.byte(0x0f); b.byte(0x11);
        b.byte(uint8_t(0x87 | (i << 3))); b.word(128 + 8 * i); // movsd [rdi+disp32], xmm[i]
    }
    for (unsigned i = 0; i < 7; ++i) {
        b.byte(0x48); b.byte(0x8b); b.byte(0x44); b.byte(0x24); b.byte(uint8_t(8 + 8 * i));
        b.save(kRax, 72 + 8 * i); // five GP and two FP overflow words, declaration order
    }
    b.byte(0x48); b.byte(0xb8); b.quad(kReturn); b.byte(0xc3);
}

// A test caller prefix sets hidden AL/R10, then transfers without altering the compiler's frame.
// The same prefix addresses the witness directly or through the production bridge.
void emit_caller(uint8_t* destination, uint8_t al, uint64_t target) {
    Bytes b{destination};
    b.byte(0xb0); b.byte(al); // mov al, imm8 (only AL is specified by the SysV contract)
    b.byte(0x49); b.byte(0xba); b.quad(kStaticChain);
    b.byte(0x49); b.byte(0xbb); b.quad(target);
    b.byte(0x41); b.byte(0xff); b.byte(0xe3);
}
using Call = GUEST_ABI uint64_t (*)(Capture*, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                   uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                   double, double, double, double, double, double, double, double,
                                   double, double);
uint64_t invoke(Call fn, Capture& capture) {
    return fn(&capture, 101,102,103,104,105,106,107,108,109,110,
              20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5);
}
void validate(const Capture& c, uint8_t al) {
    check((c.rax & 255) == al, "AL preserved");
    check(c.r10 == kStaticChain, "R10 static-chain carrier preserved");
    check((c.rsp & 15) == 8, "handler entry stack alignment");
    check(c.gp[0] == reinterpret_cast<uintptr_t>(&c), "RDI output pointer preserved");
    for (unsigned i = 1; i < 6; ++i) check(c.gp[i] == 100 + i, "GP register argument preserved");
    for (unsigned i = 0; i < 5; ++i) check(c.stack[i] == 106 + i, "GP overflow argument preserved");
    for (unsigned i = 0; i < 8; ++i) check(c.fp[i] == 20.5 + i, "XMM argument preserved");
    for (unsigned i = 0; i < 2; ++i) {
        double value{}; std::memcpy(&value, &c.stack[5 + i], 8);
        check(value == 28.5 + i, "FP overflow argument preserved");
    }
}
} // namespace
#endif

int main() {
#if defined(__x86_64__) || defined(_M_X64)
    Page page;
    if (!page.p) { std::fprintf(stderr, "FAIL executable allocation\n"); return 1; }
    const auto handler = reinterpret_cast<uintptr_t>(page.p + 0x200);
    check((handler & 255) == 0, "discriminating target low byte zero");
    emit_witness(page.p + 0x200);
    BridgeParams parameters{};
    parameters.handler = handler; parameters.guest_abi = true;
    std::array<uint8_t, kMaxBridgeBytes + 2> staged{};
    staged.fill(0xa5);
    const size_t length = emit_sysv_to_ms_bridge(staged.data() + 1, parameters);
    check(length == 13, "exact tail-jump length");
    check(staged.front() == 0xa5 && staged[14] == 0xa5, "bounded write canaries");
    const uint8_t expected_prefix[] = {0x49,0xbb};
    const uint8_t expected_suffix[] = {0x41,0xff,0xe3};
    check(std::memcmp(staged.data() + 1, expected_prefix, 2) == 0 &&
          std::memcmp(staged.data() + 3, &handler, 8) == 0 &&
          std::memcmp(staged.data() + 11, expected_suffix, 3) == 0, "exact R11-only bytes");
    if (length > kMaxBridgeBytes) return 1;
    std::memcpy(page.p + 0x100, staged.data() + 1, length);
    for (unsigned al = 0; al <= 8; ++al) {
        emit_caller(page.p, uint8_t(al), handler);
        emit_caller(page.p + 0x40, uint8_t(al), reinterpret_cast<uintptr_t>(page.p + 0x100));
#if defined(_WIN32)
        if (!FlushInstructionCache(GetCurrentProcess(), page.p, 4096)) return 1;
#else
        __builtin___clear_cache(reinterpret_cast<char*>(page.p), reinterpret_cast<char*>(page.p + 4096));
#endif
        Capture direct{}, bridged{};
        check(invoke(reinterpret_cast<Call>(page.p), direct) == kReturn, "direct return value");
        validate(direct, uint8_t(al));
        check(invoke(reinterpret_cast<Call>(page.p + 0x40), bridged) == kReturn, "bridged return value");
        validate(bridged, uint8_t(al));
    }
    std::printf("guest_tailjump: %d checks, %d failures (AL 0..8, 10 GP + 10 FP)\n", checks, failures);
    return failures ? 1 : 0;
#else
    std::fprintf(stderr, "guest_tailjump: x86-64 execution required\n");
    return 77;
#endif
}
