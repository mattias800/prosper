// test_prot_none (#342) — a guest mprotect with prot 0 (SCE_KERNEL_PROT_NONE) must make the page
// INACCESSIBLE, not read-write. host_prot() used to default prot 0 to RW, so a guest guard page stayed
// writable and an overrun never faulted. This drives the real sceKernelMprotect HLE and probes the page
// with a SIGSEGV handler: after prot 0 a read must fault; after prot RW it must be writable again (so
// the common non-zero path is unbroken). Linux-only (mmap/mprotect/signals).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <csetjmp>
#include <csignal>
#include <sys/mman.h>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_faulted;
static void on_segv(int) { g_faulted = 1; siglongjmp(g_jmp, 1); }

// Returns true if READING *p faulted (i.e. the page is not readable).
static bool read_faults(const volatile uint32_t* p) {
    struct sigaction sa{}, old{}; sa.sa_handler = on_segv; sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old);
    g_faulted = 0;
    if (sigsetjmp(g_jmp, 1) == 0) { volatile uint32_t v = *p; (void)v; }
    sigaction(SIGSEGV, &old, nullptr);
    return g_faulted != 0;
}

int main() {
    printf("== test_prot_none ==\n");
    register_builtin_hle();
    auto flexible = Hle::lookup(nid_hash("sceKernelMapFlexibleMemory"));
    auto mprotect_fn = Hle::lookup(nid_hash("sceKernelMprotect"));
    CHECK(flexible && mprotect_fn, "map/mprotect HLE registered");
    if (!(flexible && mprotect_fn)) { printf("== FAIL ==\n"); return 1; }

    auto U = [](const void* p) { return (uint64_t)(uintptr_t)p; };
    const uint64_t LEN = 0x10000;

    // Commit a RW page and confirm it is writable.
    uint64_t va = 0;
    CHECK(flexible(U(&va), LEN, 0x2 /*RW*/, 0, U("prot-test"), 0) == 0 && va, "MapFlexible(RW) succeeds");
    if (!va) { printf("== FAIL ==\n"); return 1; }
    volatile uint32_t* cell = (volatile uint32_t*)(uintptr_t)va;
    *cell = 0xABCD1234u;
    CHECK(!read_faults(cell) && *cell == 0xABCD1234u, "RW page: readable + writable");

    // mprotect to prot 0 -> the page must become PROT_NONE (a read faults). This is the bug: it used to
    // stay RW.
    mprotect_fn(va, LEN, 0 /*SCE_KERNEL_PROT_NONE*/, 0, 0, 0);
    CHECK(read_faults(cell), "after mprotect(prot=0): page is PROT_NONE (read FAULTS, not silently RW)");

    // mprotect back to RW -> writable again (the common non-zero path is unbroken).
    mprotect_fn(va, LEN, 0x2 /*RW*/, 0, 0, 0);
    CHECK(!read_faults(cell), "after mprotect(prot=RW): page is accessible again");
    *cell = 0x5A5A5A5Au;
    CHECK(*cell == 0x5A5A5A5Au, "restored RW page is writable");

    // mprotect to READ-only (prot 1) -> readable but not writable (host_prot maps 1 -> PROT_READ).
    mprotect_fn(va, LEN, 0x1 /*READ*/, 0, 0, 0);
    CHECK(!read_faults(cell), "after mprotect(prot=READ): page is readable");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
