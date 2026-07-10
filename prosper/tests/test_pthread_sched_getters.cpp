// test_pthread_sched_getters — scePthreadGetprio / scePthreadGetaffinity must FILL their out-param.
//
// Both have correct handlers in hle_kernel.cpp (k_getprio writes priority 700; k_attr_getaffinity
// writes an 8-core mask 0xff). hle_kernel_time.cpp ALSO registered them to a bare no-op (k_ok), and
// register_kernel_time_hle() runs AFTER register_kernel_hle(), so — registration being last-write-
// wins — the no-op silently shadowed the real handlers: the getter returned success while never
// writing its out-param, handing the caller uninitialized stack memory (the exact harmful-stub class
// the hle_kernel.cpp comment says was already fixed). This locks the shadowing out: a Get* that
// returns OK MUST have written its out-param. Fails if the no-op registration ever wins again.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_pthread_sched_getters ==\n");
    register_builtin_hle();

    HleFn getprio     = Hle::lookup(nid_hash("scePthreadGetprio"));
    HleFn getaffinity = Hle::lookup(nid_hash("scePthreadGetaffinity"));
    CHECK(getprio != nullptr, "scePthreadGetprio is registered");
    CHECK(getaffinity != nullptr, "scePthreadGetaffinity is registered");
    if (!getprio || !getaffinity) { printf("== FAIL ==\n"); return 1; }

    // scePthreadGetprio(thread, int* prio): must write the priority (Sony default 700), not leave the
    // caller's int untouched. Pre-seed a sentinel that a working handler overwrites.
    const int32_t SENT = (int32_t)0x0BADF00D;
    int32_t prio = SENT;
    uint64_t rp = getprio(1 /*dummy thread*/, (uint64_t)(uintptr_t)&prio, 0, 0, 0, 0);
    CHECK(rp == 0, "scePthreadGetprio returns OK");
    CHECK(prio != SENT, "scePthreadGetprio WROTE the prio out-param (not the shadowing no-op)");
    CHECK(prio == 700, "scePthreadGetprio writes the Sony default priority 700");

    // scePthreadGetaffinity(thread, SceKernelCpumask* mask): must write a usable (non-empty) core mask.
    uint64_t mask = 0xEEEEEEEEEEEEEEEEull;
    uint64_t ra = getaffinity(1 /*dummy thread*/, (uint64_t)(uintptr_t)&mask, 0, 0, 0, 0);
    CHECK(ra == 0, "scePthreadGetaffinity returns OK");
    CHECK(mask != 0xEEEEEEEEEEEEEEEEull, "scePthreadGetaffinity WROTE the mask out-param (not the shadowing no-op)");
    CHECK(mask != 0 && (mask & 0x1) != 0, "scePthreadGetaffinity writes a non-empty core mask (>=1 usable CPU)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
