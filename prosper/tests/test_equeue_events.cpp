// test_equeue_events — the sceKernel equeue user/timer event sources (hle_kernel_time.cpp).
//
// These were previously no-op stubs, which silently starved any equeue the guest feeds via user or
// timer events (e.g. Unity's Frame Timing Manager registers a user event id=999 and blocks WaitEqueue
// on it). This verifies the real backend: a triggered user event and an expired timer both post a
// SceKernelEvent that WaitEqueue then returns, with the FreeBSD-style filter ids and udata intact.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// SceKernelEvent (FreeBSD kevent layout, 0x20 bytes): ident@0, filter@8(i16), flags@0xA, fflags@0xC,
// data@0x10, udata@0x18. Must match the backend's struct exactly.
struct KEvent { int64_t ident; int16_t filter; uint16_t flags; uint32_t fflags; int64_t data; uint64_t udata; };
static_assert(sizeof(KEvent) == 0x20, "SceKernelEvent must be 0x20 bytes");

int main() {
    printf("== test_equeue_events ==\n");
    register_builtin_hle();

    auto create  = Hle::lookup(nid_hash("sceKernelCreateEqueue"));
    auto adduser = Hle::lookup(nid_hash("sceKernelAddUserEvent"));
    auto trigger = Hle::lookup(nid_hash("sceKernelTriggerUserEvent"));
    auto addhrt  = Hle::lookup(nid_hash("sceKernelAddHRTimerEvent"));
    auto wait    = Hle::lookup(nid_hash("sceKernelWaitEqueue"));
    CHECK(create && adduser && trigger && addhrt && wait, "all equeue event fns registered");
    if (!(create && adduser && trigger && addhrt && wait)) { printf("== FAIL ==\n"); return 1; }

    // Create an equeue: the handle is written to *arg0.
    uint64_t eq = 0;
    create((uint64_t)(uintptr_t)&eq, 0, 0, 0, 0, 0);
    CHECK(eq != 0, "CreateEqueue produced a handle");

    // --- User event: register id=999 with a udata, trigger it, then WaitEqueue should return it. ---
    const int64_t kId = 999; const uint64_t kUdata = 0xCAFEF00D;
    adduser(eq, (uint64_t)kId, kUdata, 0, 0, 0);
    trigger(eq, (uint64_t)kId, 0, 0, 0, 0);            // udata=0 -> echo the registered udata

    KEvent ev{}; int32_t out = -1; uint32_t timeout_us = 50000;   // 50ms cap; event is already queued
    wait(eq, (uint64_t)(uintptr_t)&ev, 1, (uint64_t)(uintptr_t)&out,
         (uint64_t)(uintptr_t)&timeout_us, 0);
    CHECK(out == 1, "WaitEqueue returned exactly 1 user event");
    CHECK(ev.ident == kId, "user event ident == 999");
    CHECK(ev.filter == -11, "user event filter == EVFILT_USER (-11)");
    CHECK(ev.udata == kUdata, "user event echoed the registered udata");

    // --- HRTimer: arm a ~5ms one-shot; WaitEqueue (100ms cap) should block then return the timer. ---
    int64_t ts[2] = { 0, 5 * 1000 * 1000 };   // 0s + 5,000,000 ns = 5ms
    addhrt(eq, (uint64_t)1234, (uint64_t)(uintptr_t)ts, 0xABCD, 0, 0);
    KEvent tev{}; int32_t tout = -1; uint32_t tcap = 100000;
    wait(eq, (uint64_t)(uintptr_t)&tev, 1, (uint64_t)(uintptr_t)&tout,
         (uint64_t)(uintptr_t)&tcap, 0);
    CHECK(tout == 1, "WaitEqueue returned the timer event");
    CHECK(tev.ident == 1234, "timer event ident == 1234");
    CHECK(tev.filter == -7, "timer event filter == EVFILT_TIMER (-7)");

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
