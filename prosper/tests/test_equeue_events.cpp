// test_equeue_events — the sceKernel equeue user/timer event sources (hle_kernel_time.cpp).
//
// These were previously no-op stubs, which silently starved any equeue the guest feeds via user or
// timer events (e.g. Unity's Frame Timing Manager registers a user event id=999 and blocks WaitEqueue
// on it). This verifies the real backend: a triggered user event and an expired timer both post a
// SceKernelEvent that WaitEqueue then returns, with the FreeBSD-style filter ids and udata intact.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>

using namespace prosper;

namespace prosper { void prosper_eq_trigger_eop(); }

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// SceKernelEvent (FreeBSD kevent layout, 0x20 bytes): ident@0, filter@8(i16), flags@0xA, fflags@0xC,
// data@0x10, udata@0x18. Must match the backend's struct exactly.
struct KEvent { int64_t ident; int16_t filter; uint16_t flags; uint32_t fflags; int64_t data; uint64_t udata; };
static_assert(sizeof(KEvent) == 0x20, "SceKernelEvent must be 0x20 bytes");

int main() {
    printf("== test_equeue_events ==\n");
#ifdef _WIN32
    _putenv_s("PROSPER_EOP_SYNC", "1");
#else
    setenv("PROSPER_EOP_SYNC", "1", 1);
#endif
    register_builtin_hle();

    auto create  = Hle::lookup(nid_hash("sceKernelCreateEqueue"));
    auto adduser = Hle::lookup(nid_hash("sceKernelAddUserEvent"));
    auto trigger = Hle::lookup(nid_hash("sceKernelTriggerUserEvent"));
    auto addhrt  = Hle::lookup(nid_hash("sceKernelAddHRTimerEvent"));
    auto wait    = Hle::lookup(nid_hash("sceKernelWaitEqueue"));
    auto getcount = Hle::lookup(nid_hash("sceKernelGetEventCount"));
    CHECK(create && adduser && trigger && addhrt && wait && getcount,
          "all equeue event fns registered");
    if (!(create && adduser && trigger && addhrt && wait && getcount)) {
        printf("== FAIL ==\n"); return 1;
    }

    // Create an equeue: the handle is written to *arg0.
    uint64_t eq = 0;
    create((uint64_t)(uintptr_t)&eq, 0, 0, 0, 0, 0);
    CHECK(eq != 0, "CreateEqueue produced a handle");

    // #987 diagnostic path: observe one completion before any EOP source is registered. The gated
    // counter/log must remain observation-only; registration below must not receive a phantom replay.
    prosper_eq_trigger_eop();

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
    CHECK(tev.filter == -15, "HR-timer event filter == EVFILT_HRTIMER (-15)");

    // --- GPU EOP event (sceGnmAddEqEvent, NID b0xyllnVY-I): register id=0x40 (GfxEop) on the equeue,
    //     then a completed SubmitDcb must fire it — our fold is synchronous, so submit == pipe drain. ---
    auto addeq  = Hle::lookup("b0xyllnVY-I");   // sceGnmAddEqEvent / GraphicsAddEqEvent (raw NID)
    auto submit = Hle::lookup("UglJIZjGssM");   // sceAgcDriverSubmitDcb (raw NID)
    CHECK(addeq && submit, "GnmAddEqEvent + SubmitDcb registered");
    if (addeq && submit) {
        const int64_t kEop = 0x40; const uint64_t kEopUdata = 0x1234BEEF;
        addeq(eq, (uint64_t)kEop, kEopUdata, 0, 0, 0);
        CHECK(getcount(eq, 0, 0, 0, 0, 0) == 0,
              "pre-registration EOP diagnostic does not replay or alter queue behavior");
        // Minimal valid Dcb: a single DRAW_RESET packet (2 dwords). header = 0xC0000000 | (op<<8) | (r<<2).
        uint32_t dcbbuf[2] = { 0xC0000000u | (0x10u << 8) | (0x05u << 2), 0 };  // IT_NOP=0x10, R_DRAW_RESET=0x05
        struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; } pkt{ dcbbuf, 2, {0,0,0,0} };
        submit((uint64_t)(uintptr_t)&pkt, 0, 0, 0, 0, 0);

        KEvent gev{}; int32_t gout = -1; uint32_t gcap = 50000;
        wait(eq, (uint64_t)(uintptr_t)&gev, 1, (uint64_t)(uintptr_t)&gout, (uint64_t)(uintptr_t)&gcap, 0);
        CHECK(gout == 1, "WaitEqueue returned the EOP event after a completed submit");
        CHECK(gev.ident == kEop, "EOP event ident == 0x40 (GfxEop)");
        CHECK(gev.filter == -14, "EOP event filter == GraphicsCore (-14)");
        CHECK(gev.data == kEop, "EOP event data == id (sceGnmGetEqEventType semantics)");
        CHECK(gev.udata == kEopUdata, "EOP event echoed the registered udata");
    }

    // VideoOut flip event accessor (#394 F5): the producer stores the submitted flipArg raw in
    // kevent.data, so GetEventData must preserve all 64 bits rather than decode a packed value.
    auto vo_open   = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto addflip   = Hle::lookup(nid_hash("sceVideoOutAddFlipEvent"));
    auto submitflp = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    auto vo_evid   = Hle::lookup("U2JJtSqNKZI");
    auto vo_evdata = Hle::lookup("rWUTcKdkUzQ");
    CHECK(vo_open && addflip && submitflp && vo_evid && vo_evdata,
          "VideoOut flip event producer and accessors registered");
    if (vo_open && addflip && submitflp && vo_evid && vo_evdata) {
        const uint64_t handle = vo_open(0, 0, 0, 0, 0, 0);
        constexpr uint64_t kFlipArg = 0x8000000000001234ull;
        constexpr uint64_t kFlipUdata = 0xFACEB00Cull;
        addflip(eq, handle, kFlipUdata, 0, 0, 0);
        CHECK((uint32_t)submitflp(handle, (uint64_t)(int64_t)-2, 0, kFlipArg, 0, 0) ==
                  0x8029000au && getcount(eq, 0, 0, 0, 0, 0) == 0,
              "invalid low flip index does not post a VideoOut completion event");
        CHECK((uint32_t)submitflp(handle, 16, 0, kFlipArg, 0, 0) == 0x8029000au &&
                  getcount(eq, 0, 0, 0, 0, 0) == 0,
              "invalid high flip index does not post a VideoOut completion event");
        submitflp(handle, 0, 0, kFlipArg, 0, 0);
        CHECK(getcount(eq, 0, 0, 0, 0, 0) == 1,
              "valid flip posts one VideoOut completion event");

        KEvent fev{}; int32_t fout = -1; uint32_t fcap = 50000;
        wait(eq, (uint64_t)(uintptr_t)&fev, 1, (uint64_t)(uintptr_t)&fout,
             (uint64_t)(uintptr_t)&fcap, 0);
        CHECK(fout == 1 && fev.ident == 0 && fev.filter == -13 && fev.udata == kFlipUdata,
              "WaitEqueue returns the registered VideoOut flip event");
        int64_t decoded = 0;
        CHECK(vo_evid((uint64_t)(uintptr_t)&fev, 0, 0, 0, 0, 0) == 0 &&
              vo_evdata((uint64_t)(uintptr_t)&fev, (uint64_t)(uintptr_t)&decoded,
                        0, 0, 0, 0) == 0 &&
              (uint64_t)decoded == kFlipArg,
              "VideoOut accessors return flip id and the raw 64-bit flipArg");

        decoded = 0x1122334455667788ll;
        CHECK((uint32_t)vo_evdata(0, (uint64_t)(uintptr_t)&decoded, 0, 0, 0, 0) ==
                  0x80290002u && decoded == 0x1122334455667788ll,
              "GetEventData rejects a null event without changing output");
        CHECK((uint32_t)vo_evdata((uint64_t)(uintptr_t)&fev, 0, 0, 0, 0, 0) ==
                  0x80290002u,
              "GetEventData rejects a null output pointer");
        KEvent non_video{}; non_video.filter = -14; non_video.data = 0x55;
        CHECK((uint32_t)vo_evdata((uint64_t)(uintptr_t)&non_video,
                                  (uint64_t)(uintptr_t)&decoded, 0, 0, 0, 0) ==
                  0x8029000du && decoded == 0x1122334455667788ll,
              "GetEventData rejects a non-VideoOut event without changing output");
    }

    // --- Invalid WaitEqueue arguments (#388): num < 1 is EINVAL and a null event array is EFAULT.
    //     Neither failure may consume a ready event; the next valid wait must still receive it.
    {
        trigger(eq, (uint64_t)kId, 0x3333, 0, 0, 0);
        KEvent iev{}; int32_t iout = -1; uint32_t icap = 2000;
        uint64_t ri = wait(eq, (uint64_t)(uintptr_t)&iev, 0,
                           (uint64_t)(uintptr_t)&iout, (uint64_t)(uintptr_t)&icap, 0);
        CHECK((uint32_t)ri == 0x80020016u && iout == 0,
              "WaitEqueue num<1 returns EINVAL with 0 events");
        iout = -1;
        ri = wait(eq, (uint64_t)(uintptr_t)&iev, static_cast<uint64_t>(-1ll),
                  (uint64_t)(uintptr_t)&iout, (uint64_t)(uintptr_t)&icap, 0);
        CHECK((uint32_t)ri == 0x80020016u && iout == 0,
              "WaitEqueue negative count returns EINVAL with 0 events");
        uint64_t rv = wait(eq, (uint64_t)(uintptr_t)&iev, 1,
                           (uint64_t)(uintptr_t)&iout, (uint64_t)(uintptr_t)&icap, 0);
        CHECK(rv == 0 && iout == 1 && iev.udata == 0x3333,
              "invalid-count wait leaves its ready event queued");

        trigger(eq, (uint64_t)kId, 0x4444, 0, 0, 0);
        iout = -1;
        uint64_t rf = wait(eq, 0, 1, (uint64_t)(uintptr_t)&iout,
                           (uint64_t)(uintptr_t)&icap, 0);
        CHECK((uint32_t)rf == 0x8002000eu && iout == -1,
              "WaitEqueue null event array returns EFAULT without changing out");
        iout = -2;
        rf = wait(eq, 0, 0, (uint64_t)(uintptr_t)&iout,
                  (uint64_t)(uintptr_t)&icap, 0);
        CHECK((uint32_t)rf == 0x8002000eu && iout == -2,
              "null event array takes EFAULT precedence over an invalid count");
        rv = wait(eq, (uint64_t)(uintptr_t)&iev, 1,
                  (uint64_t)(uintptr_t)&iout, (uint64_t)(uintptr_t)&icap, 0);
        CHECK(rv == 0 && iout == 1 && iev.udata == 0x4444,
              "null-array wait leaves its ready event queued");
    }

    // --- Real wait semantics (#67): a timed wait that expires with nothing returns ETIMEDOUT
    //     (0x8002003C, Kyty EventQueue.cpp:310), NOT success-with-0-events; an unknown queue
    //     handle returns EBADF (0x80020009), not success. ---
    {
        KEvent xev{}; int32_t xout = -1; uint32_t xcap = 2000;   // 2ms, queue is drained by now
        uint64_t r = wait(eq, (uint64_t)(uintptr_t)&xev, 1, (uint64_t)(uintptr_t)&xout,
                          (uint64_t)(uintptr_t)&xcap, 0);
        CHECK((uint32_t)r == 0x8002003Cu && xout == 0, "timed empty wait returns ETIMEDOUT with 0 events");
        xout = -1;
        uint64_t rb = wait(0xDEAD0000ull, 0, 0, (uint64_t)(uintptr_t)&xout,
                           (uint64_t)(uintptr_t)&xcap, 0);
        CHECK((uint32_t)rb == 0x80020009u && xout == 0,
              "unknown equeue handle takes EBADF precedence and reports 0 events");
    }

    // --- kqueue coalescing (#67): two triggers of the SAME (ident, filter) update one pending
    //     event in place — the wait returns exactly 1 event carrying the NEWEST udata; a queue
    //     stuffed with pumped events can no longer drop a fresh event. ---
    {
        trigger(eq, (uint64_t)kId, 0x1111, 0, 0, 0);
        trigger(eq, (uint64_t)kId, 0x2222, 0, 0, 0);
        KEvent cev[2]{}; int32_t cout = -1; uint32_t ccap = 50000;
        uint64_t r = wait(eq, (uint64_t)(uintptr_t)cev, 2, (uint64_t)(uintptr_t)&cout,
                          (uint64_t)(uintptr_t)&ccap, 0);
        CHECK(r == 0 && cout == 1, "double-trigger coalesces to exactly 1 pending event");
        CHECK(cev[0].udata == 0x2222, "coalesced event carries the NEWEST trigger's udata");
    }

    // --- Timer cancellation (#67): DeleteHRTimerEvent stops a pending one-shot — previously a
    //     no-op, so the "cancelled" timer still fired with possibly-freed udata. ---
    auto deltimer = Hle::lookup(nid_hash("sceKernelDeleteHRTimerEvent"));
    CHECK(deltimer != nullptr, "DeleteHRTimerEvent registered");
    if (deltimer) {
        int64_t cts[2] = { 0, 20 * 1000 * 1000 };   // 20ms one-shot
        addhrt(eq, (uint64_t)777, (uint64_t)(uintptr_t)cts, 0xDEAD, 0, 0);
        deltimer(eq, 777, 0, 0, 0, 0);              // cancel before expiry
        KEvent dev{}; int32_t dout = -1; uint32_t dcap = 60000;   // > the timer's 20ms
        uint64_t r = wait(eq, (uint64_t)(uintptr_t)&dev, 1, (uint64_t)(uintptr_t)&dout,
                          (uint64_t)(uintptr_t)&dcap, 0);
        CHECK((uint32_t)r == 0x8002003Cu && dout == 0, "cancelled timer never fires (wait times out)");
    }

    // --- Delete-while-wait (#67): DeleteEqueue while a thread is blocked in WaitEqueue must WAKE
    //     it with EBADF — the old raw-pointer scheme destroyed the mutex/condvar under the waiter
    //     (use-after-free). The shared_ptr + deleted-flag scheme keeps the state alive. ---
    auto dele = Hle::lookup(nid_hash("sceKernelDeleteEqueue"));
    CHECK(dele != nullptr, "DeleteEqueue registered");
    if (dele) {
        uint64_t eq2 = 0;
        create((uint64_t)(uintptr_t)&eq2, 0, 0, 0, 0, 0);
        uint64_t wret = ~0ull; int32_t wout = -1;
        std::thread waiter([&]{
            KEvent wev{};
            wret = wait(eq2, (uint64_t)(uintptr_t)&wev, 1, (uint64_t)(uintptr_t)&wout, 0 /*infinite*/, 0);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));   // let the waiter block
        dele(eq2, 0, 0, 0, 0, 0);
        waiter.join();   // must not hang, must not crash
        CHECK((uint32_t)wret == 0x80020009u && wout == 0, "DeleteEqueue wakes an infinite waiter with EBADF");
    }

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
