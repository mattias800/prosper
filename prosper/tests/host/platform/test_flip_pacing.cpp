// test_flip_pacing — the cadence prosper holds the GUEST's flips to (#3379).
//
// prosper's present is synchronous: the flip path completes the guest's flip the moment the work is
// done, so with no pacing a title whose frame loop is gated on flip completion simulates at HOST
// speed. On a 180 Hz panel that is 3x real time, which is a correctness defect and also makes every
// speed measurement on such a host meaningless.
//
// What is asserted here is the POLICY — which display the period comes from, which divisor the guest
// selected, and which of (env override, harness opt-out, guest-derived default) wins — through
// prosper_vo_flip_pace_period_ns(), a pure function of that state. Asserting a wall-clock frame rate
// instead would be a flake by construction on a loaded host; test_videoout.cpp carries the same note
// about #1770/#1793.
//
// One arm is timed anyway, and deliberately one-sided: N paced flips must take at LEAST the period
// they asked for. A lower bound cannot flake — a loaded host makes a sleep longer, never shorter —
// and it is the only arm that distinguishes "the policy function returns the right number" from "the
// flip path actually waits". Without the fix it measures ~0 ms.
//
// Its own executable rather than an arm of test_videoout, for the same reason test_display_mode is:
// the advertised display mode resolves ONCE per process, and these arms want a process whose
// environment is untouched so the `legacy` 59.94 Hz default is what the derivation is read against.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>   // setenv / _putenv_s
#include <cstring>

using namespace prosper;

extern "C" int      prosper_vo_flip_rate();
extern "C" uint64_t prosper_vo_flip_count();
extern "C" uint64_t prosper_vo_vblank_period_ns();
extern "C" uint64_t prosper_vo_flip_pace_period_ns();
extern "C" void     prosper_vo_set_flip_pacing_unpaced_default();
extern "C" void     prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx,
                                             uint32_t flip_mode, int64_t flip_arg);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_pace_env(const char* value) {
#ifdef _WIN32
    _putenv_s("PROSPER_FLIP_PACE_FPS", value ? value : "");
#else
    if (value) setenv("PROSPER_FLIP_PACE_FPS", value, 1);
    else       unsetenv("PROSPER_FLIP_PACE_FPS");
#endif
}

static uint64_t steady_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    printf("== test_flip_pacing ==\n");
    register_builtin_hle();
    set_pace_env(nullptr);   // start from "nobody named a rate", whatever the caller's env held

    auto open    = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto setrate = Hle::lookup(nid_hash("sceVideoOutSetFlipRate"));
    if (!open || !setrate) { printf("== FAIL: VideoOut entry points missing ==\n"); return 1; }

    const uint64_t handle = open(0, 0, 0, 0, 0, 0);
    CHECK((int64_t)handle > 0, "VideoOutOpen returns a live positive handle");

    // ---- the derivation: the advertised display's vblank, times the guest's own divisor ---------
    // vblank_period_ns is what sceVideoOutWaitVblank and the kevent pump are scheduled on (#3017,
    // #3024), so a flip held for one of these is held for exactly one vblank of the display the
    // guest was told it has. Reading the expected value from the same accessor rather than hardcoding
    // 16683350 is the point: if the advertised mode ever changes, both move together or this reddens.
    const uint64_t vblank = prosper_vo_vblank_period_ns();
    CHECK(vblank > 0, "the advertised display reports a non-zero vblank period");
    CHECK(prosper_vo_flip_rate() == 0, "the flip-rate selector starts at 60 Hz");
    CHECK(prosper_vo_flip_pace_period_ns() == vblank,
          "with no override, a flip is held for ONE advertised vblank period at the 60 Hz selector");

    CHECK(setrate(handle, 1, 0, 0, 0, 0) == 0 &&
              prosper_vo_flip_pace_period_ns() == 2 * vblank,
          "SetFlipRate(30 Hz) doubles the held period -- the rate is read per flip, never cached");
    CHECK(setrate(handle, 2, 0, 0, 0, 0) == 0 &&
              prosper_vo_flip_pace_period_ns() == 3 * vblank,
          "SetFlipRate(20 Hz) triples it");

    // ---- the flip path actually waits ----------------------------------------------------------
    // At the 20 Hz selector the period is three vblanks (~50 ms), so four flips cover three holds.
    // The first flip anchors rather than sleeping (it takes the "we fell behind" re-anchor leg), so
    // the floor is two holds -- and an unpaced loop of four flip_from_gpu calls returns in tens of
    // MICROseconds, so nothing but a real sleep can reach it.
    //
    // The floor is computed from `vblank`, NOT from prosper_vo_flip_pace_period_ns(). That matters:
    // spelling it `elapsed >= 2 * period` reads identically and is VACUOUS on exactly the build this
    // arm exists to catch -- with pacing off the period is 0, the assertion becomes `elapsed >= 0`,
    // and the arm reports [ok] over a run that never slept. Measured while arming the without-fix
    // arm for #3379, which is the only reason it was noticed.
    {
        const uint64_t period = prosper_vo_flip_pace_period_ns();
        const uint64_t floor_ns = 2 * 3 * vblank;
        const uint64_t before = prosper_vo_flip_count();
        const uint64_t t0 = steady_ns();
        for (int i = 0; i < 4; ++i) prosper_vo_flip_from_gpu((uint32_t)handle, -1, 1, i);
        const uint64_t elapsed = steady_ns() - t0;
        CHECK(prosper_vo_flip_count() == before + 4, "all four flips completed");
        CHECK(elapsed >= floor_ns,
              "four flips at the 20 Hz selector take at least two held periods -- the flip path "
              "really sleeps, rather than the policy function merely returning the right number");
        printf("  [info] 4 flips at %llu ns/flip took %llu ns (floor %llu)\n",
               (unsigned long long)period, (unsigned long long)elapsed,
               (unsigned long long)floor_ns);
    }
    CHECK(setrate(handle, 0, 0, 0, 0, 0) == 0, "restored the 60 Hz selector");

    // ---- who wins --------------------------------------------------------------------------
    set_pace_env("120");
    CHECK(prosper_vo_flip_pace_period_ns() == 1000000000ull / 120,
          "an explicit PROSPER_FLIP_PACE_FPS overrides the guest-derived period");
    set_pace_env("0");
    CHECK(prosper_vo_flip_pace_period_ns() == 0,
          "PROSPER_FLIP_PACE_FPS=0 is an explicit, deliberate 'run free'");
    // A typo must keep the default rather than select the OFF arm (#3538): silently handing back
    // the 3x speed-up is exactly the failure this knob exists to remove.
    set_pace_env("sixty");
    CHECK(prosper_vo_flip_pace_period_ns() == vblank,
          "a malformed PROSPER_FLIP_PACE_FPS keeps the guest-derived default, not the OFF arm");
    set_pace_env("1001");
    CHECK(prosper_vo_flip_pace_period_ns() == vblank,
          "an out-of-range rate keeps the guest-derived default too");
    set_pace_env(nullptr);
    CHECK(prosper_vo_flip_pace_period_ns() == vblank,
          "clearing the override returns to the guest-derived period");

    // ---- the measurement harnesses' opt-out ----------------------------------------------------
    // tools/screenshot and tools/boot_trace install this: their pad routes are wall-clock anchored
    // against free-running flips, and the snapshot tool sets the variable explicitly on both halves.
    prosper_vo_set_flip_pacing_unpaced_default();
    CHECK(prosper_vo_flip_pace_period_ns() == 0,
          "a measurement harness can opt the process out of the console cadence");
    set_pace_env("30");
    CHECK(prosper_vo_flip_pace_period_ns() == 1000000000ull / 30,
          "an explicit rate still wins over the harness opt-out");
    set_pace_env(nullptr);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
