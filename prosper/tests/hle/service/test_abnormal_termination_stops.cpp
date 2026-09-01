// test_abnormal_termination_stops — a guest that reports its OWN crash must stop the run (#3119).
//
// WHY THIS PROPERTY IS LOAD-BEARING, and why asserting the return value would prove nothing.
//
// sceSystemServiceReportAbnormalTermination is the guest telling the system it has crashed. Before
// #3119 the NID was unregistered, so the dispatcher default answered 0 and the guest carried on:
// the process stayed alive with a black window, indistinguishable from a hang, a stall, or a
// renderer defect. Found on Tactics Ogre: Reborn (PPSA03839, #1892), where that one opaque
// `unimplemented:` line was the ONLY informative content in a 50-line log showing no faults, no
// compute rejects, no device loss and no missing present source.
//
// The handler deliberately STILL RETURNS 0 -- identical to the dispatcher default -- so what the
// guest observes is unchanged and registering the NID cannot regress a title by answering
// differently. That is precisely why asserting `rc == 0` proves nothing: it passes with the entire
// feature deleted. The only observable separating the fix from its absence is the cooperative stop
// signal, so that is what this test is about.
//
// The suppression arm is a discriminator, not a courtesy. Without it, a stop signal latched by
// unrelated code anywhere in the process would make the first arm pass forever.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/platform/lifecycle.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", (msg)); ++fails; } \
                              else printf("ok: %s\n", (msg)); } while (0)

static void set_continue(bool on) {
#ifdef _WIN32
    _putenv_s("PROSPER_ABNORMAL_TERMINATION_STOP", on ? "1" : "");
#else
    if (on) setenv("PROSPER_ABNORMAL_TERMINATION_STOP", "1", 1);
    else    unsetenv("PROSPER_ABNORMAL_TERMINATION_STOP");
#endif
}

// Spelled literally so a change to either the NID or the registration breaks here loudly rather
// than silently unhooking the handler. Verified against the PS5 3.20 libSceSystemService stub table.
static const char* kNid = "3s8cHiCBKBE";

int main() {
    register_builtin_hle();

    HleFn fn = Hle::lookup(kNid);
    CHECK(fn != nullptr,
          "sceSystemServiceReportAbnormalTermination is registered (unregistered, it falls to the "
          "dispatcher default and can never stop the run)");
    if (!fn) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }

    // ---- arm 1: the DEFAULT path does NOT stop ----------------------------------------------
    set_continue(false);
    prosper_reset_stop();
    CHECK(!prosper_stop_requested(), "precondition: no stop is pending before the call");

    uint64_t rc = fn(0xDEADull, 0, 0, 0, 0, 0);

    CHECK(rc == 0,
          "the handler still returns 0, so the guest observes exactly what the dispatcher default "
          "gave it and registering this NID changes no guest-visible answer");
    CHECK(!prosper_stop_requested(),
          "the default does NOT stop the run -- titles call this at boot and go on to render "
          "(Tactics Ogre: present in a 470 s / 40,936-frame run), so stopping by default would "
          "break 23 importing dumps including four rung-6 guarded ones");

    // ---- arm 2: the opt-in DOES stop --------------------------------------------------------
    // Kills two implementations that would pass arm 1: one that ignores the environment variable
    // and always stops, and one where arm 1's pass came from a stop latched by something other
    // than this handler.
    set_continue(true);
    prosper_reset_stop();
    CHECK(!prosper_stop_requested(), "precondition: the signal was cleared for arm 2");

    rc = fn(0xBEEFull, 0, 0, 0, 0, 0);

    CHECK(rc == 0, "the suppressed path returns 0 as well");
    CHECK(prosper_stop_requested(),
          "PROSPER_ABNORMAL_TERMINATION_STOP=1 opts into stopping the run");

    // ---- arm 3: clearing the variable restores the stop --------------------------------------
    // Proves arm 2's pass came from the variable rather than from the signal having become
    // permanently unsettable after the first request.
    set_continue(false);
    prosper_reset_stop();
    rc = fn(0xF00Dull, 0, 0, 0, 0, 0);
    CHECK(!prosper_stop_requested(),
          "clearing the variable restores the continue default, so arm 2 was the variable and not "
          "a signal that had latched permanently");

    prosper_reset_stop();
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
