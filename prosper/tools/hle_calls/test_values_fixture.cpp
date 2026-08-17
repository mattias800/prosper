// Fixture for test_hle_calls_values.py -- a stand-in guest+emulator for `hle_calls --values`.
//
// This is NOT part of prosper. It exists because #2075 could only be reproduced end-to-end: the
// failure was that gdb's `FinishBreakpoint.return_value` is `None` on a binary with no DWARF, so
// every value capture raised and the feature recorded nothing. No recorded-output test can see
// that -- it needs a real gdb, driving a real process, over real symbols.
//
// What it has to reproduce from the emulator, and why each part matters:
//
//   * functions in namespace `prosper` with the six-`unsigned long` signature, because that is
//     exactly what the driver's `nm`+`c++filt` enumeration keys on;
//   * `static` linkage, as the real `HLE(name)` macro emits (nm reports them as 't', not 'T');
//   * `prosper::k_usleep`, the default `--clock` symbol whose entries bound the window;
//   * `s_user_getevent`'s once-per-process LOGIN contract, including the `a0 &&` gate, because the
//     tool's built-in positive control and its `control_eligible` count both read it.
//
// Build it BOTH with and without `-g`: the no-debug-info build is the #2075 condition, and the
// debug build is what proves the two value sources agree rather than one merely being plausible.

#include <cstdint>
#include <unistd.h>

namespace prosper {

// The same shape as src/hle/*.cpp's `HLE(name)` macro: static, six uint64_t, returns uint64_t.
#define FIXTURE_HLE(name)                                                        \
    static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,     \
                         uint64_t a4, uint64_t a5)

// Mirrors hle_service.cpp's s_user_getevent closely enough for the tool's control to apply: the
// initial LOGIN (return 0) is delivered at most once per PROCESS and only when the guest passed a
// non-null out-pointer; every other call answers SCE_USER_SERVICE_ERROR_NO_EVENT.
FIXTURE_HLE(s_user_getevent) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    static int delivered = 0;
    if (a0 && !delivered) {
        delivered = 1;
        return 0;
    }
    return 0x80960007ull;
}

// A value no accident produces, and one that needs all 64 bits: it pins both the capture and the
// mask. A tool that truncated to 32 bits, or read the wrong register, cannot print this.
FIXTURE_HLE(s_fixture_wide) {
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0xDEADBEEF12345678ull;
}

// A plain success, so the histogram has a row whose value is the one this codebase returns most.
FIXTURE_HLE(s_fixture_ok) {
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0;
}

// The clock. Deliberately NOT `s_`-prefixed, so a `--filter '^s_'` run arms it as the window's
// clock only and never as a handler -- exactly how the real k_usleep is used. External linkage so
// the `break prosper::k_usleep` spec resolves from the symbol table alone, with or without DWARF.
void k_usleep() {
    usleep(2000);
}

}  // namespace prosper

using hle_fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

// Through volatile pointers so the calls survive any optimisation level: the test builds this at
// -O0, but a fixture that can be inlined away would fail in a way that looks like a tool bug.
static hle_fn volatile table[3] = {
    &prosper::s_user_getevent,
    &prosper::s_fixture_wide,
    &prosper::s_fixture_ok,
};
static volatile uint64_t sink;

int main() {
    // Runs until the window closes and hle_calls kills it. A fixture that exited first would end
    // the window early (`window=SHORT exited=1`) and the test would be asserting on a truncated run.
    for (;;) {
        uint64_t event = 0;
        sink = table[0]((uint64_t)&event, 0, 0, 0, 0, 0);   // non-null a0: LOGIN-eligible
        sink = table[1](0, 0, 0, 0, 0, 0);
        sink = table[2](0, 0, 0, 0, 0, 0);
        prosper::k_usleep();
    }
}
