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
        // The real handler writes `{ int32 eventType = 0 (LOGIN), int32 userId = 1 }` here, and only
        // on this one call. That makes it the out-struct control too: over a launch window, exactly
        // ONE of this handler's calls may show a changed out-struct, and it is the call that
        // returned 0.
        int32_t* ev = (int32_t*)a0;
        ev[0] = 0;
        ev[1] = 1;
        return 0;
    }
    return 0x80960007ull;
}

// The out-struct control, mirroring hle_service.cpp's s_syss_getstatus: writes 12 bytes at a0 and
// sets st[6] = 1. #2045 names exactly this handler, because its written bytes are known from the
// source before any run.
FIXTURE_HLE(s_fixture_writes12) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    auto* st = (uint8_t*)a0;
    if (!st) return 0x80A10003ull;
    for (int i = 0; i < 12; i++) st[i] = 0;
    st[6] = 1;
    return 0;
}

// Returns success and writes NOTHING through its out-pointer: the bug shape this feature exists to
// find, and the arm that proves "nothing was written" is visibly distinct from "nothing was read".
FIXTURE_HLE(s_fixture_nowrite) {
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0;
}

// Three ways an argument is not an out-pointer, each of which must be counted apart from the
// others. None of these dereferences a0 -- the fixture only ever passes these values in.
// The one state this method cannot resolve, made deterministic: the guest never clears this
// buffer, and the handler writes the same bytes into it every call. Call 1 shows a diff; every
// later call is byte-identical yet WAS written. `same-nonzero` is that reading, and it is why the
// counter exists apart from `same-zero`.
FIXTURE_HLE(s_fixture_rewrite) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    auto* p = (uint8_t*)a0;
    if (!p) return 0x80A10003ull;
    p[0] = 0xAB;
    p[3] = 0xCD;
    return 0;
}

FIXTURE_HLE(s_fixture_nullarg) {   // a0 == 0
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0;
}
FIXTURE_HLE(s_fixture_handlearg) { // a0 is a small handle, not an address
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0;
}
FIXTURE_HLE(s_fixture_badptr) {    // a0 points at unmapped memory
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    return 0;
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
static hle_fn volatile table[9] = {
    &prosper::s_user_getevent,
    &prosper::s_fixture_wide,
    &prosper::s_fixture_ok,
    &prosper::s_fixture_writes12,
    &prosper::s_fixture_nowrite,
    &prosper::s_fixture_rewrite,
    &prosper::s_fixture_nullarg,
    &prosper::s_fixture_handlearg,
    &prosper::s_fixture_badptr,
};
static volatile uint64_t sink;
// Deliberately NOT cleared between calls -- see s_fixture_rewrite.
static uint8_t sticky[16];

int main() {
    // Runs until the window closes and hle_calls kills it. A fixture that exited first would end
    // the window early (`window=SHORT exited=1`) and the test would be asserting on a truncated run.
    for (;;) {
        // Zeroed per iteration, the way a guest that declares its out-struct on the stack per call
        // does. That matters for --out-bytes: the diff is between the bytes at entry and at return,
        // so a guest REUSING a buffer that already holds what the handler writes produces
        // `same-nonzero` rather than `changed` -- a real ambiguity of the method, kept out of this
        // fixture deliberately so the assertions below mean one thing.
        // 16 bytes and zeroed, so a sample WIDER than the 8-byte event struct still sees a
        // known state: a window overlapping unrelated stack would report `same-nonzero` for
        // reasons that have nothing to do with the handler.
        uint8_t event[16] = {0};
        uint8_t status[16] = {0};
        uint8_t untouched[16] = {0};

        sink = table[0]((uint64_t)event, 0, 0, 0, 0, 0);        // LOGIN-eligible; writes once
        sink = table[1](0, 0, 0, 0, 0, 0);
        sink = table[2](0, 0, 0, 0, 0, 0);
        sink = table[3]((uint64_t)status, 0, 0, 0, 0, 0);       // writes 12 bytes, st[6] = 1
        sink = table[4]((uint64_t)untouched, 0, 0, 0, 0, 0);    // writes nothing
        sink = table[5]((uint64_t)sticky, 0, 0, 0, 0, 0);       // rewrites the same bytes
        sink = table[6](0, 0, 0, 0, 0, 0);                      // a0 null
        sink = table[7](0x2a, 0, 0, 0, 0, 0);                   // a0 a small handle
        sink = table[8](0xdeadbeef000ull, 0, 0, 0, 0, 0);       // a0 unmapped (~15 TB, never mapped)
        prosper::k_usleep();
    }
}
