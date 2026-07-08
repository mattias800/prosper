// test_printf — guards the variadic libc formatters (hle_libc.cpp, issue #122). The old handlers
// forwarded only 3 integer registers, so %f (XMM), stack-spilled args, and %s past the 4th argument
// produced garbage or crashed. The fix makes them real C variadic functions. This looks up the
// registered function pointer and calls it through its TRUE variadic signature (same SysV ABI the
// guest uses via the tail-jump stub), exercising the register + XMM + stack capture end to end.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using SnprintfFn = int (*)(char*, size_t, const char*, ...);
using SprintfFn  = int (*)(char*, const char*, ...);

int main() {
    printf("== test_printf ==\n");
    register_builtin_hle();

    auto snf = (SnprintfFn)(void*)Hle::lookup(nid_hash("snprintf"));
    auto spf = (SprintfFn)(void*)Hle::lookup(nid_hash("sprintf"));
    auto snfs = (SnprintfFn)(void*)Hle::lookup(nid_hash("snprintf_s"));
    CHECK(snf && spf && snfs, "printf-family fns registered");
    if (!(snf && spf && snfs)) { printf("== FAIL ==\n"); return 1; }

    char buf[256];

    // Mixed args that hit GP registers, XMM (the %f that was totally dropped), AND the stack:
    // GP varargs: "val"(rcx), 42(r8), 0xabcd(r9), 5(stack), 6(stack) — 5 GP args spill past 3 regs.
    // FP varargs: 3.25(xmm0). This is the exact shape the old register-only forward mangled.
    int r = snf(buf, sizeof buf, "%s=%d f=%.2f x=%x a5=%d a6=%lld",
                "val", 42, 3.25, 0xabcd, 5, (long long)6);
    CHECK(r > 0, "snprintf returned a positive length");
    CHECK(strcmp(buf, "val=42 f=3.25 x=abcd a5=5 a6=6") == 0, "snprintf formats %s/%d/%f/%x + stack args");

    // Float-heavy: multiple XMM args must all land correctly.
    snf(buf, sizeof buf, "%.1f %.1f %.1f %.1f", 1.5, 2.5, 3.5, 4.5);
    CHECK(strcmp(buf, "1.5 2.5 3.5 4.5") == 0, "snprintf handles multiple %f (XMM) args");

    // %s past the 4th argument (was garbage under the register-only forward).
    snf(buf, sizeof buf, "%s %s %s %s %s", "a", "b", "c", "d", "e");
    CHECK(strcmp(buf, "a b c d e") == 0, "snprintf handles %s past the 4th argument");

    // snprintf honors the size bound (truncates, returns the would-be length).
    char small[8];
    int wr = snf(small, sizeof small, "%s", "0123456789");
    CHECK(wr == 10 && strcmp(small, "0123456") == 0, "snprintf truncates to the buffer bound");

    // snprintf_s maps to the same variadic path.
    snfs(buf, sizeof buf, "%d-%.1f", 7, 2.0);
    CHECK(strcmp(buf, "7-2.0") == 0, "snprintf_s formats via the variadic path");

    // sprintf (unbounded) into a wide buffer.
    spf(buf, "%s#%d/%.3f", "k", 9, 0.125);
    CHECK(strcmp(buf, "k#9/0.125") == 0, "sprintf formats mixed args");

    if (fails) { printf("== FAIL: %d check(s) ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
