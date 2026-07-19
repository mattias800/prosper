// test_guest_tls_align — the x86-64 Variant II static-TLS layout must round each module's offset to
// its REAL PT_TLS p_align (#143), not a hardcoded 16. Otherwise a module with p_align > 16 gets its
// block at the wrong distance below the thread pointer and every initial-exec %fs:-N access resolves
// off. Drives guest_tls_set_templates and reads back the per-module offsets. Linux-only.
#include "../src/hle/dispatch.hpp"
#include "../src/loader/tls_layout.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

int main() {
    printf("== test_guest_tls_align ==\n");
#if defined(__linux__)
    unsetenv("PROSPER_NO_GUEST_FS");
#endif

    // descs[0] is the reserved slot (module id 0 = "no TLS"); real modules start at index 1.
    // Module 1: memsz 0x30, p_align 64 (aligned TLS vars — the case the old 16-hardcode broke).
    // Module 2: memsz 0x10, p_align 16.
    std::vector<TlsModuleDesc> descs = {
        { 0, 0, 0, 0 },
        { /*init_va*/0, /*filesz*/0, /*memsz*/0x30, /*align*/64 },
        { 0, 0, /*memsz*/0x10, /*align*/16 },
    };
    guest_tls_set_templates(descs.data(), descs.size());
    const StaticTlsLayout layout = make_static_tls_layout(descs.data(), descs.size());
#if defined(__linux__)
    CHECK(guest_tls_enabled(), "guest initial-exec TLS is enabled by default on Linux");
#endif

    // Variant II: off[i] = round_up(off[i-1] + memsz[i], align[i]); block start = TP - off[i].
    uint64_t exp1 = align_up(0 + 0x30, 64);          // = 0x40 (rounded to 64, NOT 0x30)
    uint64_t exp2 = align_up(exp1 + 0x10, 16);       // = 0x50
    CHECK(guest_tls_module_below(1) == exp1, "module 1 offset rounds to its real p_align 64 (0x40, was 0x30)");
    CHECK(guest_tls_module_below(2) == exp2, "module 2 offset accumulates from module 1 (0x50)");
    CHECK(layout.module_below[1] == guest_tls_module_below(1) &&
          layout.module_below[2] == guest_tls_module_below(2),
          "shared linker layout exactly matches the runtime allocator offsets");
    CHECK(guest_tls_module_below(1) != 0x30, "the old 16-byte rounding (0x30) is NOT used for p_align 64");
    // TP must be aligned to the max module align (64 here), so the total is >= exp2 rounded to 64.
    CHECK(guest_tls_total_below() >= exp2 && (guest_tls_total_below() % 64) == 0,
          "total below TP is a multiple of the max module align (64)");
    CHECK(layout.total_below == guest_tls_total_below(),
          "shared linker layout exactly matches the allocator's total size");

    // Regression: the common p_align <= 16 case is byte-identical to the old per-size rounding
    // (round_up of a 16-multiple running total by 16 == summing round_up(memsz,16)).
    std::vector<TlsModuleDesc> descs16 = {
        { 0, 0, 0, 0 },
        { 0, 0, 0x08, 16 },
        { 0, 0, 0x14, 16 },
    };
    guest_tls_set_templates(descs16.data(), descs16.size());
    CHECK(guest_tls_module_below(1) == align_up(0x08, 16),                      "p_align16 module 1 == round_up(0x08,16)=0x10");
    CHECK(guest_tls_module_below(2) == align_up(0x10 + 0x14, 16),               "p_align16 module 2 == 0x30 (accumulated)");

    // A zero align field falls back to 16 (defensive — a module with p_align 0/1 is treated as 16).
    std::vector<TlsModuleDesc> descs0 = { { 0, 0, 0, 0 }, { 0, 0, 0x10, 0 } };
    guest_tls_set_templates(descs0.data(), descs0.size());
    CHECK(guest_tls_module_below(1) == 0x10, "zero p_align falls back to 16-byte rounding");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
