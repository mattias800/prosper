// test_kernel_system_stubs — sceKernelGetOperationMode, the coredump-handler pair and the PRT
// aperture pair are registered and honour their contracts, instead of falling through to the
// "unimplemented, returning 0" logger.
//
// WHY. All five used to be unimplemented stubs. Returning 0 is the right result for a valid call, but
// the stub is wrong in ways a valid call never shows: GetOperationMode left its two out-parameters
// untouched (a guest that does not pre-zero them branches on uninitialised stack), SetPrtAperture
// accepted any slot index, and GetPrtAperture read back nothing. Assassin's Creed Black Flag Resynced
// calls GetOperationMode, SetPrtAperture and RegisterCoredumpHandler during boot.
//
// WHAT EACH ARM KILLS:
//   M1  GetOperationMode is unregistered again (stub)         -> the sentinel-overwrite arm
//   M2  GetOperationMode writes 8 bytes instead of 4          -> the guard-byte arms
//   M3  SetPrtAperture accepts any index                       -> the out-of-range arms
//   M4  GetPrtAperture does not read back what was set         -> the round-trip arm
//   M5  the coredump pair is unregistered again                -> the registered arms
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Written as a literal so a mutated sce_errno.hpp cannot make the arms agree with itself.
static constexpr uint64_t kEncodedEINVAL = 0x80020016ull;

static uint64_t call3(HleFn f, uint64_t a, uint64_t b, uint64_t c) { return f(a, b, c, 0, 0, 0); }

int main() {
    printf("== test_kernel_system_stubs ==\n");
    register_builtin_hle();

    HleFn get_mode = Hle::lookup(nid_hash("sceKernelGetOperationMode"));
    HleFn co_reg = Hle::lookup(nid_hash("sceCoredumpRegisterCoredumpHandler"));
    HleFn co_unreg = Hle::lookup(nid_hash("sceCoredumpUnregisterCoredumpHandler"));
    HleFn prt_set = Hle::lookup(nid_hash("sceKernelSetPrtAperture"));
    HleFn prt_get = Hle::lookup(nid_hash("sceKernelGetPrtAperture"));
    CHECK(get_mode != nullptr, "sceKernelGetOperationMode is registered (M1)");
    CHECK(co_reg && co_unreg, "the coredump handler pair is registered (M5)");
    CHECK(prt_set && prt_get, "the PRT aperture pair is registered");
    if (!get_mode || !co_reg || !co_unreg || !prt_set || !prt_get) return 1;

    // --- GetOperationMode: two 32-bit out-parameters, both written, nothing else touched ----------
    {
        alignas(8) uint8_t region[24];
        memset(region, 0xA5, sizeof region);
        const uint64_t rc = call3(get_mode, (uint64_t)(uintptr_t)(region + 4),
                                  (uint64_t)(uintptr_t)(region + 12), 0);
        int32_t mode = -1, submode = -1;
        memcpy(&mode, region + 4, sizeof mode);
        memcpy(&submode, region + 12, sizeof submode);
        CHECK(rc == 0, "GetOperationMode returns 0");
        CHECK(mode == 0 && submode == 0, "both out-parameters are written (not left as sentinel) (M1)");
        bool guard_ok = true;
        for (int i = 0; i < 4; ++i) guard_ok &= region[i] == 0xA5;
        for (int i = 8; i < 12; ++i) guard_ok &= region[i] == 0xA5;
        for (int i = 16; i < 24; ++i) guard_ok &= region[i] == 0xA5;
        CHECK(guard_ok, "only the two 4-byte slots are written; neighbouring bytes are untouched (M2)");
        CHECK(call3(get_mode, 0, 0, 0) == 0, "null out-pointers are tolerated");
    }

    // --- Coredump handler pair: a registration and an unregistration both succeed -----------------
    CHECK(call3(co_reg, 0x1234, 0x10000, 0x5678) == 0, "RegisterCoredumpHandler succeeds");
    CHECK(co_unreg(0, 0, 0, 0, 0, 0) == 0, "UnregisterCoredumpHandler succeeds");
    CHECK(co_unreg(0, 0, 0, 0, 0, 0) == 0, "a second Unregister is also accepted");

    // --- PRT apertures: three slots, EINVAL outside them, a set value reads back ------------------
    for (int index = 0; index < 3; ++index) {
        char text[96];
        snprintf(text, sizeof text, "SetPrtAperture accepts slot %d", index);
        CHECK(call3(prt_set, (uint64_t)index, 0x7000000000ull + (uint64_t)index * 0x1000, 0x20000) == 0,
              text);
    }
    CHECK(call3(prt_set, 3, 0x1000, 0x1000) == kEncodedEINVAL, "slot 3 is out of range: EINVAL (M3)");
    CHECK(call3(prt_set, (uint64_t)(uint32_t)-1, 0x1000, 0x1000) == kEncodedEINVAL,
          "slot -1 is out of range: EINVAL (M3)");

    for (int index = 0; index < 3; ++index) {
        uint64_t address = 0, length = 0;
        const uint64_t rc = call3(prt_get, (uint64_t)index, (uint64_t)(uintptr_t)&address,
                                  (uint64_t)(uintptr_t)&length);
        char text[96];
        snprintf(text, sizeof text, "GetPrtAperture reads back slot %d (M4)", index);
        CHECK(rc == 0 && address == 0x7000000000ull + (uint64_t)index * 0x1000 && length == 0x20000, text);
    }
    // The index is an `int`: only the low 32 bits of the register count, so 0x100000001 names slot 1.
    {
        uint64_t address = 0, length = 0;
        CHECK(call3(prt_set, 0x100000001ull, 0x7100000000ull, 0x30000) == 0,
              "the slot index is read as a 32-bit int (high register bits ignored)");
        CHECK(call3(prt_get, 1, (uint64_t)(uintptr_t)&address, (uint64_t)(uintptr_t)&length) == 0 &&
                  address == 0x7100000000ull && length == 0x30000,
              "...and it landed in slot 1");
    }
    uint64_t a = 0, l = 0;
    CHECK(call3(prt_get, 3, (uint64_t)(uintptr_t)&a, (uint64_t)(uintptr_t)&l) == kEncodedEINVAL,
          "GetPrtAperture slot 3: EINVAL (M3)");
    CHECK(call3(prt_get, 0, 0, (uint64_t)(uintptr_t)&l) == kEncodedEINVAL,
          "GetPrtAperture with a null out-pointer: EINVAL, no host fault");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
