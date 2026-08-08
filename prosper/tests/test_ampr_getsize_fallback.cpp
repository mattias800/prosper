// sceAmprCommandBufferGetSize (tZDDEo2tE5k) must never answer for one command buffer with the
// recorded capacity of a DIFFERENT one.
//
// The guest contract this protects (The Forgotten City, PPSA03026, UE4/IoStore; the same append
// loop DOLL and Pathless use): the IoStore batch builder only appends its next command once
//
//     sceAmprCommandBufferGetSize(cb) - sceAmprCommandBufferGetCurrentOffset(cb) > 0xff
//
// and it polls that in a tight loop with no bound. So a GetSize answer that is merely *small* is
// not a degraded answer, it is a permanent hang: PPSA03026's IoService thread span at 100% CPU
// forever, the GameThread waited on I/O that could never be enqueued, and the title never
// composited a frame past its very first flip.
//
// The cause was the legacy `g_apr_last_cb_size` global. It is recorded together with the cb it
// belongs to (`g_apr_last_cb`) but was previously served as the fallback for *any* cb whose own
// capacity we failed to recognise. PPSA03026 constructs its IoService cb as (a1=0, a2=<ptr>,
// a5=<ptr>) — a shape no capacity can be read from — while one unrelated construction elsewhere in
// the same boot records a1=0x81 (129). 129 - 0 never exceeds 255.
//
// Both arms below are load-bearing:
//   - the victim cb must get the roomy unknown-capacity default (this FAILS before the fix, which
//     returned 129), and
//   - the cb the global was actually recorded against must still get 129 (this FAILS if the fix is
//     over-applied and drops the legacy path altogether).
#include "../src/hle/dispatch.hpp"
#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_ampr_getsize_fallback ==\n");
    register_builtin_hle();

    HleFn construct  = Hle::lookup("8aI7R7WaOlc");
    HleFn get_size   = Hle::lookup("tZDDEo2tE5k");
    HleFn get_offset = Hle::lookup("GnxKOHEawhk");
    CHECK(construct && get_size && get_offset, "the three AMPR command-buffer NIDs are registered");
    if (!construct || !get_size || !get_offset) return 1;

    // A construction whose capacity IS legible, and which therefore also records the legacy
    // (g_apr_last_cb = a3, g_apr_last_cb_size = a1) pair. Shape taken from a live PPSA03026 boot.
    constexpr uint64_t kRecordedCb   = 0x7f0000001000ull;  // a0: the constructed cb
    constexpr uint64_t kRecordedSize = 0x81ull;            // a1: 129 bytes — below the guest's 0xff gate
    constexpr uint64_t kLegacyCb     = 0x7f0000002000ull;  // a3: what g_apr_last_cb is set to
    construct(kRecordedCb, kRecordedSize, 0x1, kLegacyCb, 0, 0);

    CHECK(get_size(kRecordedCb, 0, 0, 0, 0, 0) == kRecordedSize,
          "a cb with a legible capacity still reports exactly that capacity");
    // Unguarded since #1970. This arm used to be `#if defined(__linux__) || defined(__APPLE__)`
    // because `g_apr_last_cb`/`g_apr_last_cb_size` lived inside the POSIX arm of the platform
    // split, so Windows had no legacy path to assert. That guard was not a property of the test —
    // it was the defect, written down: the two arms answered one guest ABI question differently,
    // and Windows discarded a capacity prosper had already recorded. The pair is now defined above
    // the split and recorded by both `k_ampr_init`s, so this holds everywhere. If it ever needs a
    // platform guard again, the divergence has come back rather than the test having got stricter.
    CHECK(get_size(kLegacyCb, 0, 0, 0, 0, 0) == kRecordedSize,
          "the legacy global answers for the cb it was recorded against, on every platform");

    // PPSA03026's IoService cb: a1 is zero and a2/a5 are guest pointers far above any plausible
    // command-buffer capacity, so no capacity can be recovered from the constructor arguments.
    constexpr uint64_t kVictimCb = 0x2200df50b0ull;
    construct(kVictimCb, 0, 0x2000ad0000ull, 0, 0x3001a60000ull, 0x3001a70720ull);

    const uint64_t size   = get_size(kVictimCb, 0, 0, 0, 0, 0);
    const uint64_t offset = get_offset(kVictimCb, 0, 0, 0, 0, 0);
    std::printf("  victim cb: GetSize=0x%llx GetCurrentOffset=0x%llx free=0x%llx\n",
                (unsigned long long)size, (unsigned long long)offset,
                (unsigned long long)(size - offset));

    CHECK(size != kRecordedSize,
          "an unknown-capacity cb is NOT served another cb's recorded size");
    CHECK(offset == 0, "a freshly constructed cb reports a zero command offset");
    CHECK(size - offset > 0xff,
          "an unknown-capacity cb reports enough free space to clear the guest's 0xff append gate");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
