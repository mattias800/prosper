// test_compute_address_watch — the address-watch matcher's edge cases.
//
// The diagnostic this backs answers "which programs touch this allocation", and a census is only
// worth anything if the matcher cannot miss. Its first version compared base equality over top-level
// resources, which silently misses an interior address, an overlapping subview, and every
// runtime-selected array element — and a census built on it was used to claim a writer set was
// closed at two. It was not; the matcher could not have seen a third.
//
// These are the cases where a range test misbehaves quietly rather than loudly: the exclusive end,
// an empty range, and arithmetic near the top of the address space, where a subtraction in the wrong
// order wraps and reports a match for an address nowhere near the allocation.
#include "../src/gpu/gpu_execute.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>

using prosper::gpu::compute_address_range_contains;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_compute_address_watch ==\n");
    constexpr uint64_t kBase = 0x20f848417cull;
    constexpr uint64_t kSize = 8252;

    CHECK(compute_address_range_contains(kBase, kSize, kBase),
          "the exact base is contained");
    CHECK(compute_address_range_contains(kBase, kSize, kBase + 4),
          "an interior address is contained (base equality would miss it)");
    CHECK(compute_address_range_contains(kBase, kSize, kBase + kSize - 1),
          "the last byte is contained");
    CHECK(!compute_address_range_contains(kBase, kSize, kBase + kSize),
          "one past the end is NOT contained (the range is half-open)");
    CHECK(!compute_address_range_contains(kBase, kSize, kBase - 1),
          "an address below the base is not contained");

    // A zero-size resource is a real shape here: prosper's proven-empty markers carry addr 0 size 0,
    // and a matcher that treats size 0 as "matches its base" would report every such marker.
    CHECK(!compute_address_range_contains(kBase, 0, kBase),
          "a zero-size range contains nothing, not even its own base");
    CHECK(!compute_address_range_contains(0, 0, 0),
          "and an all-zero resource does not match address zero");

    // The ordering guard. With the subtraction performed before the comparison, `wanted - base`
    // wraps for any address below the base and the predicate reports a spurious match far from the
    // allocation. Near UINT64_MAX that is the difference between a clean census and a fabricated hit.
    constexpr uint64_t kTop = std::numeric_limits<uint64_t>::max();
    CHECK(compute_address_range_contains(kTop - 3, 4, kTop),
          "a range ending at UINT64_MAX contains its final byte");
    CHECK(!compute_address_range_contains(kTop - 3, 4, 0),
          "and address zero does not match it");

    // THE ARM THAT ACTUALLY PINS THE ORDERING. The case above does not: with the `wanted >= base`
    // test removed, `0 - (kTop-3)` wraps to exactly 4, and `4 < 4` is false, so the broken predicate
    // returns the right answer by arithmetic luck. Verified by mutation -- deleting the guard left
    // every assertion here passing.
    //
    // For the wrap to land INSIDE the range, the range must extend past the top of the address
    // space, which a malformed descriptor can express. base = 2^64-2 with size 4 covers
    // {2^64-2, 2^64-1, 0, 1} only if the arithmetic is allowed to wrap; correctly, it contains just
    // the first two. Address 1 is the discriminator: `1 - (2^64-2)` wraps to 3, and 3 < 4.
    CHECK(!compute_address_range_contains(kTop - 1, 4, 1),
          "an address BELOW a range that overruns the address space does not match "
          "(deleting the `wanted >= base` ordering guard fails HERE, where the case above does not)");
    CHECK(compute_address_range_contains(kTop - 1, 4, kTop),
          "while an address genuinely inside that range still matches");
    CHECK(!compute_address_range_contains(kTop, 1, kTop - 1),
          "a one-byte range at the very top excludes the byte below it");
    CHECK(compute_address_range_contains(kTop, 1, kTop),
          "and contains itself");

    printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
