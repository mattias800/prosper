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
#include "../src/gpu/shader_resources.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>

using prosper::gpu::compute_address_range_contains;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// The windowed form exists because the address form asked a different question from the change
// detection that consumed it. These arms pin the difference.
//
// Worked example this comes from: a program binding [window + 128, +160) was reported as touching
// NOTHING, because the address form asks only whether a resource contains the window's FIRST byte.
// The tree watch detects changes anywhere in the window, so the two halves disagreed, and the
// disagreement was published as "this program writes bytes it does not bind" -- an out-of-bounds
// write that never existed. An instrument whose halves ask different questions manufactures exactly
// the finding it was built to detect.
static prosper::gpu::ShaderResource watch_scalar(uint32_t binding, uint32_t fetch_pc, uint64_t base,
                                   uint32_t size) {
    prosper::gpu::ShaderResource r;
    r.binding = binding;
    r.fetch_pc = fetch_pc;
    r.gpu_addr = base;
    r.size = size;
    return r;
}

static void test_window_hits() {
    using prosper::gpu::compute_address_ranges_overlap;
    using prosper::gpu::compute_address_window_hits;
    using prosper::gpu::compute_address_watch_hits;

    constexpr uint64_t kWindowBase = 0x209cc76000ull;
    constexpr uint64_t kWindowBytes = 2063ull * 4ull;

    prosper::gpu::ShaderResourceTable table;
    table.resources.push_back(watch_scalar(7, 43, kWindowBase + 128, 160));   // interior
    table.resources.push_back(watch_scalar(8, 44, kWindowBase - 0x1000, 64)); // wholly below
    table.resources.push_back(watch_scalar(9, 45, kWindowBase + 8200, 64));   // straddles the end (window is 8252 bytes)

    CHECK(compute_address_watch_hits(table, kWindowBase).empty(),
          "the ADDRESS form finds nothing: no resource contains the window's first byte");

    const auto window = compute_address_window_hits(table, kWindowBase, kWindowBytes);
    CHECK(window.size() == 2u,
          "the WINDOW form finds the interior resource and the one straddling the end");
    bool saw_interior = false, saw_below = false;
    for (const auto& hit : window) {
        if (hit.binding == 7u && hit.fetch_pc == 43u) saw_interior = true;
        if (hit.binding == 8u) saw_below = true;
    }
    CHECK(saw_interior, "the interior resource is named with its binding and fetch pc");
    CHECK(!saw_below,
          "a resource wholly below the window is still not a toucher (no false positive traded in)");

    prosper::gpu::ShaderResourceTable adjacent;
    adjacent.resources.push_back(watch_scalar(1, 1, kWindowBase - 64, 64));
    CHECK(compute_address_window_hits(adjacent, kWindowBase, kWindowBytes).empty(),
          "a resource ending exactly where the window begins does not overlap it");

    prosper::gpu::ShaderResourceTable touching;
    touching.resources.push_back(watch_scalar(1, 1, kWindowBase - 64, 65));
    CHECK(compute_address_window_hits(touching, kWindowBase, kWindowBytes).size() == 1u,
          "one byte of overlap is overlap");

    CHECK(!compute_address_ranges_overlap(kWindowBase, 0, kWindowBase, 16),
          "a zero-size resource never overlaps");
    CHECK(!compute_address_ranges_overlap(kWindowBase, 16, kWindowBase, 0),
          "a zero-size window never overlaps");
    CHECK(!compute_address_ranges_overlap(UINT64_MAX - 8, 16, 0x1000, 64),
          "a range near UINT64_MAX does not wrap into a low window");
    CHECK(!compute_address_ranges_overlap(0x1000, 64, UINT64_MAX - 8, 16),
          "and does not wrap with the arguments the other way round");
}

int main() {
    test_window_hits();
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

    // --- TRAVERSAL, which the range helper above cannot cover ------------------------------------
    //
    // The call site decides WHICH ranges to test, and that decision is where the real defect was: a
    // runtime-selected array parent carries the DESCRIPTOR TABLE address in `gpu_addr` and the widest
    // element size in `size`. That pair is not a backing range. Testing it unconditionally
    // fabricates a match against the descriptor table, and double-reports whenever the parent and
    // entry 0 share an address — which the descriptor-array fixture makes them do.
    {
        using prosper::gpu::ComputeAddressWatchHit;
        using prosper::gpu::compute_address_watch_hits;
        constexpr uint64_t kTableBase = 0x1000;   // descriptor table: a DECOY, never a backing range
        constexpr uint64_t kEntry0 = 0x1000;      // deliberately equal to the parent's address
        constexpr uint64_t kEntry1 = 0x9000;

        prosper::gpu::ShaderResourceTable table;
        prosper::gpu::ShaderResource array_resource;
        array_resource.binding = 7;
        array_resource.fetch_pc = 42;
        array_resource.gpu_addr = kTableBase;
        array_resource.size = 64;                 // widest element, not a backing extent
        array_resource.table_index_count = 2;
        array_resource.table_entries.push_back({{}, kEntry0, 32, 4, nullptr, 0});
        array_resource.table_entries.push_back({{}, kEntry1, 32, 4, nullptr, 0});
        table.resources.push_back(array_resource);

        prosper::gpu::ShaderResource scalar_resource;
        scalar_resource.binding = 3;
        scalar_resource.fetch_pc = 91;
        scalar_resource.gpu_addr = 0x20000;
        scalar_resource.size = 8252;
        table.resources.push_back(scalar_resource);

        // An address inside entry 0, which is also inside the parent's decoy range. A traversal that
        // tests the parent as well reports this TWICE.
        const auto shared = compute_address_watch_hits(table, kEntry0 + 8);
        CHECK(shared.size() == 1,
              "an address in entry 0 is reported exactly once, not also as an array-parent hit "
              "(testing the parent range as well double-reports it)");
        CHECK(shared.size() == 1 && shared[0].from_array_entry && shared[0].entry_index == 0,
              "and it is attributed to the realized entry, not the parent");

        // An address inside the parent's decoy range but in NO entry. The descriptor table is not
        // backing storage, so nothing should match.
        const auto decoy = compute_address_watch_hits(table, kTableBase + 48);
        CHECK(decoy.empty(),
              "an address inside the array PARENT's range but in no entry does not match "
              "(the parent's gpu_addr is the descriptor table, not a backing range)");

        const auto second = compute_address_watch_hits(table, kEntry1 + 31);
        CHECK(second.size() == 1 && second[0].from_array_entry && second[0].entry_index == 1,
              "a realized entry away from the parent address is found");

        const auto scalar = compute_address_watch_hits(table, 0x20000 + 4);
        CHECK(scalar.size() == 1 && !scalar[0].from_array_entry && scalar[0].binding == 3,
              "an ordinary scalar resource is still matched through its own range");

        CHECK(compute_address_watch_hits(table, 0x777777).empty(),
              "and an address in neither matches nothing");
    }

    printf(fails ? "== FAILURES: %d ==\n" : "== all passed (%d failures) ==\n", fails);
    return fails ? 1 : 0;
}
