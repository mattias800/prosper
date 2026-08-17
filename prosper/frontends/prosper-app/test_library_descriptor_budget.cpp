// test_library_descriptor_budget — the library view's descriptor-pool sizing and the exhaustion rule
// that keeps a failed allocation from ever being written to (#1649). Pure arithmetic and a fake
// allocator, so this runs in the default core build with no Vulkan, no ImGui and no window.
//
// The interesting assertion is the LAST group: acquire_texture_set() must not merely return a null
// handle once the budget is spent, it must not CALL the allocator at all. ImGui's backend passes the set
// vkAllocateDescriptorSets() produced into vkUpdateDescriptorSets() without checking it, so a guard that
// only inspects the return value inspects it after the undefined behaviour. The fake allocator below
// records whether it ran, which is the only way to tell those two guards apart.
#include "library_descriptor_budget.hpp"

#include <cstdio>
#include <limits>

using prosper::frontend::DescriptorBudget;
using prosper::frontend::acquire_texture_set;
using prosper::frontend::kLibraryBackgroundSets;
using prosper::frontend::kLibraryImGuiReservedSets;
using prosper::frontend::kLibraryMaxPoolSets;
using prosper::frontend::kLibraryMinPoolSets;
using prosper::frontend::library_descriptor_pool_sets;
using prosper::frontend::library_texture_budget;
using prosper::frontend::release_texture_set;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

// Stands in for a VkDescriptorSet. A plain int keeps the header Vulkan-free and lets the test count
// allocations, which a real handle could not.
using FakeSet = int;
static constexpr FakeSet kNone = 0;

struct FakeAllocator {
    int calls = 0;
    int next = 100;
    bool fail = false;      // simulate a driver-side failure that is NOT pool exhaustion
    FakeSet operator()() {
        ++calls;
        if (fail) return kNone;
        return next++;
    }
};

int main() {
    std::printf("== test_prosper_app_library_descriptor_budget ==\n");

    // --- pool sizing follows the library, not a fixed guess ---------------------------------------
    // The defect: a constant 256 sets shared by the covers and the backgrounds, so a library of about
    // 250 titles exhausted the pool. The size must move with the title count.
    CHECK(library_descriptor_pool_sets(1000) == 1000 + kLibraryBackgroundSets + kLibraryImGuiReservedSets,
          "a 1000-title library gets a set per cover plus the background and ImGui allowance");
    CHECK(library_descriptor_pool_sets(250) > 256,
          "250 titles now ask for MORE than the old fixed 256 (the exact case in #1649)");
    CHECK(library_descriptor_pool_sets(300) > library_descriptor_pool_sets(299),
          "one more title asks for one more set");
    // The allowance is what the background layer can hold at once (cache + insert + retired + slack); a
    // pool sized for covers alone would starve it on the very first focus change.
    CHECK(library_descriptor_pool_sets(40) - 40 == kLibraryBackgroundSets + kLibraryImGuiReservedSets,
          "the overhead above the cover count is exactly the background plus ImGui allowance");

    // --- clamped at both ends ---------------------------------------------------------------------
    CHECK(library_descriptor_pool_sets(0) == kLibraryMinPoolSets,
          "an empty library still gets the floor, so the font atlas and backgrounds have room");
    CHECK(library_descriptor_pool_sets(1) == kLibraryMinPoolSets,
          "a one-title library is still floored rather than given a 10-set pool");
    CHECK(library_descriptor_pool_sets(kLibraryMaxPoolSets + 5000) == kLibraryMaxPoolSets,
          "an absurd library is capped rather than asking the driver for an absurd pool");
    CHECK(library_descriptor_pool_sets(std::numeric_limits<size_t>::max()) == kLibraryMaxPoolSets,
          "a saturating count clamps to the ceiling instead of wrapping to a tiny pool");
    CHECK(kLibraryMinPoolSets > kLibraryBackgroundSets + kLibraryImGuiReservedSets,
          "the floor leaves room for at least one cover after the fixed overhead");

    // --- the budget reserves ImGui's own set -------------------------------------------------------
    // prosper never allocates the font atlas set, so it must be held back rather than counted; handing
    // the whole pool out would put the LAST cover exactly where the atlas lives.
    CHECK(library_texture_budget(64) == 64 - kLibraryImGuiReservedSets,
          "the budget is the pool minus ImGui's own reserved set");
    CHECK(library_texture_budget(0) == 0, "a zero-set pool yields no budget rather than underflowing");
    CHECK(library_texture_budget(kLibraryImGuiReservedSets) == 0,
          "a pool with only ImGui's set leaves nothing for covers");

    // --- accounting -------------------------------------------------------------------------------
    {
        DescriptorBudget b;
        b.reset(3);
        CHECK(b.capacity() == 3 && b.used() == 0 && b.available() == 3, "reset starts empty");
        CHECK(b.acquire() && b.acquire() && b.acquire(), "the first three acquisitions succeed");
        CHECK(b.used() == 3 && b.available() == 0, "the budget is now spent");
        CHECK(!b.acquire(), "the fourth acquisition is refused");
        CHECK(b.used() == 3, "a refused acquisition does not consume a slot");
        b.release();
        CHECK(b.used() == 2 && b.available() == 1, "a release returns a slot");
        CHECK(b.acquire(), "the returned slot can be taken again");
        b.release(); b.release(); b.release();
        CHECK(b.used() == 0, "releasing everything empties the budget");
        b.release();
        CHECK(b.used() == 0, "an unbalanced release clamps at zero instead of wrapping to 4 billion");
        b.reset(1);
        CHECK(b.capacity() == 1 && b.used() == 0, "reset re-arms with the new capacity");
    }
    {
        DescriptorBudget b;   // never reset: capacity 0
        CHECK(!b.acquire(), "a default-constructed budget refuses everything");
    }

    // --- the guard runs BEFORE the allocator, not after it -----------------------------------------
    // This is the correctness half of #1649. With the guard removed, every CHECK in this block that
    // counts `alloc.calls` fails, because the allocator is reached on an exhausted pool — which is
    // exactly the path that hands VK_NULL_HANDLE to vkUpdateDescriptorSets.
    {
        DescriptorBudget b;
        b.reset(2);
        FakeAllocator alloc;
        const FakeSet a = acquire_texture_set(b, kNone, [&] { return alloc(); });
        const FakeSet c = acquire_texture_set(b, kNone, [&] { return alloc(); });
        CHECK(a != kNone && c != kNone && a != c, "two allocations inside the budget both succeed");
        CHECK(alloc.calls == 2, "the allocator ran once per successful acquisition");

        const FakeSet over = acquire_texture_set(b, kNone, [&] { return alloc(); });
        CHECK(over == kNone, "an allocation past the budget yields a null handle");
        CHECK(alloc.calls == 2,
              "the allocator was NOT called past the budget (a return-value check would have run it)");
        CHECK(b.used() == 2, "the refused allocation did not consume a slot");

        // Freeing one must make room again, and must make room only once.
        FakeSet freed = a;
        int removed = 0;
        release_texture_set(b, freed, kNone, [&](FakeSet) { ++removed; });
        CHECK(removed == 1 && freed == kNone, "release hands the set to the remover and clears it");
        CHECK(b.used() == 1, "release returns the slot");
        release_texture_set(b, freed, kNone, [&](FakeSet) { ++removed; });
        CHECK(removed == 1 && b.used() == 1,
              "releasing an already-cleared handle is a no-op, so the budget cannot drift");

        const FakeSet again = acquire_texture_set(b, kNone, [&] { return alloc(); });
        CHECK(again != kNone && alloc.calls == 3, "the freed slot can be allocated again");
    }
    {
        // An allocator that fails for a reason other than exhaustion must give its slot straight back,
        // or a run of transient failures would permanently shrink the usable pool.
        DescriptorBudget b;
        b.reset(2);
        FakeAllocator alloc;
        alloc.fail = true;
        const FakeSet s = acquire_texture_set(b, kNone, [&] { return alloc(); });
        CHECK(s == kNone && alloc.calls == 1, "a failing allocator is still called while there is room");
        CHECK(b.used() == 0, "its slot is returned rather than leaked");
        alloc.fail = false;
        CHECK(acquire_texture_set(b, kNone, [&] { return alloc(); }) != kNone,
              "the budget is undamaged by the failure");
    }

    // --- a pool sized for a library can hold that library's covers ---------------------------------
    // Ties the two halves together: the sizing must satisfy the accounting, which is the property that
    // actually failed at 250 titles.
    for (size_t titles : {size_t(0), size_t(1), size_t(30), size_t(250), size_t(255), size_t(1000)}) {
        DescriptorBudget b;
        b.reset(library_texture_budget(library_descriptor_pool_sets(titles)));
        size_t got = 0;
        for (size_t i = 0; i < titles; i++) if (b.acquire()) ++got;
        const bool all = got == titles;
        const bool backgrounds = b.available() >= kLibraryBackgroundSets;
        char msg[160];
        std::snprintf(msg, sizeof msg,
                      "%zu titles: every cover fits (%zu) and %u background sets remain", titles, got,
                      b.available());
        CHECK(all && backgrounds, msg);
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
