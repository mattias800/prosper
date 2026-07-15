#include "host/guest_memory_map.hpp"

#include <cstdio>

using prosper::host::GuestReadableRangeCache;

int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* name) {
        std::printf("  [%s] %s\n", condition ? "ok" : "FAIL", name);
        if (!condition) ++failures;
    };

    std::puts("== test_guest_memory_map ==");
    GuestReadableRangeCache cache;
    cache.sync_generation(10);
    cache.insert(0x3000, 0x4000);
    cache.insert(0x1000, 0x2000);
    check(cache.contains(0x1000, 0x1800), "out-of-order disjoint insert remains searchable");
    check(cache.contains(0x3000, 0x4000), "last inserted range remains searchable");
    check(!cache.contains(0x1800, 0x3800), "query spanning an unmapped gap misses");

    cache.insert(0x2000, 0x3000);
    check(cache.range_count() == 1, "adjacent ranges merge into one interval");
    check(cache.contains(0x1000, 0x4000), "merged interval covers its complete extent");

    cache.erase(0x1800, 0x3800);
    check(cache.contains(0x1000, 0x1800), "range removal preserves the lower fragment");
    check(!cache.contains(0x1800, 0x3800), "range removal drops the overlapped span");
    check(cache.contains(0x3800, 0x4000), "range removal preserves the upper fragment");
    cache.insert(0x1800, 0x3800);

    cache.sync_generation(10);
    check(cache.contains(0x1800, 0x3800), "unchanged mapping generation preserves ranges");
    cache.sync_generation(11);
    check(cache.range_count() == 0, "mapping generation change clears cached ranges");
    check(!cache.contains(0x1000, 0x1800), "cleared generation cannot return a stale hit");

    const uint64_t generation = prosper::host::guest_mapping_generation();
    prosper::host::notify_guest_mapping_added(0x10000, 0x4000, true);
    check(prosper::host::guest_mapping_generation() != generation,
          "mapping notification advances the process generation");
    prosper::host::GuestReadableRange mapping{};
    check(prosper::host::guest_readable_mapping_containing(0x11000, 0x12000, mapping) &&
              mapping.begin == 0x10000 && mapping.end == 0x14000,
          "readable HLE mapping exposes its stable containing extent");
    prosper::host::notify_guest_mapping_added(0x11000, 0x1000, false);
    check(!prosper::host::guest_readable_mapping_containing(0x11000, 0x12000, mapping),
          "unreadable overlay is ineligible for cross-submit reuse");
    prosper::host::notify_guest_mapping_removed(0x10000, 0x4000);
    check(!prosper::host::guest_readable_mapping_containing(0x10000, 0x11000, mapping),
          "removed HLE mapping cannot retain a positive readability result");
    prosper::host::notify_guest_mapping_added(0x20000, 0x1000, true);
    prosper::host::notify_guest_mapping_removed(UINT64_MAX - 1, 4);
    check(!prosper::host::guest_readable_mapping_containing(0x20000, 0x21000, mapping),
          "malformed mapping change conservatively clears readable extents");

    std::puts(failures ? "== FAIL ==" : "== PASS ==");
    return failures ? 1 : 0;
}
