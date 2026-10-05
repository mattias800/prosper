#include "host/memory/host_address_layout.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace prosper::host {

uint64_t confine_host_allocations_above_4gib() {
    uint64_t reserved = 0;
#ifdef _WIN32
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return 0;
    constexpr uint64_t kLo = 0x10000, kHi = 0x100000000ull, kGranule = 0x10000;
    // Two passes: another thread can allocate inside a free region between the query and the
    // reserve, which fails that whole reserve and leaves the rest of the region free.
    for (int pass = 0; pass < 2; ++pass) {
        uint64_t cur = kLo;
        while (cur < kHi) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cur)), &mbi,
                              sizeof mbi))
                break;
            const uint64_t base =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress));
            const uint64_t end = std::min<uint64_t>(base + mbi.RegionSize, kHi);
            if (end <= cur) break;
            if (mbi.State == MEM_FREE) {
                // A reservation starts on the 64 KiB allocation granularity.
                const uint64_t from = (std::max(base, kLo) + kGranule - 1) & ~(kGranule - 1);
                const uint64_t to = end & ~(kGranule - 1);
                if (from < to &&
                    VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(from)),
                                 static_cast<SIZE_T>(to - from), MEM_RESERVE, PAGE_NOACCESS))
                    reserved += to - from;
            }
            cur = end;
        }
    }
#endif
    return reserved;
}

GuestRangeOccupancy query_guest_range_occupancy(uint64_t lo, uint64_t hi) {
    GuestRangeOccupancy result;
#ifdef _WIN32
    constexpr int kMaxRegions = 1 << 20; // bounds the walk; a live range has a few thousand
    result.available = true;
    uint64_t cur = lo, last_allocation = UINT64_MAX;
    for (int i = 0; cur < hi && i < kMaxRegions; ++i) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(cur)), &mbi, sizeof mbi))
            return result; // incomplete: `complete` stays false
        const uint64_t base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.BaseAddress));
        const uint64_t end = std::min<uint64_t>(base + mbi.RegionSize, hi);
        if (end <= cur) return result;
        if (mbi.State == MEM_FREE) {
            result.largest_free_gap = std::max(result.largest_free_gap, end - std::max(base, lo));
        } else {
            const uint64_t allocation =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbi.AllocationBase));
            if (allocation != last_allocation) {
                if (result.occupants < GuestRangeOccupancy::kMaxShown)
                    result.shown[result.occupants] = {base, mbi.RegionSize, mbi.State, mbi.Type};
                ++result.occupants;
                last_allocation = allocation;
            }
        }
        cur = end;
    }
    result.complete = cur >= hi;
#else
    (void)lo;
    (void)hi;
#endif
    return result;
}

void* report_unplaceable(uint64_t len, uint64_t lo, uint64_t hi) {
    static std::atomic<bool> reported{false};
    if (reported.exchange(true)) return nullptr;
    const GuestRangeOccupancy o = query_guest_range_occupancy(lo, hi);
    if (!o.available) {
        fprintf(stderr,
                "[memhle] reserve FAILED: a %llu GiB guest reservation does not fit the guest "
                "range [0x%llx, 0x%llx). The guest gets ENOMEM (#4426).\n",
                (unsigned long long)(len >> 30), (unsigned long long)lo, (unsigned long long)hi);
        return nullptr;
    }
    fprintf(stderr,
            "[memhle] reserve FAILED: a %llu GiB guest reservation does not fit the guest range "
            "[0x%llx, 0x%llx) -- its largest free gap is %llu GiB, with %d allocation(s) standing "
            "in it%s. The guest gets ENOMEM; a UE4 title then hangs in its allocator bootstrap "
            "(#4426).\n",
            (unsigned long long)(len >> 30), (unsigned long long)lo, (unsigned long long)hi,
            (unsigned long long)(o.largest_free_gap >> 30), o.occupants,
            o.complete ? "" : " (walk incomplete: both figures are lower bounds)");
    const int shown = std::min(o.occupants, GuestRangeOccupancy::kMaxShown);
    for (int i = 0; i < shown; ++i)
        fprintf(stderr, "[memhle]   occupant base=0x%llx size=0x%llx state=0x%lx type=0x%lx\n",
                (unsigned long long)o.shown[i].base, (unsigned long long)o.shown[i].size,
                (unsigned long)o.shown[i].state, (unsigned long)o.shown[i].type);
    if (o.occupants > shown) fprintf(stderr, "[memhle]   ... and %d more\n", o.occupants - shown);
    return nullptr;
}

} // namespace prosper::host
