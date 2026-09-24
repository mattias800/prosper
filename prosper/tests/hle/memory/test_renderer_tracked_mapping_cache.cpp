#include "gpu/execute/gpu_execute.hpp"
#include "hle/memory/renderer_tracked_mapping.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "shared/live/buffer_source_gate.hpp"
#include "shared/live/guest_source_read.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

extern "C" int prosper_reserved_range_state(uint64_t addr);

using namespace prosper;

namespace {
struct TestState {
    int failures = 0;
    ProsperRendererTrackedMappingResult fake_tracked_result =
        ProsperRendererTrackedMappingResult::Untracked;
    int fake_reserved_result = 0;
    int fake_tracked_calls = 0;
    int fake_reserved_calls = 0;
};

TestState& state() {
    static TestState value;
    return value;
}

ProsperRendererTrackedMappingResult fake_probe_tracked(uint64_t) {
    ++state().fake_tracked_calls;
    return state().fake_tracked_result;
}

int fake_query_reserved(uint64_t) {
    ++state().fake_reserved_calls;
    return state().fake_reserved_result;
}

bool tracked(uint64_t address) {
    return prosper_renderer_guest_address_tracked(address) !=
        ProsperRendererTrackedMappingResult::Untracked;
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++state().failures;
    }
}

std::byte* map_untracked_page(size_t size) {
#ifdef _WIN32
    return static_cast<std::byte*>(
        VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    const auto result = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return result == MAP_FAILED ? nullptr : static_cast<std::byte*>(result);
#endif
}

void unmap_untracked_page(std::byte* page, size_t size) {
#ifdef _WIN32
    (void)size;
    VirtualFree(page, 0, MEM_RELEASE);
#else
    munmap(page, size);
#endif
}
}

int main() {
    register_builtin_hle();
    const auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    const auto flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
    const auto protect = Hle::lookup(nid_hash("sceKernelMprotect"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    if (!reserve || !flexible || !protect || !unmap) {
        std::puts("FAIL: memory HLE entry points are registered");
        return 1;
    }

    constexpr uint64_t page = 0x4000;
    TestState& test = state();

    test.fake_tracked_calls = 0;
    test.fake_reserved_calls = 0;
    auto gate = frontend::classify_buffer_source(
        true, 0x2000, true, fake_probe_tracked, fake_query_reserved);
    check(!gate.unavailable && !test.fake_tracked_calls && !test.fake_reserved_calls,
          "host-owned sources bypass both guest mapping probes");

    gate = frontend::classify_buffer_source(
        false, 0x800, true, fake_probe_tracked, fake_query_reserved);
    check(gate.unavailable && !test.fake_tracked_calls && !test.fake_reserved_calls,
          "low guest addresses remain unavailable without probing");

    test.fake_tracked_result = ProsperRendererTrackedMappingResult::CachedTracked;
    gate = frontend::classify_buffer_source(
        false, 0x2000, true, fake_probe_tracked, fake_query_reserved);
    check(!gate.unavailable && gate.tracked_cache_hit && !gate.tracked_cache_fill &&
              !gate.tracked_untracked_miss && !gate.reserved_state_query &&
              test.fake_tracked_calls == 1 && !test.fake_reserved_calls,
          "a warm tracked hit skips the numeric fallback and is counted exactly");

    test.fake_tracked_result = ProsperRendererTrackedMappingResult::AuthoritativeTracked;
    gate = frontend::classify_buffer_source(
        false, 0x2000, true, fake_probe_tracked, fake_query_reserved);
    check(!gate.unavailable && !gate.tracked_cache_hit && gate.tracked_cache_fill &&
              !gate.tracked_untracked_miss && !gate.reserved_state_query &&
              test.fake_tracked_calls == 2 && !test.fake_reserved_calls,
          "a cold tracked fill is distinct from a warm hit");

    test.fake_tracked_result = ProsperRendererTrackedMappingResult::Untracked;
    test.fake_reserved_result = 1;
    gate = frontend::classify_buffer_source(
        false, 0x2000, true, fake_probe_tracked, fake_query_reserved);
    check(!gate.unavailable && gate.tracked_untracked_miss && gate.reserved_state_query &&
              test.fake_tracked_calls == 3 && test.fake_reserved_calls == 1,
          "an untracked cache miss retains an accepting numeric fallback");

    test.fake_reserved_result = 0;
    gate = frontend::classify_buffer_source(
        false, 0x2000, false, fake_probe_tracked, fake_query_reserved);
    check(gate.unavailable && !gate.tracked_cache_hit && !gate.tracked_cache_fill &&
              !gate.tracked_untracked_miss && gate.reserved_state_query &&
              test.fake_tracked_calls == 3 && test.fake_reserved_calls == 2,
          "the disabled control bypasses the cache and preserves the numeric decision");

    frontend::BufferSourceGateCounters counters;
    counters.record(frontend::BufferSourceGateResult{.tracked_cache_hit = true});
    counters.record(frontend::BufferSourceGateResult{
        .tracked_cache_fill = true, .tracked_untracked_miss = true,
        .reserved_state_query = true});
    frontend::BufferSourceGateCounters totals;
    totals.add(counters);
    check(totals.tracked_cache_hits == 1 && totals.tracked_cache_fills == 1 &&
              totals.tracked_untracked_misses == 1 && totals.reserved_state_queries == 1,
          "gate telemetry records and aggregates each outcome exactly");

    std::byte* host_page = map_untracked_page(page);
    check(host_page != nullptr, "create an OS-readable untracked mapping");
    if (host_page) {
        const uint64_t address = reinterpret_cast<uint64_t>(host_page);
        check(gpu::guest_readable(address, 1), "warm the ordinary readability cache");
        check(prosper_reserved_range_state(address) == 0,
              "OS-readable memory remains absent from the guest tracker");
        check(!tracked(address),
              "readability cache does not authorize renderer tracked membership");
        check(prosper_renderer_guest_mapped_readable_prefix(address, page) == 0,
              "untracked host memory cannot become a renderer guest source");
    }
    if (host_page) unmap_untracked_page(host_page, page);

#ifndef _WIN32
    // mmap only guarantees host-page alignment. Give the HLE a complete guest
    // page so its 16 KiB normalization cannot extend outside the test mapping.
    constexpr size_t adoption_span = 3 * page;
    std::byte* adoption_mapping = map_untracked_page(adoption_span);
    check(adoption_mapping != nullptr, "create an aligned adoption test mapping");
    if (adoption_mapping) {
        const uint64_t raw = reinterpret_cast<uint64_t>(adoption_mapping);
        const uint64_t address = (raw + page - 1) & ~(page - 1);
        check(!tracked(address),
              "warm a negative lookup before protection-created tracking");
        check(protect(address, page, 0x1, 0, 0, 0) == 0,
              "protect adopts an otherwise untracked host mapping");
        check(prosper_renderer_guest_address_tracked(address) ==
                  ProsperRendererTrackedMappingResult::AuthoritativeTracked &&
                  prosper_renderer_guest_address_tracked(address) ==
                  ProsperRendererTrackedMappingResult::CachedTracked &&
                  prosper_reserved_range_state(address) != 0,
              "protection-created tracking produces one cold fill then a warm hit");
        check(unmap(address, page, 0, 0, 0, 0) == 0,
              "remove the protection-created tracked mapping");
        check(!tracked(address),
              "removing the adopted mapping invalidates its positive cache entry");

        const size_t prefix = address - raw;
        const size_t suffix = raw + adoption_span - (address + page);
        if (prefix) munmap(reinterpret_cast<void*>(raw), prefix);
        if (suffix) munmap(reinterpret_cast<void*>(address + page), suffix);
    }
#endif

    uint64_t reservation = 0;
    check(reserve(reinterpret_cast<uint64_t>(&reservation), page, 0, page, 0, 0) == 0 &&
              reservation,
          "create an uncommitted guest reservation");
    if (reservation) {
        check(prosper_reserved_range_state(reservation) == 1 &&
                  prosper_renderer_guest_address_tracked(reservation) ==
                      ProsperRendererTrackedMappingResult::AuthoritativeTracked &&
                  prosper_renderer_guest_address_tracked(reservation) ==
                      ProsperRendererTrackedMappingResult::CachedTracked,
              "uncommitted reservations produce one cold fill then a warm hit");
        check(unmap(reservation, page, 0, 0, 0, 0) == 0,
              "remove the uncommitted reservation");
        check(!tracked(reservation),
              "unmapping a warm reservation invalidates its positive cache entry");
    }

    uint64_t base = 0;
    constexpr uint64_t span = 4 * page;
#ifdef _WIN32
    const bool base_reserved =
        reserve(reinterpret_cast<uint64_t>(&base), span, 0, page, 0, 0) == 0 && base;
    check(base_reserved, "reserve a placeholder-backed multi-page range");
    uint64_t committed_base = base;
    const bool base_mapped = base_reserved &&
        flexible(reinterpret_cast<uint64_t>(&committed_base), span, 0x2, 0, 0, 0) == 0 &&
        committed_base == base;
#else
    const bool base_mapped =
        flexible(reinterpret_cast<uint64_t>(&base), span, 0x2, 0, 0, 0) == 0 && base;
#endif
    check(base_mapped, "create a multi-page tracked readable mapping");
    if (base_mapped) {
        check(prosper_renderer_guest_mapped_readable_prefix(base, span) == span,
              "committed guest mapping covers the requested source");
        check(prosper_renderer_guest_address_tracked(base) ==
                  ProsperRendererTrackedMappingResult::AuthoritativeTracked &&
                  prosper_renderer_guest_address_tracked(base + page) ==
                  ProsperRendererTrackedMappingResult::CachedTracked &&
                  prosper_renderer_guest_address_tracked(base + span - 1) ==
                  ProsperRendererTrackedMappingResult::CachedTracked,
              "one authoritative lookup warms the exact tracked interval");
        check(!tracked(base + span),
              "the exact interval end is excluded");

        uint64_t unmap_result = UINT64_MAX;
        std::jthread mutation([&unmap_result, unmap, base] {
            unmap_result = unmap(base + page, page, 0, 0, 0, 0);
        });
        mutation.join();
        check(unmap_result == 0, "partially unmap the middle page on another thread");
        check(tracked(base) && !tracked(base + page) && tracked(base + 2 * page),
              "first post-mutation query sees the hole and preserves both neighbors");
        check(prosper_renderer_guest_mapped_readable_prefix(base + page - 0x44c,
                                                            page) == 0x44c,
              "readable prefix stops at a split mapping hole");

        uint64_t hole = base + page;
#ifdef _WIN32
        // Partial flexible unmap retains a Windows private reservation. The ordinary hint path
        // recommits that page in place; the exact-address assertion below remains the test gate.
        constexpr uint64_t remap_flags = 0;
#else
        constexpr uint64_t remap_flags = 0x10; // SCE_KERNEL_MAP_FIXED
#endif
        check(flexible(reinterpret_cast<uint64_t>(&hole), page, 0x2,
                       remap_flags, 0, 0) == 0 && hole == base + page,
              "remap the exact address-reuse hole");
        check(prosper_renderer_guest_address_tracked(hole) ==
                  ProsperRendererTrackedMappingResult::AuthoritativeTracked,
              "address reuse invalidates the miss and admits the replacement mapping");
        check(prosper_renderer_guest_mapped_readable_prefix(base + page - 0x44c,
                                                            page) == page,
              "adjacent committed records restore the complete readable range");

        check(unmap(base, span, 0, 0, 0, 0) == 0,
              "remove the remapped three-page range");
        check(!tracked(base) && !tracked(base + page) && !tracked(base + 2 * page),
              "full unmap invalidates every cached subrange");
    }
#ifdef _WIN32
    else if (base) {
        unmap(base, span, 0, 0, 0, 0);
    }
#endif

    uint64_t unreadable = 0;
    check(flexible(reinterpret_cast<uint64_t>(&unreadable), page, 0, 0, 0, 0) == 0 &&
              unreadable,
          "create a tracked no-access mapping");
    if (unreadable) {
        check(tracked(unreadable) &&
                  prosper_reserved_range_state(unreadable) != 0,
              "unreadable tracked mappings retain the numeric gate decision");
        check(prosper_renderer_guest_mapped_readable_prefix(unreadable, page) == 0,
              "no-access guest mapping is not a readable source");
        check(unmap(unreadable, page, 0, 0, 0, 0) == 0,
              "remove the no-access mapping");
    }

    uint64_t split = 0;
    check(reserve(reinterpret_cast<uint64_t>(&split), 2 * page, 0, page, 0, 0) == 0 &&
              split, "reserve a committed-source/empty-tail pair");
    if (split) {
        uint64_t first = split;
        const bool first_mapped =
            flexible(reinterpret_cast<uint64_t>(&first), page, 0x2, 0, 0, 0) == 0 &&
            first == split;
        check(first_mapped && prosper_reserved_range_state(split + page) == 1,
              "first page is committed and adjacent page remains reserved");
        if (first_mapped) {
            check(protect(split + page, page, 0x1, 0, 0, 0) == 0 &&
                      prosper_reserved_range_state(split + page) == 1,
                  "read-protected reservation still has no committed source bytes");
            constexpr size_t available = 0x44c;
            constexpr size_t requested = 0x1000;
            const uint64_t source = split + page - available;
            std::memset(reinterpret_cast<void*>(source), 0x5a, available);
            std::vector<uint8_t> copied(requested, 0);
            const size_t count = frontend::copy_guest_source(
                copied.data(), source, requested, gpu::guest_readable);
            check(count == available &&
                      std::all_of(copied.begin(), copied.begin() + available,
                                  [](uint8_t byte) { return byte == 0x5a; }) &&
                      std::all_of(copied.begin() + available, copied.end(),
                                  [](uint8_t byte) { return byte == 0; }) &&
                      prosper_reserved_range_state(split + page) == 1,
                  "overdeclared copy preserves its zero tail without committing reservation");
            size_t compared = 0;
            check(frontend::equal_guest_source_prefix(
                      copied.data(), source, requested, compared, gpu::guest_readable) &&
                      compared == available && prosper_reserved_range_state(split + page) == 1,
                  "validation compares only the mapped source prefix");
            copied[0] ^= 1;
            check(!frontend::equal_guest_source_prefix(
                      copied.data(), source, requested, compared, gpu::guest_readable),
                  "mutated readable bytes fail exact validation");

            uint64_t second = split + page;
            const bool second_mapped =
                flexible(reinterpret_cast<uint64_t>(&second), page, 0x2, 0, 0, 0) == 0 &&
                second == split + page;
            check(second_mapped,
                  "commit adjacent page without changing the source address");
            if (second_mapped) {
                std::memset(reinterpret_cast<void*>(second), 0x33, requested - available);
                std::fill(copied.begin(), copied.end(), 0);
                check(frontend::copy_guest_source(
                          copied.data(), source, requested, gpu::guest_readable) == requested &&
                          std::all_of(copied.begin(), copied.begin() + available,
                                      [](uint8_t byte) { return byte == 0x5a; }) &&
                          std::all_of(copied.begin() + available, copied.end(),
                                      [](uint8_t byte) { return byte == 0x33; }),
                      "newly committed neighbor becomes readable without reusing old zeros");
            }
        }
        check(unmap(split, 2 * page, 0, 0, 0, 0) == 0,
              "remove committed-source/empty-tail pair");
    }

    if (test.failures) {
        std::printf("FAIL: %d check(s)\n", test.failures);
        return 1;
    }
    std::puts("PASS");
    return 0;
}
