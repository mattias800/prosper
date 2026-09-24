#pragma once

#include "hle/memory/renderer_tracked_mapping.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace prosper::frontend {

// The guest's descriptor length is not a mapping length. Check the exact tracked mapping
// prefix before any host probe can materialize an uncommitted reservation as zero pages.
// On Windows, the probe also prepares committed sparse direct-memory pages for host access.
template <typename HostReadable>
size_t guest_source_readable_prefix(uint64_t address, size_t bytes,
                                    HostReadable&& host_readable) {
    if (!bytes || address > std::numeric_limits<uint64_t>::max() - bytes) return 0;
    const size_t mapped = static_cast<size_t>(
        prosper_renderer_guest_mapped_readable_prefix(address, bytes));
#ifdef _WIN32
    if (mapped <= std::numeric_limits<uint32_t>::max() &&
        (mapped == 0 || host_readable(address, static_cast<uint32_t>(mapped))))
        return mapped;
    size_t ready = 0;
    while (ready < mapped) {
        const uint64_t current = address + ready;
        const size_t chunk = std::min(mapped - ready,
                                      size_t{0x4000} - static_cast<size_t>(current & 0x3fff));
        if (!host_readable(current, static_cast<uint32_t>(chunk))) break;
        ready += chunk;
    }
    return ready;
#else
    (void)host_readable;
    return mapped;
#endif
}

// The caller owns a zero-initialized destination. A short read leaves its tail untouched.
template <typename HostReadable>
size_t copy_guest_source(uint8_t* destination, uint64_t address, size_t bytes,
                         HostReadable&& host_readable) {
    const size_t readable = guest_source_readable_prefix(
        address, bytes, std::forward<HostReadable>(host_readable));
    if (readable) std::memcpy(destination, reinterpret_cast<const void*>(address), readable);
    return readable;
}

// Return the number of whole chunks that matched, as the renderer's validation counters do.
// A short readable prefix is reported to the caller; cache admission checks that count.
template <typename HostReadable>
bool equal_guest_source_prefix(const uint8_t* expected, uint64_t address, size_t bytes,
                               size_t& compared, HostReadable&& host_readable) {
    const size_t readable = guest_source_readable_prefix(
        address, bytes, std::forward<HostReadable>(host_readable));
    compared = 0;
    while (compared < readable) {
        const uint64_t current = address + compared;
        const size_t chunk = std::min(readable - compared,
                                      size_t{0x10000} - static_cast<size_t>(current & 0xffff));
        if (!expected || std::memcmp(expected + compared,
                                     reinterpret_cast<const void*>(current), chunk)) return false;
        compared += chunk;
    }
    return true;
}

} // namespace prosper::frontend
