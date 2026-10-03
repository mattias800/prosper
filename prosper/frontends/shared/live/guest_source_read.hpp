#pragma once

#include "hle/memory/renderer_tracked_mapping.hpp"
#include "gpu/resources/buffer_source_read.hpp"

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
    return prosper::gpu::copy_buffer_source(destination, address, bytes,
        [&](uint64_t source, size_t requested) {
            return guest_source_readable_prefix(
                source, requested, std::forward<HostReadable>(host_readable));
        });
}
// Optional observations of the existing comparison, never an equality/admission authority.
// Extent is the sum of memcmp argument lengths, not physical reads inside its implementation.
enum class GuestSourceComparisonOutcome : uint8_t {
    ReadablePrefixEqual, ExpectedMissing, BytesDiffer
};
struct GuestSourceComparisonObservation {
    GuestSourceComparisonOutcome outcome = GuestSourceComparisonOutcome::ReadablePrefixEqual;
    size_t readable_prefix = 0;
    size_t memcmp_calls = 0;
    size_t memcmp_extent_bytes = 0;
};

// `compared` counts bytes in fully matching chunks, excluding the first differing chunk
// (at most 64 KiB) and any chunk rejected for a null expected pointer. It is not a count
// of all comparison reads. A short readable prefix may return true; cache admission must
// also check `compared` against its required prefix length.
template <typename HostReadable>
bool equal_guest_source_prefix(const uint8_t* expected, uint64_t address, size_t bytes,
                               size_t& compared, HostReadable&& host_readable,
                               GuestSourceComparisonObservation* observation = nullptr) {
    const size_t readable = guest_source_readable_prefix(
        address, bytes, std::forward<HostReadable>(host_readable));
    compared = 0;
    if (observation) {
        *observation = {};
        observation->readable_prefix = readable;
    }
    while (compared < readable) {
        const uint64_t current = address + compared;
        const size_t chunk = std::min(readable - compared,
                                      size_t{0x10000} - static_cast<size_t>(current & 0xffff));
        if (!expected) {
            if (observation) observation->outcome = GuestSourceComparisonOutcome::ExpectedMissing;
            return false;
        }
        if (observation) {
            ++observation->memcmp_calls;
            observation->memcmp_extent_bytes += chunk;
        }
        if (std::memcmp(expected + compared, reinterpret_cast<const void*>(current), chunk)) {
            if (observation) observation->outcome = GuestSourceComparisonOutcome::BytesDiffer;
            return false;
        }
        compared += chunk;
    }
    return true;
}

} // namespace prosper::frontend
