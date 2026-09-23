// Address admission for a bounded PROSPER_DUMP_PERSISTENT readback.
#pragma once

#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace prosper::frontend {

enum class PersistentReadbackFilterState : uint8_t {
    All,
    Selected,
    Malformed,
    TooMany,
};

struct PersistentReadbackFilter {
    static constexpr size_t kMaxAddresses = 8;

    PersistentReadbackFilterState state = PersistentReadbackFilterState::All;
    std::vector<uint64_t> addresses;

    bool allows(uint64_t address) const {
        return state == PersistentReadbackFilterState::All ||
               (state == PersistentReadbackFilterState::Selected &&
                std::find(addresses.begin(), addresses.end(), address) != addresses.end());
    }
};

enum class PersistentReadbackCharge : uint8_t {
    Admitted,
    InvalidSize,
    SizeOverflow,
    OverBudget,
};

// Charge before issuing a selected readback. A failed readback keeps its charge, so even repeated
// failures cannot exceed the per-callback transfer budget. The broad legacy census does not use it.
struct PersistentReadbackBudget {
    static constexpr uint64_t kMaxBytes = 256ull << 20;
    uint64_t charged_bytes = 0;

    PersistentReadbackCharge admit(uint32_t width, uint32_t height, uint32_t bytes_per_pixel,
                                   uint64_t& admitted_bytes) {
        admitted_bytes = 0;
        if (!width || !height || !bytes_per_pixel) return PersistentReadbackCharge::InvalidSize;
        const uint64_t pixels = static_cast<uint64_t>(width) * height;
        if (pixels > std::numeric_limits<uint64_t>::max() / bytes_per_pixel)
            return PersistentReadbackCharge::SizeOverflow;
        const uint64_t bytes = pixels * bytes_per_pixel;
        if (charged_bytes > kMaxBytes || bytes > kMaxBytes - charged_bytes)
            return PersistentReadbackCharge::OverBudget;
        charged_bytes += bytes;
        admitted_bytes = bytes;
        return PersistentReadbackCharge::Admitted;
    }
};

// Null preserves the broad legacy census. A supplied malformed or oversized filter admits nothing:
// it must not quietly fall back to reading every retained target after the user asked for a bound.
inline PersistentReadbackFilter parse_persistent_readback_filter(const char* spec) {
    if (!spec) return {};
    PersistentReadbackFilter filter;
    if (!gpu::parse_hex_watch_list(spec, filter.addresses)) {
        filter.state = PersistentReadbackFilterState::Malformed;
        return filter;
    }
    if (filter.addresses.size() > PersistentReadbackFilter::kMaxAddresses) {
        filter.addresses.clear();
        filter.state = PersistentReadbackFilterState::TooMany;
        return filter;
    }
    filter.state = PersistentReadbackFilterState::Selected;
    return filter;
}

} // namespace prosper::frontend
