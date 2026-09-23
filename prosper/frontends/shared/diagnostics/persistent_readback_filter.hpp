// Address admission for a bounded PROSPER_DUMP_PERSISTENT readback.
#pragma once

#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
