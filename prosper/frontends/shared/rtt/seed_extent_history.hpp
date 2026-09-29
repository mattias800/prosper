#pragma once

#include <cstdint>

namespace prosper::frontend {

struct SeedExtent {
    uint32_t width = 0, height = 0, format = 0;
    bool operator==(const SeedExtent&) const = default;
};

// Observation only, owned by one CPU RTT entry. Opposite misses distinguish A->B->A extent churn
// from the repeated A->B request of a masked pass that never replaces A (#3907).
class SeedExtentHistory {
public:
    bool observe(uint64_t address, SeedExtent stored, SeedExtent requested) {
        if (!address || !stored.width || !stored.height || !requested.width ||
            !requested.height || !stored.format || stored.format != requested.format ||
            stored == requested) {
            reset();
            return false;
        }
        const bool reversed = address == address_ && stored == requested_ && requested == stored_;
        address_ = address;
        stored_ = stored;
        requested_ = requested;
        return reversed;
    }

    void reset() { address_ = 0; }

private:
    uint64_t address_ = 0;
    SeedExtent stored_, requested_;
};

} // namespace prosper::frontend
