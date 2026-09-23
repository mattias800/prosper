#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::frontend {

struct PixelSourceCandidate {
    std::shared_ptr<const std::vector<uint8_t>> pixels;
    uint64_t source_submit = 0;
};

// A numeric current-submit value is not pixel provenance. In the CPU compositor path, certify an
// output only when it is the very same immutable allocation as a completed current-pass candidate.
// Equal bytes at a different address, or conflicting labels on an aliased candidate, stay unknown.
inline uint64_t selected_source_submit(
    const std::shared_ptr<const std::vector<uint8_t>>& selected,
    const std::array<PixelSourceCandidate, 3>& candidates) {
    if (!selected) return 0;
    uint64_t source = 0;
    bool matched = false;
    for (const PixelSourceCandidate& candidate : candidates) {
        if (!candidate.pixels || selected != candidate.pixels) continue;
        if (matched && source != candidate.source_submit) return 0;
        source = candidate.source_submit;
        matched = true;
    }
    return matched ? source : 0;
}

} // namespace prosper::frontend
