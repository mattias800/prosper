#pragma once

#include <cstdint>

// The renderer needs only tracked/untracked membership, while the signal-facing numeric query
// also classifies sparse and AMM states. Keep the cache result explicit so diagnostics and tests
// distinguish a warm lookup from the authoritative lookup that populated it.
enum class ProsperRendererTrackedMappingResult : int {
    Untracked = 0,
    AuthoritativeTracked = 1,
    CachedTracked = 2,
};

extern "C" ProsperRendererTrackedMappingResult
prosper_renderer_guest_address_tracked(uint64_t addr);
