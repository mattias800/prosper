#pragma once

#include "gpu/diagnostics/diagnostic_selectors.hpp"

namespace prosper::tools {

// Keep requested overrides, captured guest launch ABI and the historical unknown-width default
// separate. Neither a request nor a stored SPIR-V subgroup marker supplies capture provenance.
// A scoped enum cannot silently convert a numeric width to recompile_fragment's Boolean mode.
enum class ReplayFragmentWaveSize : uint32_t { Wave32 = 32, Wave64 = 64 };

struct ReplayFragmentWaveSelection {
    ReplayFragmentWaveSize size = ReplayFragmentWaveSize::Wave64;
    bool explicit_override = false;

    bool wave32() const { return size == ReplayFragmentWaveSize::Wave32; }
};

inline ReplayFragmentWaveSelection resolve_replay_fragment_wave_size(
    const ReplayFragmentWaveSelection& requested, bool captured_available, bool captured_wave32) {
    if (requested.explicit_override) return requested;
    if (!captured_available) return {};
    return {captured_wave32 ? ReplayFragmentWaveSize::Wave32 : ReplayFragmentWaveSize::Wave64,
            false};
}

inline bool parse_replay_fragment_wave_size(const char* text,
                                           ReplayFragmentWaveSelection& selection) {
    ReplayFragmentWaveSelection parsed;
    if (text) {
        uint64_t size = 0;
        if (!gpu::parse_diagnostic_uint64(text, size) || (size != 32 && size != 64))
            return false;
        parsed.size = size == 32 ? ReplayFragmentWaveSize::Wave32
                                : ReplayFragmentWaveSize::Wave64;
        parsed.explicit_override = true;
    }
    selection = parsed;
    return true;
}

} // namespace prosper::tools
