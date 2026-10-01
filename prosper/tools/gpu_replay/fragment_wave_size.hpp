#pragma once

#include "gpu/diagnostics/diagnostic_selectors.hpp"

namespace prosper::tools {

// Realized draws do not yet retain SPI_PS_IN_CONTROL.PS_W32_EN. Keep the historical
// requested default distinct from an explicit diagnostic override, never a captured ABI claim.
// A scoped enum cannot silently convert a numeric width to recompile_fragment's Boolean mode.
enum class ReplayFragmentWaveSize : uint32_t { Wave32 = 32, Wave64 = 64 };

struct ReplayFragmentWaveSelection {
    ReplayFragmentWaveSize size = ReplayFragmentWaveSize::Wave64;
    bool explicit_override = false;

    bool wave32() const { return size == ReplayFragmentWaveSize::Wave32; }
};

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
