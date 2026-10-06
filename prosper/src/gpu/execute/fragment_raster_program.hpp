#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <string>
#include <vector>

namespace prosper::gpu {
// Cold original-code proof for the first live/helper recipe. These are required input routes,
// not an initialized lane, an incoming mask, or permission to convert host helper observations.
struct FragmentRasterPositionInput {
    uint32_t reg = 0, collector_word = 0;
    bool operator==(const FragmentRasterPositionInput&) const = default;
};
struct FragmentRasterProgram {
    std::vector<FragmentRasterPositionInput> positions;
    std::string rejection;
};
// Proves that wave placement outside a genuine four-lane quad cannot affect this original
// program's exports. Saved incoming masks are kept only in the mask domain, WQM establishes
// genuine within-quad consumers, and original live EXEC is restored before the final VM export.
// Wider mask exposure, branching, LDS and inter-quad consumers need their own general recipes.
FragmentRasterProgram fragment_raster_program(const std::vector<Rdna2Inst>&,
                                              uint32_t original_words, PixelSystemInputMapping,
                                              uint32_t genuine_user_presence);
} // namespace prosper::gpu
