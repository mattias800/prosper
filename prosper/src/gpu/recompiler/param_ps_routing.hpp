// param_ps_routing.hpp -- which fragment-input locations one vertex PARAM export feeds.
//
// The ordinary vertex compiler routes `EXP PARAM<source>` to logical PS inputs through
// SPI_PS_INPUT_CNTL_n.OFFSET (PixelInputMapping). Every vertex-side commit stage that publishes
// already-computed PARAM values (the owned-wave export commit, the merged-NGG raster commit) must
// publish them to exactly the same locations, so the rule lives here once.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>

namespace prosper::gpu {

// Call `publish(location)` for every fragment-input location PARAM `source` (0..31) feeds:
//   * with no mapping, the PARAM's own index;
//   * with a mapping whose slot `source` is not valid, the PARAM's own index when that input is
//     consumed (a default-routed attribute);
//   * every valid, consumed input whose OFFSET names `source` (OFFSET[4:0] for pass-through inputs).
// A location may be published twice when both rules name it; callers that store the same vec4 twice
// are unaffected.
template <typename Publish>
void for_each_param_ps_location(uint32_t source, const PixelInputMapping* pixel_inputs,
                                Publish&& publish) {
    if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source))) {
        if (!pixel_inputs || pixel_inputs->consumes(source)) publish(source);
    }
    if (!pixel_inputs) return;
    const uint32_t passthrough = pixel_inputs->effective_passthrough_mask();
    for (uint32_t input = 0; input < pixel_inputs->controls.size(); ++input) {
        if (!(pixel_inputs->valid_mask & (1u << input)) || !pixel_inputs->consumes(input)) continue;
        const uint32_t raw_offset = pixel_inputs->controls[input] & 0x3fu;
        const uint32_t offset = (passthrough & (1u << input)) ? raw_offset & 0x1fu : raw_offset;
        if (offset == source) publish(input);
    }
}

}   // namespace prosper::gpu
