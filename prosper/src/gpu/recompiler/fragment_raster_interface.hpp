#pragma once
#include <cstdint>
#include <span>

namespace prosper::gpu {
struct FragmentRasterOutputInterface {
    bool available = false, position = false;
    uint64_t components = 0, location_end = 0;
    uint32_t float4_locations = 0, output_vertices = 0;
};
// Cold selected-module interface inspection, including built-ins. This is neither an all-path
// output-definition proof nor guest interpolation authority. Unknown interface types refuse;
// they are not silently omitted to fit a device budget.
FragmentRasterOutputInterface fragment_raster_output_interface(std::span<const uint32_t>,
                                                               uint32_t execution_model);
FragmentRasterOutputInterface fragment_raster_input_interface(std::span<const uint32_t>,
                                                              uint32_t execution_model);
} // namespace prosper::gpu
