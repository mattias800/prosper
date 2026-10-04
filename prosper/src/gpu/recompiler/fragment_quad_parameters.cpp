#include "gpu/recompiler/fragment_quad_parameters.hpp"
#include <algorithm>
#include <bit>

namespace prosper::gpu {
FragmentQuadParameterLayout
fragment_quad_parameter_layout(const FragmentInterpolationLayout& original,
                               const PixelSystemInputMapping& system,
                               uint32_t max_fragment_input_components) {
    const auto refuse = [](const char* reason) {
        FragmentQuadParameterLayout result;
        result.layout.valid = false;
        result.rejection = reason;
        return result;
    };
    if (!original.valid ||
        ((original.smooth_mask | original.passthrough_mask | original.flat_mask) &
         ~original.attribute_mask))
        return refuse("fragment-quad-original-parameter-layout-unavailable");
    if (!max_fragment_input_components || (max_fragment_input_components & 3u))
        return refuse("fragment-quad-enabled-input-component-limit-unavailable");
    // Inputs at location=attribute remain in the interface. Do not collide with a high original
    // attribute or silently drop it to make extra parameter planes fit a convenient fixture.
    const uint32_t first = std::bit_width(original.attribute_mask);
    const uint32_t systems = (system.ena & system.addr) & 0x7fu;
    const uint64_t required =
        uint64_t(first) + 3u * std::popcount(original.attribute_mask) + std::popcount(systems);
    if (required * 4u > max_fragment_input_components)
        return refuse("fragment-quad-enabled-input-component-budget");
    FragmentQuadParameterLayout result;
    result.layout = original;
    for (auto& locations : result.layout.parameter_locations)
        locations.fill(FragmentInterpolationLayout::kUnusedLocation);
    result.layout.system_locations.fill(FragmentInterpolationLayout::kUnusedLocation);
    result.layout.requires_geometry = original.attribute_mask || systems;
    uint32_t location = first;
    for (uint32_t attribute = 0; attribute < 32; ++attribute)
        if (original.attribute_mask & (1u << attribute))
            for (uint32_t selector = 0; selector < 3; ++selector)
                result.layout.parameter_locations[attribute][selector] = location++;
    for (uint32_t field = 0; field < 7; ++field)
        if (systems & (1u << field)) result.layout.system_locations[field] = location++;
    return result;
}
}   // namespace prosper::gpu
