#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <string>

namespace prosper::gpu {
struct FragmentQuadParameterLayout {
    FragmentInterpolationLayout layout;
    std::string rejection;
};
// Cold original-program/profile transformation for the logical-quad producer. Ordinary P1/P2
// interpolation ALSO needs genuine P0/P10/P20, not a coefficient guessed from its final smooth
// value. This requests those three flat planes from the existing exact triangle GS and preserves
// all ordinary varying/system modes. The device owner supplies its actual input-component limit.
// A layout alone is not a producing VS/GS association, all-path vertex definition, parameter-LDS
// allocation, guest launch or live scheduling certificate; those remain draw-bound obligations.
FragmentQuadParameterLayout
fragment_quad_parameter_layout(const FragmentInterpolationLayout& original,
                               const PixelSystemInputMapping&,
                               uint32_t max_fragment_input_components);
} // namespace prosper::gpu
