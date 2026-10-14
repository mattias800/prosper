#include "gpu/recompiler/rdna2_launch_vgprs.hpp"

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {

// Whether this backend binds a value for an ENABLED PS input field. Each answer names the shell
// code that binds it in recompile_fragment; anything else stays a placeholder.
bool pixel_field_supplied(const PixelSystemInputMapping& inputs, uint32_t field,
                          bool interpolation_geometry) {
    if (field <= 6) return interpolation_geometry;   // system_interpolation_component
    if (field >= 8 && field <= 11) return true;      // guest_pixel_position_component
    if (field == 12)                                 // fragment_front_face_word
        return inputs.front_face_encoding == FragmentFrontFaceEncoding::SignedFloat ||
               inputs.front_face_encoding == FragmentFrontFaceEncoding::AllBits;
    if (field == 13) return true;                    // fragment_ancillary_layer_bits
    // 7 LINE_STIPPLE_TEX, 14 SAMPLE_COVERAGE and 15 POS_FIXED_PT have no binding yet.
    return false;
}

}   // namespace

std::vector<LaunchVgpr> pixel_launch_vgprs(const PixelSystemInputMapping& inputs,
                                           bool interpolation_geometry) {
    std::vector<LaunchVgpr> out;
    out.reserve(32);   // the sixteen fields total 25 words
    uint32_t vgpr = 0;
    for (uint32_t field = 0; field < kPixelLaunchFields; ++field) {
        const uint32_t bit = 1u << field;
        if (!(inputs.addr & bit)) continue;
        const LaunchVgprState state = !(inputs.ena & bit) ? LaunchVgprState::Reserved
                                      : pixel_field_supplied(inputs, field, interpolation_geometry)
                                          ? LaunchVgprState::Supplied
                                          : LaunchVgprState::Unsupplied;
        for (uint32_t component = 0; component < kPixelLaunchFieldWidths[field]; ++component)
            out.push_back({vgpr + component, field, component, state});
        vgpr += kPixelLaunchFieldWidths[field];
    }
    return out;
}

std::vector<LaunchVgpr> compute_launch_vgprs(uint32_t tidig_comp_cnt) {
    // TIDIG_COMP_CNT is a two-bit field; 3 loads the same three components as 2.
    const uint32_t components = std::min(tidig_comp_cnt, 2u) + 1;
    std::vector<LaunchVgpr> out;
    out.reserve(components);
    for (uint32_t component = 0; component < components; ++component)
        out.push_back({component, component, 0, LaunchVgprState::Supplied});
    return out;
}

const char* pixel_launch_field_name(uint32_t field) {
    static constexpr const char* names[kPixelLaunchFields] = {
        "persp_sample",  "persp_center",  "persp_centroid",  "persp_pull_model",
        "linear_sample", "linear_center", "linear_centroid", "line_stipple_tex",
        "pos_x",         "pos_y",         "pos_z",           "pos_w",
        "front_face",    "ancillary",     "sample_coverage", "pos_fixed_pt"};
    return field < kPixelLaunchFields ? names[field] : "unknown";
}

const char* launch_vgpr_state_name(LaunchVgprState state) {
    switch (state) {
        case LaunchVgprState::Reserved: return "reserved";
        case LaunchVgprState::Unsupplied: return "unsupplied";
        case LaunchVgprState::Supplied: return "supplied";
    }
    return "unknown";
}

}   // namespace prosper::gpu
