#pragma once

#include "gpu/rdna2_to_spirv.hpp"

#include <cstdint>

namespace prosper::gpu::replay_tool {

// Offline classification of the resolved SPI_PS_INPUT_CNTL linkage retained with a captured draw
// (capture v36+).
//
// Why this exists: a pixel-shader input either consumes a producer PARAM export or asks the
// interpolator to synthesize a CONSTANT instead. Nothing in the VS/FS instruction streams reveals
// which — a varying wired to `default` carries no producer data at all, so a stage that looks
// perfectly correct in isolation can still be fed a hardware constant. `PROSPER_INTERPLOG` reports
// the same linkage but only fires on the live executor path, so capsule analysis had no way to see
// it. Deciding whether a suspect varying is real producer output or a synthesized constant
// otherwise required a live boot.
//
// The predicates below deliberately mirror recompile_vertex_impl's export lowering (the
// effective-passthrough test, the OFFSET==0x20 constant form, and the DEFAULT_VAL decode) so this
// diagnostic can never describe a mapping the lowering does not actually perform.

enum class PixelInputKind {
    Unused,            // the control is not valid for this input slot
    Param,             // consumes producer PARAM `param`
    ConstantDefault,   // OFFSET=0x20: interpolator synthesizes a constant, no PARAM consumed
    ParamPassthrough,  // explicit parameter-cache pass-through of PARAM `param`
};

struct PixelInputLinkage {
    PixelInputKind kind = PixelInputKind::Unused;
    // Producer PARAM slot for Param / ParamPassthrough. Meaningless for the other kinds.
    uint32_t param = 0;
    // ConstantDefault only: GFX10 DEFAULT_VAL selects 0.0 or 1.0 independently for xyz and for w.
    bool default_xyz_one = false;
    bool default_w_one = false;
};

// Classify one pixel-shader input slot. `effective_passthrough` is the caller-hoisted result of
// mapping.effective_passthrough_mask(); it is passed in so a loop over all 32 slots does not
// recompute that O(32) scan per slot.
inline PixelInputLinkage pixel_input_linkage(const PixelInputMapping& mapping, uint32_t input,
                                             uint32_t effective_passthrough) {
    PixelInputLinkage linkage;
    if (input >= mapping.controls.size()) return linkage;
    if (!(mapping.valid_mask & (1u << input))) return linkage;

    const uint32_t control = mapping.controls[input];
    if (effective_passthrough & (1u << input)) {
        linkage.kind = PixelInputKind::ParamPassthrough;
        linkage.param = control & 0x1Fu;
        return linkage;
    }
    const uint32_t raw_offset = control & 0x3Fu;
    if (raw_offset == 0x20u) {
        linkage.kind = PixelInputKind::ConstantDefault;
        const uint32_t default_val = (control >> 8) & 0x3u;
        linkage.default_xyz_one = (default_val & 0x2u) != 0u;
        linkage.default_w_one = (default_val & 0x1u) != 0u;
        return linkage;
    }
    linkage.kind = PixelInputKind::Param;
    linkage.param = raw_offset;
    return linkage;
}

inline PixelInputLinkage pixel_input_linkage(const PixelInputMapping& mapping, uint32_t input) {
    return pixel_input_linkage(mapping, input, mapping.effective_passthrough_mask());
}

}  // namespace prosper::gpu::replay_tool
