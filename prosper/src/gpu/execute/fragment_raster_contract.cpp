#include "gpu/execute/fragment_raster_contract.hpp"

namespace prosper::gpu {
const char* fragment_raster_workitem_gap(const RasterLaunchFacts& launch) {
    if (!launch.canonical()) return "fragment-draw-original-raster-facts-noncanonical";
    if (!launch.sc_shader_control_available)
        return "fragment-draw-original-sc-shader-control-unavailable";
    if (!launch.sc_mode_cntl_0_available || !launch.sc_mode_cntl_1_available)
        return "fragment-draw-original-sc-mode-unavailable";
    if (!launch.sc_aa_config_available) return "fragment-draw-original-sc-aa-config-unavailable";
    if (!launch.db_shader_control_available)
        return "fragment-draw-original-db-shader-control-unavailable";
    // AMD's gfx10/10.3 PAL register definitions identify the fields below. RDNA2 Table 6
    // calls initial EXEC the workitem-valid mask; it does not equate it to a host helper flag.
    // This recipe removes depth/sample/kill/NOOP transformations and pairs these predicates
    // with the actual backend raster contract before using covered nonhelpers as live workitems.
    if (launch.sc_shader_control & ~0x6fu)
        return "fragment-draw-original-sc-shader-control-bits-unimplemented";
    if (launch.sc_mode_cntl_0 & ((1u << 0) | (1u << 3)))
        return "fragment-draw-msaa-or-unlit-workitems-unimplemented";
    if (launch.sc_mode_cntl_1 & 0x09ffc000u)
        return "fragment-draw-post-raster-kill-sample-or-discard-unimplemented";
    if (launch.sc_aa_config & 0x0f700007u)
        return "fragment-draw-sample-coverage-selection-unimplemented";
    // LATE_Z=0. No shader kill, coverage/mask/depth export, hierarchy-fail/NOOP execution,
    // early depth, POPS/overlap, or pre-depth coverage. The genuine shader bypass bit must
    // disable alpha-to-mask too; an unobserved DB_ALPHA_TO_MASK state is not assumed disabled.
    constexpr uint32_t db_effects = 0x00f317f7u;
    if (launch.db_shader_control & db_effects)
        return "fragment-draw-depth-kill-noop-or-overlap-workitems-unimplemented";
    if (!(launch.db_shader_control & (1u << 11)))
        return "fragment-draw-alpha-to-mask-coverage-unproved";
    using C = RasterCoverageControl;
    const auto& coverage = launch.coverage;
    for (const auto control :
         {C::DepthControl, C::RenderOverride, C::RenderOverride2, C::Eqaa, C::VertexControl,
          C::Conservative, C::AaMaskTop, C::AaMaskBottom, C::RenderControl})
        if (!coverage.has(control)) return "fragment-draw-original-coverage-control-unavailable";
    const uint32_t depth_control = coverage.word(C::DepthControl);
    if (depth_control & 0xfu) return "fragment-draw-depth-stencil-coverage-unimplemented";
    // DB_DEPTH_CONTROL's defined gfx10.3 fields (pm4_registers.hpp, from AMD PAL's register map):
    // enables 0..3, ZFUNC 4..6, BACKFACE_ENABLE 7, STENCILFUNC 8..10, STENCILFUNC_BF 20..22,
    // ENABLE_COLOR_WRITES_ON_DEPTH_FAIL 30, DISABLE_COLOR_WRITES_ON_DEPTH_PASS 31. A set bit
    // outside those has no proved meaning here and is refused rather than assumed inert.
    if (depth_control & 0x3f8ff800u)
        return "fragment-draw-depth-control-reserved-bits-unimplemented";
    // With Z, stencil and bounds tests all disabled (checked above) no fragment can fail the depth
    // stage, so DISABLE_COLOR_WRITES_ON_DEPTH_PASS means this draw writes no colour at all. The
    // shipping colour state (resolve_pipeline_state) does not yet apply that, so the replay would
    // publish colour hardware suppresses (#4397): refuse until both paths carry the same gate.
    // ENABLE_COLOR_WRITES_ON_DEPTH_FAIL only adds writes for failing fragments, of which there
    // are none here, so it is admitted as inert.
    // CONFIDENCE: MED -- that a disabled test counts as "pass" is inferred from the field names,
    // not measured; the bit-31 refusal does not depend on it.
    if (depth_control & 0x80000000u)
        return "fragment-draw-color-writes-disabled-on-depth-pass-unimplemented";
    if (coverage.word(C::RenderControl) & 0x101fu)
        return "fragment-draw-depth-clear-copy-or-decompress-unimplemented";
    if (coverage.word(C::RenderOverride) & 0x867fu)
        return "fragment-draw-forced-depth-color-or-noop-coverage-unimplemented";
    if (coverage.word(C::RenderOverride2) & 0x9fu)
        return "fragment-draw-partial-squad-or-color-validation-unimplemented";
    if (coverage.word(C::Eqaa) & 0x0f007777u) return "fragment-draw-eqaa-coverage-unimplemented";
    if (coverage.word(C::Conservative)) return "fragment-draw-conservative-coverage-unimplemented";
    if ((coverage.word(C::AaMaskTop) & 0x00010001u) != 0x00010001u ||
        (coverage.word(C::AaMaskBottom) & 0x00010001u) != 0x00010001u)
        return "fragment-draw-sample-mask-coverage-unimplemented";
    if (!(coverage.word(C::VertexControl) & 1u))
        return "fragment-draw-integer-pixel-center-unimplemented";
    // Sample locations/centroid priorities and BC optimization are not read by this position-XY
    // quad-local class. Their absence stays absence: this function supplies no BC/system/M0 value.
    return nullptr;
}
}   // namespace prosper::gpu
