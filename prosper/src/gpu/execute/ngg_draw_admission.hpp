// ngg_draw_admission.hpp -- which merged ES+GS NGG draws the subgroup path may run, decided from the
// draw's own registers and shape (#3135 phase P5, design section 4). Pure CPU and Vulkan-free, so the
// live realization (ngg_live_draw.hpp), the Vulkan backend and gpu_replay all ask the same questions.
//
// Two halves:
//   * admit_ngg_draw -- the register and draw-shape table. It runs before anything is compiled and
//     either refuses by name or returns the configuration the planner (P1), the shell (P2) and the
//     raster commit (P3) take: partition limits, draw shape, LDS size, user SGPR count, output
//     topology, provoking vertex, whether the layer is read and the layered target's slice count.
//   * ngg_device_refusal -- whether a BUILT description fits the device: compute queue, layer route,
//     native Wave64, workgroup and buffer limits, LDS. The backend asks this before it records, and
//     the live producer asks it before it hands a description over, so the backend never refuses a
//     draw the producer admitted (and gpu_replay asks it of a captured description).
//
// Every refusal is a static string naming the rule (ngg-...), so the dropped-draw census and the
// refused-shader index can carry it without allocation.
#pragma once

#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/ngg_raster_commit.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"

#include <cstdint>

namespace prosper::gpu {

// ---- Register fields (gfx10.3) ------------------------------------------------------------------
// PA_CL_VS_OUT_CNTL. CONFIDENCE: HIGH on [15:0] and bits 16-21 (Mesa's gfx10 S_02881C_* and the
// RDNA2 register reference agree); Kena programs 0x01240000 = bits 18, 21 and 24 and writes the
// layer to POS1.z, which is bit 18's documented meaning.
inline constexpr uint32_t kVsOutClipDistanceMask = 0xffu;   // CLIP_DIST_ENA_0..7 [7:0]
inline constexpr uint32_t kVsOutCullDistanceMask = 0xff00u;   // CULL_DIST_ENA_0..7 [15:8]
inline constexpr uint32_t kVsOutUseVtxPointSize = 1u << 16;
inline constexpr uint32_t kVsOutUseVtxEdgeFlag = 1u << 17;
inline constexpr uint32_t kVsOutUseVtxRenderTargetIndx = 1u << 18;
inline constexpr uint32_t kVsOutUseVtxViewportIndx = 1u << 19;
inline constexpr uint32_t kVsOutUseVtxKillFlag = 1u << 20;
inline constexpr uint32_t kVsOutMiscVecEna = 1u << 21;
inline constexpr uint32_t kVsOutCcDist0VecEna = 1u << 22;
inline constexpr uint32_t kVsOutCcDist1VecEna = 1u << 23;
// Bits [31:25] (gfx10.3: GS cut flag, line width, VRS rate and its combiner bypasses, among
// others) are not modelled by the pass-through stage. Refused until each one is decoded.
inline constexpr uint32_t kVsOutUndecodedMask = 0xfe000000u;
// VGT_GS_INSTANCE_CNT: ENABLE [0], CNT [8:2].
inline constexpr uint32_t kGsInstanceEnable = 1u;
inline uint32_t ngg_gs_instance_count(uint32_t vgt_gs_instance_cnt) {
    return (vgt_gs_instance_cnt & kGsInstanceEnable) ? (vgt_gs_instance_cnt >> 2) & 0x7fu : 1u;
}
// SPI_PS_INPUT_ENA.FRONT_FACE_ENA [12] (RDNA2 ISA, the PS input VGPR table). CONFIDENCE: HIGH.
inline constexpr uint32_t kPsInputFrontFace = 1u << 12;
// The SPI_SHADER_PGM_RSRC2_GS user SGPR count: USER_SGPR [5:1] plus USER_SGPR_MSB [27].
inline uint32_t ngg_rsrc2_gs_user_sgprs(uint32_t rsrc2) {
    return ((rsrc2 >> 1) & 0x1fu) | (((rsrc2 >> 27) & 1u) << 5);
}

// ---- Host capabilities ----------------------------------------------------------------------------
// What the Vulkan device the backend runs on can do for this path. The backend publishes it once
// its device exists (ngg_subgroup_gpu.h); realization reads the published copy. Until published,
// nothing is admitted.
struct NggHostCapabilities {
    bool published = false;
    bool compute = false;   // the graphics queue also runs compute
    bool vertex_pipeline_stores = false;   // the pass-through stage may count violations
    bool shader_output_layer = false;   // vertex-stage gl_Layer
    bool geometry_shader = false;
    bool native_wave64 = false;   // compute requiredSubgroupSize 64 with full subgroups
    uint32_t max_compute_workgroup_subgroups = 0;
    uint32_t max_compute_shared_memory = 0;   // bytes
    uint32_t max_compute_workgroup_size_x = 0;
    uint32_t max_compute_workgroup_invocations = 0;
    uint32_t max_compute_workgroup_count_x = 0;
    uint32_t max_storage_buffer_range = 0;
    uint32_t max_push_constants_size = 0;
    bool operator==(const NggHostCapabilities&) const = default;
};
void publish_ngg_host_capabilities(const NggHostCapabilities& capabilities);
NggHostCapabilities published_ngg_host_capabilities();

// ---- The register and draw-shape table ------------------------------------------------------------
// Raw register values at the draw. `missing` names the first register the draw state lacks among
// those admission cannot default (null when all are present).
struct NggDrawRegisters {
    uint32_t vgt_shader_stages_en = 0;
    uint32_t vgt_gs_onchip_cntl = 0;
    uint32_t ge_cntl = 0;
    uint32_t ge_max_output_per_subgroup = 0;
    uint32_t vgt_gs_max_vert_out = 0;
    uint32_t vgt_esgs_ring_itemsize = 0;
    uint32_t spi_shader_pgm_rsrc2_gs = 0;
    uint32_t vgt_gs_out_prim_type = 0;
    // Absent means the reset value 0 for these.
    uint32_t vgt_gs_instance_cnt = 0;
    uint32_t pa_su_sc_mode_cntl = 0;
    uint32_t pa_cl_vs_out_cntl = 0;
    uint32_t pa_cl_clip_cntl = 0;
    uint32_t spi_ps_input_ena = 0;
    uint32_t primitive_type = 0;   // VGT_PRIMITIVE_TYPE.PRIM_TYPE (4 list, 6 strip)
    const char* missing = nullptr;
};

// The draw packet and its bound state, already decoded by the caller.
struct NggDrawFacts {
    uint32_t vertex_count = 0;
    uint32_t instance_count = 1;
    bool indexed = false;
    bool indirect = false;
    bool vertex_offset = false;   // GE_INDX_OFFSET or an indirect vertex offset is non-zero
    // Colour target 0's renderable volume view (color_target_volume_view); zero slices = not layered.
    uint32_t target_slices = 0;
    uint32_t target_first_slice = 0;
    // The user-data range the program's AGC header declares (user_data_range_start/end).
    bool user_data_range_known = false;
    uint32_t user_data_range_start = 0;
    uint32_t user_data_range_end = 0;
    // Pixel-side order-visible inputs, for strips: interpolants the pixel stage reads flat, or as
    // raw per-vertex values (explicit interpolation).
    uint32_t flat_input_mask = 0;
    uint32_t raw_vertex_input_mask = 0;
    // The pixel stage needs the interpolation geometry stage anyway.
    bool interpolation_geometry_required = false;
};

struct NggDrawAdmission {
    bool applies = false;   // a merged ES+GS NGG draw (GS_EN and PRIMGEN_EN)
    const char* refusal = nullptr;   // the rule that refused; null when admitted
    NggSubgroupLimits limits;
    NggDrawShape shape;
    uint32_t lds_granules = 0;   // RSRC2_GS.LDS_SIZE, raw
    uint32_t user_sgprs = 0;   // user_data_range_end
    NggOutputTopology topology = NggOutputTopology::TriangleList;
    bool provoking_vertex_last = false;
    bool layer_from_pos1 = false;
    uint32_t layer_slices = 1;
    NggLayerRoute route = NggLayerRoute::None;
    bool count_violations = false;
    bool native_wave64 = false;
    bool ok() const { return applies && !refusal; }
};

// The section 4 admission table. Refusal names (each a rule, in check order):
//   ngg-host-unpublished          no backend device has published its capabilities
//   ngg-register-missing          a partition/launch register is absent from the draw state
//   ngg-gs-wave32                 VGT_SHADER_STAGES_EN.GS_W32_EN (P6)
//   ngg-gs-instancing             VGT_GS_INSTANCE_CNT enabled with a count above 1 (P6)
//   ngg-input-topology            anything but a triangle list or strip (adjacency, rect, quad,
//                                 fan, points, lines, patches)
//   ngg-output-topology           VGT_GS_OUT_PRIM_TYPE points or rect list
//   ngg-indexed / ngg-indirect / ngg-vertex-offset   (P6)
//   ngg-viewport-index / ngg-point-size / ngg-clip-cull-distance / ngg-user-clip-plane /
//   ngg-vertex-kill-flag          per-vertex state the pass-through stage does not model
//   ngg-vs-out-undecoded          PA_CL_VS_OUT_CNTL bits [31:25], until each is decoded
//   ngg-layer-target-not-layered  the layer is read and colour target 0 is not a layered volume
//   ngg-layer-slice-start         the layer is read and the view's SLICE_START is not 0
//   ngg-strip-order-visible       a strip with culling, a FRONT_FACE input, a flat input or a raw
//                                 per-vertex input (odd-triangle order is open question 3).
//                                 CONFIDENCE: MED. This guards what the RASTERIZER can see of the
//                                 odd-triangle vertex order. The guest GS itself also sees it: a
//                                 GS deriving a facet normal or signed area from its inputs, or a
//                                 layer per input vertex, would produce different values on odd
//                                 triangles if the hardware swaps them. Whether a GS's outputs
//                                 depend on input order is not statically decidable here, so it is
//                                 not refused; admitted strips are counted (strip_draws, and
//                                 "input=strip" on the [ngg-live] line) for P6 to audit.
//   ngg-user-data-range           no AGC user-data range, a range not starting at 0, or more than
//                                 the shell's push-constant budget
//   ngg-user-sgpr-count           RSRC2_GS.USER_SGPR is non-zero and disagrees with the range
//   ngg-lds-limit                 RSRC2_GS.LDS_SIZE above 64 KiB or the device's shared memory
//   ngg-host-compute              no compute on the graphics queue
//   ngg-layer-route-unavailable / ngg-interpolation-geometry-needs-triangles   (route selection)
// A draw whose stages are not merged ES+GS NGG returns applies=false and no refusal: the path does
// not apply to it.
NggDrawAdmission admit_ngg_draw(const NggDrawRegisters& registers, const NggDrawFacts& facts,
                                const NggHostCapabilities& host);

// Whether `draw` fits the device `host` describes. Null when it does; otherwise one of
// ngg-backend-no-compute, ngg-backend-description, ngg-backend-vertex-stores-unavailable,
// ngg-backend-layer-route-unavailable, ngg-backend-wave64-unavailable, ngg-backend-lds-limit,
// ngg-backend-workgroup-limit, ngg-backend-buffer-range.
const char* ngg_device_refusal(const NggSubgroupDraw& draw, const NggHostCapabilities& host);

}   // namespace prosper::gpu
