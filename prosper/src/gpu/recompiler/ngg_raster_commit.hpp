// ngg_raster_commit.hpp -- rasterize the primitives a merged ES+GS NGG subgroup shell exported
// (#3135 phase P3). The shell (ngg_subgroup_shell.hpp) has already run the guest; this stage only
// reads its export record buffer (ngg_export_record.hpp) and hands the primitives to the fixed
// function rasterizer. Compile and offline execution only: nothing live draws through it yet.
//
// THE DRAW. A non-indexed, non-instanced draw of blocks * 64W * K vertices, K = 3 for triangles and
// 2 for lines. Vertex `idx` is corner idx % K of primitive slot idx / K; slot s is thread
// s % 64W of block s / 64W. Every slot is drawn; an empty one is a degenerate primitive.
//
// CONNECTIVITY. Slot p of a block is drawn only when ALL of these hold, otherwise all its corners
// collapse onto one point outside the clip volume:
//   * the block is valid: alloc_requests == 1, stray_requests == 0, launch_mismatches == 0, and
//     verts_alloc and prims_alloc both fit the block's 64W records (a larger allocation would lose
//     primitives that have no record, so the whole block is refused and counted instead);
//   * p < prims_alloc;
//   * thread p exported PRIM, and its null bit 31 is clear;
//   * each of the K 9-bit indices is below verts_alloc and below 64W, and names a thread that
//     exported POS0 (and POS1 when the layer is read).
// Every reason except "p >= prims_alloc" and "null" is a protocol VIOLATION and is counted (below).
//
// CORNER ORDER. Vulkan's provoking vertex is the first. Under PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST
// the guest's provoking vertex is its last, so output corner c takes guest corner (c + K - 1) % K:
// a rotation, which keeps a triangle's winding. Flat attributes then come from the right vertex.
// For a line this swaps the endpoints. CONFIDENCE: MED -- it is the inference that the provoking
// endpoint must be Vulkan's first; it also moves which endpoint pixel the half-open line rule
// leaves out, and no test (nor hardware evidence) pins that pixel.
//
// THE LAYER. When `layer_from_pos1` (PA_CL_VS_OUT_CNTL.USE_VTX_RENDER_TARGET_INDX) is set, the layer
// is POS1.z's raw bits of the guest's PROVOKING vertex, and every corner carries that one value. A
// layer at or above `layer_slices` culls the primitive and is counted. CONFIDENCE: LOW on cull
// versus clamp: the hardware may clamp an out-of-range index to the view's SLICE_MAX instead. Culling
// is the fail-visible choice (the counter names it); a title that depends on the clamp would show it.
// How the layer reaches
// gl_Layer is the route (`select_ngg_layer_route`):
//   * ShaderOutputLayer   the vertex stage writes BuiltIn Layer (SPV_EXT_shader_viewport_index_layer;
//                         the device must enable VK_EXT_shader_viewport_index_layer);
//   * a geometry route    the vertex stage writes the layer as a flat uint output at
//                         `NggRasterCommitInterface::layer_location`, and a geometry stage
//                         (build_ngg_layer_forward_geometry, or the interpolation geometry stage
//                         built with that location) writes BuiltIn Layer.
//
// ATTRIBUTES. Each PARAM the program exported is published to every fragment-input location the
// ordinary vertex compiler would publish it to (param_ps_routing.hpp).
//
// BINDINGS (descriptor set 2, the shell's set):
//   binding 1  the export record buffer, read only;
//   binding 2  violation counters, when `count_violations` (the vertex stage then needs
//              vertexPipelineStoresAndAtomics). Each count is an atomic add made by corner 0 of a
//              primitive, so it is at least the number of offending primitives; an implementation
//              that shades a vertex twice may count it twice. Zero is exact.
#pragma once

#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {

inline constexpr uint32_t kNggRasterDescriptorSet = 2;   // the shell's set (ngg_subgroup_shell.hpp)
inline constexpr uint32_t kNggRasterExportBinding = 1;
inline constexpr uint32_t kNggRasterCounterBinding = 2;
// The counter buffer, in words.
inline constexpr uint32_t kNggViolationInvalidBlocks = 0;   // once per invalid block (slot 0)
inline constexpr uint32_t kNggViolationConnectivity = 1;   // once per malformed primitive
inline constexpr uint32_t kNggViolationLayerCulled =
    2;   // once per primitive whose layer is out of range
inline constexpr uint32_t kNggViolationWords = 3;

// VGT_GS_OUT_PRIM_TYPE.OUTPRIM_TYPE [5:0]: 0 points, 1 line strip, 2 triangle strip, 3 rect list.
// Under NGG the PRIM export names every vertex of one primitive, so "strip" here selects lines vs
// triangles; points and rect lists are refused.
enum class NggOutputTopology : uint8_t { Unsupported, LineList, TriangleList };
NggOutputTopology ngg_output_topology(uint32_t vgt_gs_out_prim_type);

enum class NggLayerRoute : uint8_t {
    None,   // the layer is not read; nothing writes gl_Layer
    InterpolationGeometry,   // the draw already needs the interpolation geometry stage
    ShaderOutputLayer,   // the vertex stage writes gl_Layer
    ForwardingGeometry,   // a dedicated forwarding geometry stage writes gl_Layer
};

struct NggLayerRouteQuery {
    NggOutputTopology topology = NggOutputTopology::TriangleList;
    bool layer_from_pos1 = false;   // USE_VTX_RENDER_TARGET_INDX
    bool interpolation_geometry_required = false;
    bool shader_output_layer = false;   // host capability: vertex-stage gl_Layer is enabled
    bool geometry_shader = false;   // host capability: geometryShader is enabled
};

// The route, in the order: interpolation geometry (when it is required anyway), shader output
// layer, forwarding geometry. Returns None with an EMPTY `refusal` when the layer is not read and
// nothing else is wrong. Returns None with `refusal` set to "reason=ngg-layer-route-unavailable" when
// the layer is read and no route exists, and with "reason=ngg-interpolation-geometry-needs-triangles"
// whenever the pixel shader needs the interpolation geometry stage for a line list: that stage's
// input is Triangles, and no other route can supply the AMD vertex parameters it exists for
// (the ordinary draw path refuses the same case, gpu_execute.hpp's triangle_topology).
NggLayerRoute select_ngg_layer_route(const NggLayerRouteQuery& query, std::string* refusal);

struct NggRasterCommitConfig {
    NggExportRecordLayout layout;   // what the shell wrote
    uint32_t waves = 1;   // the shell's W
    NggOutputTopology topology = NggOutputTopology::TriangleList;
    bool provoking_vertex_last = false;   // PA_SU_SC_MODE_CNTL.PROVOKING_VTX_LAST
    bool layer_from_pos1 = false;   // PA_CL_VS_OUT_CNTL.USE_VTX_RENDER_TARGET_INDX
    uint32_t layer_slices = 1;   // the bound target's layer count
    NggLayerRoute route = NggLayerRoute::None;
    // Locations the geometry-route layer output must not take besides the routed PARAMs: for the
    // interpolation route, the interpolation layout's attribute_mask (its inputs).
    uint32_t reserved_locations = 0;
    const PixelInputMapping* pixel_inputs = nullptr;
    FloatTransportConfig float_transport = {};
    bool count_violations = true;
    // Per-slice replay of a layered DEPTH target (#3135, layered NGG depth). The backend keeps one
    // single-layer depth image per guest slice, so a draw whose layer addresses a depth array is
    // drawn once per slice: each pass draws only the primitives whose layer equals the selected
    // one, with route None (nothing writes gl_Layer). The selection is a draw-time value, not a
    // compile-time one, so every slice runs the same module and pipeline: the stage reads it as
    // gl_InstanceIndex, which the backend sets through the run's firstInstance
    // (NggSubgroupDraw::layer_select; a run draws one instance). A layer at or above
    // `layer_slices` is culled in every pass and counted by the layer-0 pass alone, so once.
    bool layer_select = false;
    // TEST ONLY: draw every slot below prims_alloc whatever its connectivity. Exists so a test can
    // prove the checks are what keeps a corrupted record from drawing.
    bool skip_connectivity_for_test = false;
    // TEST ONLY: take the layer from guest corner 0 instead of the provoking vertex.
    bool layer_from_first_corner_for_test = false;
    // TEST ONLY: do not rotate the corners under PROVOKING_VTX_LAST.
    bool skip_provoking_rotation_for_test = false;
};

// What the vertex stage publishes, for the stages after it.
struct NggRasterCommitInterface {
    uint32_t vertices_per_primitive = 0;   // K
    uint32_t output_mask = 0;   // fragment-input locations carrying a routed PARAM (vec4)
    uint32_t layer_location = kNggRecordAbsent;   // flat uint layer output, geometry routes only
};

// The vertex count to draw for `blocks` blocks.
uint32_t ngg_raster_commit_vertex_count(const NggRasterCommitConfig& config, uint32_t blocks);

// The pass-through vertex stage. Returns {} with `refusal` set ("reason=<name> ...") when the
// configuration cannot be committed: ngg-raster-topology-unsupported, ngg-raster-config (waves,
// layout or slices out of range), ngg-layer-without-pos1 (the layer is read but the program exports
// no POS1.z), ngg-raster-layer-location-exhausted (no free location for the layer output),
// ngg-layer-route-unavailable (the layer is read with route None).
std::vector<uint32_t>
build_ngg_raster_commit_vertex(const NggRasterCommitConfig& config,
                               NggRasterCommitInterface* out_interface = nullptr,
                               std::string* refusal = nullptr);

// The forwarding geometry stage of the ForwardingGeometry route: reads K vertices of `commit`'s
// outputs and re-emits them unchanged with gl_Layer from the flat layer input. Returns {} when `commit`
// carries no layer location or K is not 2 or 3.
std::vector<uint32_t> build_ngg_layer_forward_geometry(const NggRasterCommitInterface& commit,
                                                       FloatTransportConfig float_transport = {});

}   // namespace prosper::gpu
