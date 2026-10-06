// The merged-NGG raster commit (#3135 P3): the pass-through vertex stage that rasterizes a subgroup
// shell's export record buffer, its layer routes, and its connectivity checks.
//
// End to end, Kena's captured LUT producer runs in the P2 shell under the P1 planner's launch, and
// its export buffer is drawn into a 32-layer target: every layer must be fully covered by its two
// triangles with PARAM.xy equal to the pixel's normalized screen position, and no violation may be
// counted. The layer routes (vertex-stage gl_Layer, a forwarding geometry stage, the interpolation
// geometry stage) must give identical pixels.
//
// Hand-built export buffers then pin each rule with a control that breaks it: the layer comes from
// the PROVOKING vertex (a triangle whose corners disagree); PROVOKING_VTX_LAST rotates the corners
// (a flat attribute, with back-face culling on to prove the winding survives); the connectivity
// checks keep a corrupted index or a null primitive from drawing; an invalid block and an
// out-of-range layer draw nothing and are counted; line lists rasterize as lines.
#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/ngg_raster_commit.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/ngg_raster_runner.h"
#include "fixtures/ngg_subgroup_runner.h"
#include "fixtures/test_data.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using prosper::test::NggRasterResult;
using prosper::test::NggRasterRun;

namespace {

constexpr uint32_t kLayers = 32;
constexpr uint32_t kSize = 16;

// ---- Compile-only contract ------------------------------------------------------------------

NggExportRecordLayout hand_layout() {
    // Kena's record: flags, PRIM, POS0.xyzw, POS1.xyzw (layer in z), PARAM0.xyzw.
    NggExportRecordLayout layout;
    layout.words_per_lane = 14;
    layout.pos1_word = 6;
    layout.first_param_word = 10;
    layout.param_targets = {kExpTargetParam0};
    layout.prim_channels = 1;
    layout.pos0_channels = 0xf;
    layout.pos1_channels = 4;
    layout.param_channels = {0xf};
    return layout;
}

NggRasterCommitConfig base_config(NggLayerRoute route,
                                  NggOutputTopology topology = NggOutputTopology::TriangleList) {
    NggRasterCommitConfig config;
    config.layout = hand_layout();
    config.topology = topology;
    config.layer_from_pos1 = route != NggLayerRoute::None;
    config.layer_slices = kLayers;
    config.route = route;
    return config;
}

TEST(NggRasterCommit, OutputTopologyDecodesLinesAndTrianglesOnly) {
    EXPECT_EQ(ngg_output_topology(0), NggOutputTopology::Unsupported) << "points";
    EXPECT_EQ(ngg_output_topology(1), NggOutputTopology::LineList);
    EXPECT_EQ(ngg_output_topology(2), NggOutputTopology::TriangleList);
    EXPECT_EQ(ngg_output_topology(3), NggOutputTopology::Unsupported) << "rect list";
    EXPECT_EQ(ngg_output_topology(0x42), NggOutputTopology::TriangleList) << "only [5:0] counts";
    NggRasterCommitConfig config = base_config(NggLayerRoute::None);
    config.topology = NggOutputTopology::Unsupported;
    std::string why;
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_EQ(why, "reason=ngg-raster-topology-unsupported");
}

TEST(NggRasterCommit, LayerRouteOrder) {
    std::string why;
    NggLayerRouteQuery query;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::None) << "no layer, no route";
    EXPECT_TRUE(why.empty());
    query.layer_from_pos1 = true;
    query.shader_output_layer = query.geometry_shader = true;
    query.interpolation_geometry_required = true;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::InterpolationGeometry);
    query.interpolation_geometry_required = false;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::ShaderOutputLayer);
    query.shader_output_layer = false;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::ForwardingGeometry);
    query.geometry_shader = false;
    query.interpolation_geometry_required = true;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::None);
    EXPECT_NE(why.find("reason=ngg-layer-route-unavailable"), std::string::npos) << why;
    query.shader_output_layer = true;
    EXPECT_EQ(select_ngg_layer_route(query, &why), NggLayerRoute::ShaderOutputLayer)
        << "an interpolation stage the device cannot run is no route";
}

TEST(NggRasterCommit, LayerConfigurationsThatCannotWorkAreRefused) {
    std::string why;
    NggRasterCommitConfig config = base_config(NggLayerRoute::ForwardingGeometry);
    config.layout.pos1_word = kNggRecordAbsent;
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_EQ(why, "reason=ngg-layer-without-pos1");
    config = base_config(NggLayerRoute::ForwardingGeometry);
    config.layout.pos1_channels = 1;   // POS1.x only: no layer channel
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_EQ(why, "reason=ngg-layer-without-pos1");
    config = base_config(NggLayerRoute::None);
    config.layer_from_pos1 = true;
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_NE(why.find("reason=ngg-layer-route-unavailable"), std::string::npos) << why;
    config = base_config(NggLayerRoute::ForwardingGeometry);
    config.layer_slices = 0;
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_EQ(why, "reason=ngg-raster-config");
    config = base_config(NggLayerRoute::ForwardingGeometry);
    config.reserved_locations = ~1u;   // PARAM0 takes location 0 and everything else is reserved
    EXPECT_TRUE(build_ngg_raster_commit_vertex(config, nullptr, &why).empty());
    EXPECT_EQ(why, "reason=ngg-raster-layer-location-exhausted");
}

// The Location of every Output variable a module declares, as a mask.
uint32_t output_location_mask(const std::vector<uint32_t>& spirv) {
    std::map<uint32_t, uint32_t> location;
    std::set<uint32_t> outputs;
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16, opcode = spirv[word] & 0xffffu;
        if (!count) break;
        if (opcode == 71 && count >= 4 && spirv[word + 2] == 30)   // OpDecorate Location
            location[spirv[word + 1]] = spirv[word + 3];
        if (opcode == 59 && count >= 4 && spirv[word + 3] == 3)   // OpVariable Output
            outputs.insert(spirv[word + 2]);
        word += count;
    }
    uint32_t mask = 0;
    for (uint32_t id : outputs)
        if (const auto it = location.find(id); it != location.end()) mask |= 1u << it->second;
    return mask;
}

// The PARAMs reach exactly the locations the owned-wave export commit publishes them to, and the
// geometry-route layer output takes the first location none of them (nor a reserved one) uses.
TEST(NggRasterCommit, ParamsAndTheLayerTakeTheRoutedLocations) {
    PixelInputMapping mapping;
    mapping.valid_mask = 0b111;
    mapping.controls[0] = 0;   // input 0 <- PARAM0
    mapping.controls[1] = 2;   // input 1 <- PARAM2 (not exported)
    mapping.controls[2] = 0;   // input 2 <- PARAM0 as well
    const auto owned = build_owned_vertex_export_commit({0u}, 3u, &mapping, {});
    ASSERT_FALSE(owned.empty());
    ASSERT_EQ(output_location_mask(owned), 0b101u);

    NggRasterCommitConfig config = base_config(NggLayerRoute::ShaderOutputLayer);
    config.pixel_inputs = &mapping;
    NggRasterCommitInterface published;
    std::string why;
    const auto direct = build_ngg_raster_commit_vertex(config, &published, &why);
    ASSERT_FALSE(direct.empty()) << why;
    EXPECT_EQ(published.output_mask, 0b101u);
    EXPECT_EQ(output_location_mask(direct), 0b101u) << "the same locations as the owned commit";
    EXPECT_EQ(published.layer_location, kNggRecordAbsent) << "the vertex stage writes gl_Layer";
    EXPECT_TRUE(build_ngg_layer_forward_geometry(published).empty())
        << "no layer location, no forwarding stage";

    config.route = NggLayerRoute::ForwardingGeometry;
    const auto forwarded = build_ngg_raster_commit_vertex(config, &published, &why);
    ASSERT_FALSE(forwarded.empty()) << why;
    EXPECT_EQ(published.layer_location, 1u);
    EXPECT_EQ(output_location_mask(forwarded), 0b111u) << "PARAMs at 0 and 2, the layer at 1";
    EXPECT_EQ(published.vertices_per_primitive, 3u);
    EXPECT_FALSE(build_ngg_layer_forward_geometry(published).empty());
    config.reserved_locations = 0b10;
    ASSERT_FALSE(build_ngg_raster_commit_vertex(config, &published, &why).empty()) << why;
    EXPECT_EQ(published.layer_location, 3u);
}

// ---- Execution ------------------------------------------------------------------------------

struct Rendered {
    bool ok = false;   // the draw ran and was read back
    NggRasterResult result;
    std::string refusal;
};

// Rasterize `export_words` through the commit built from `config`, with the PS reading location
// `ps_location` (Flat when `flat`). The route's geometry stage is built here.
Rendered render(const NggRasterCommitConfig& config, const std::vector<uint32_t>& export_words,
                uint32_t blocks, uint32_t ps_location = 0, bool flat = false,
                VkCullModeFlags cull = VK_CULL_MODE_NONE) {
    Rendered out;
    NggRasterCommitInterface published;
    NggRasterCommitConfig effective = config;
    FragmentInterpolationLayout interpolation;
    interpolation.attribute_mask = interpolation.smooth_mask = 1u << ps_location;
    interpolation.requires_geometry = true;
    if (config.route == NggLayerRoute::InterpolationGeometry)
        effective.reserved_locations |= interpolation.attribute_mask;
    NggRasterRun run;
    run.vertex = build_ngg_raster_commit_vertex(effective, &published, &out.refusal);
    if (run.vertex.empty()) return out;
    if (effective.route == NggLayerRoute::ForwardingGeometry)
        run.geometry = build_ngg_layer_forward_geometry(published);
    if (effective.route == NggLayerRoute::InterpolationGeometry)
        run.geometry = recompile_interpolation_geometry(interpolation, false, false, {}, false,
                                                        published.layer_location);
    if ((effective.route == NggLayerRoute::ForwardingGeometry ||
         effective.route == NggLayerRoute::InterpolationGeometry) &&
        run.geometry.empty()) {
        out.refusal = "geometry stage refused";
        return out;
    }
    run.fragment = prosper::test::ngg_param_fragment(ps_location, flat);
    run.export_words = export_words;
    run.counters = effective.count_violations;
    run.vertex_count = ngg_raster_commit_vertex_count(effective, blocks);
    run.topology = effective.topology == NggOutputTopology::LineList
                       ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                       : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    run.cull = cull;
    run.width = run.height = kSize;
    run.layers = kLayers;
    run.shader_output_layer = effective.route == NggLayerRoute::ShaderOutputLayer;
    auto result = prosper::test::run_ngg_raster(run);
    out.ok = result.has_value();
    if (result) out.result = std::move(*result);
    return out;
}

// The routes this device can run, best first.
std::vector<NggLayerRoute> available_routes() {
    const auto device = prosper::test::ngg_raster_device();
    std::vector<NggLayerRoute> routes;
    if (!device.present || !device.vertex_stores) return routes;
    if (device.shader_output_layer) routes.push_back(NggLayerRoute::ShaderOutputLayer);
    if (device.geometry) routes.push_back(NggLayerRoute::ForwardingGeometry);
    if (device.geometry) routes.push_back(NggLayerRoute::InterpolationGeometry);
    return routes;
}

bool covered(const NggRasterResult& r, uint32_t layer, uint32_t x, uint32_t y) {
    return r.at(layer, x, y)[0] != -1.0f;
}

uint32_t covered_pixels(const NggRasterResult& r, uint32_t layer) {
    uint32_t count = 0;
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) count += covered(r, layer, x, y);
    return count;
}

// ---- Kena end to end ----

std::vector<uint32_t> kena_exports(NggExportRecordLayout* layout) {
    using namespace prosper::test::ngg;
    NggDrawShape draw;
    draw.topology = NggInputTopology::TriangleStrip;
    draw.vertex_count = 4;
    draw.instance_count = kLayers;
    const NggSubgroupPlan plan = plan_ngg_subgroups(draw, kena_limits());
    if (!plan.ok() || plan.subgroups.size() != kLayers) return {};
    const auto linked = kena_linked(prosper::test::tests_root(__FILE__) / "data");
    const ShaderResourceTable table = kena_resources(4);
    NggSubgroupShellConfig shell;
    shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(kKenaRsrc2Gs);
    shell.user_sgprs = kKenaUserSgprs;
    std::string why;
    prosper::test::NggSubgroupDispatch dispatch;
    dispatch.spirv =
        recompile_ngg_subgroup(linked.data(), linked.size(), &table, shell, layout,
                               {RecompileDiagnosticStage::Vertex, 0x5009440000ull}, &why);
    if (dispatch.spirv.empty()) {
        ADD_FAILURE() << why;
        return {};
    }
    dispatch.workgroups = kLayers;
    dispatch.launch = launch_records(plan, kena_limits(), 1);
    dispatch.export_words = kLayers * layout->block_words(1);
    dispatch.guest_buffers = kena_buffers(vertex_records(kLutQuad));
    dispatch.push_constants.assign(kKenaUserSgprs, 0u);
    return prosper::test::run_ngg_subgroup(dispatch).value_or(std::vector<uint32_t>{});
}

::testing::AssertionResult is_complete_lut(const NggRasterResult& r) {
    for (uint32_t layer = 0; layer < kLayers; ++layer)
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float* p = r.at(layer, x, y);
                const float sx = (static_cast<float>(x) + 0.5f) / kSize;
                const float sy = (static_cast<float>(y) + 0.5f) / kSize;
                if (std::fabs(p[0] - sx) > 1e-4f || std::fabs(p[1] - sy) > 1e-4f)
                    return ::testing::AssertionFailure()
                           << "layer " << layer << " pixel (" << x << "," << y << ") = (" << p[0]
                           << "," << p[1] << "), expected (" << sx << "," << sy << ")";
            }
    if (r.counters != std::array<uint32_t, 3>{})
        return ::testing::AssertionFailure()
               << "violations " << r.counters[0] << "/" << r.counters[1] << "/" << r.counters[2];
    return ::testing::AssertionSuccess();
}

TEST(NggRasterCommit, KenaLutCoversEveryLayerThroughEachRoute) {
    const auto routes = available_routes();
    if (routes.empty()) GTEST_SKIP() << "no layer route with vertex-stage stores on this device";
    NggExportRecordLayout layout;
    const auto exports = kena_exports(&layout);
    ASSERT_FALSE(exports.empty());
    ASSERT_EQ(layout.words_per_lane, 14u);
    ASSERT_NE(layout.pos1_word, kNggRecordAbsent);
    std::vector<float> first;   // the first route's pixels
    for (NggLayerRoute route : routes) {
        NggRasterCommitConfig config = base_config(route);
        config.layout = layout;
        const Rendered rendered = render(config, exports, kLayers);
        ASSERT_TRUE(rendered.ok) << rendered.refusal << " route " << static_cast<int>(route);
        std::printf("[ngg-raster] Kena LUT through route %d\n", static_cast<int>(route));
        EXPECT_TRUE(is_complete_lut(rendered.result)) << "route " << static_cast<int>(route);
        // Every route must give the same pixels as the first one this device runs.
        if (first.empty())
            first = rendered.result.pixels;
        else
            EXPECT_EQ(rendered.result.pixels, first) << "route " << static_cast<int>(route);
    }
}

// Without the layer, all 64 triangles land on layer 0: the routes really are what spreads them.
TEST(NggRasterCommit, KenaLutWithoutTheLayerCollapsesOntoLayerZero) {
    const auto routes = available_routes();
    if (routes.empty()) GTEST_SKIP() << "no layer route with vertex-stage stores on this device";
    NggExportRecordLayout layout;
    const auto exports = kena_exports(&layout);
    ASSERT_FALSE(exports.empty());
    NggRasterCommitConfig config = base_config(NggLayerRoute::None);
    config.layout = layout;
    const Rendered rendered = render(config, exports, kLayers);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    EXPECT_FALSE(is_complete_lut(rendered.result));
    EXPECT_EQ(covered_pixels(rendered.result, 0), kSize * kSize);
    for (uint32_t layer = 1; layer < kLayers; ++layer)
        EXPECT_EQ(covered_pixels(rendered.result, layer), 0u) << "layer " << layer;
}

// ---- Hand-built export buffers ----

struct HandVertex {
    std::array<float, 4> pos{0, 0, 0, 1};
    uint32_t layer = 0;
    std::array<float, 4> param{};
};

// One-wave blocks in hand_layout().
struct HandBlocks {
    NggExportRecordLayout layout = hand_layout();
    std::vector<uint32_t> words;
    uint32_t blocks = 0;
    uint32_t add_block(uint32_t verts, uint32_t prims) {
        words.resize(words.size() + layout.block_words(1), 0u);
        const uint32_t base = blocks * layout.block_words(1);
        words[base + kNggHeaderVertsAlloc] = verts;
        words[base + kNggHeaderPrimsAlloc] = prims;
        words[base + kNggHeaderAllocRequests] = 1;
        return blocks++;
    }
    uint32_t* header(uint32_t block) {
        return &words[static_cast<size_t>(block) * layout.block_words(1)];
    }
    uint32_t* record(uint32_t block, uint32_t thread) {
        return &words[layout.record_offset(1, block, thread)];
    }
    void vertex(uint32_t block, uint32_t thread, const HandVertex& v) {
        uint32_t* r = record(block, thread);
        r[kNggRecordFlagsWord] |= kNggFlagPos0 | kNggFlagPos1 | (1u << kNggFlagParamShift);
        for (uint32_t c = 0; c < 4; ++c) {
            r[kNggRecordPos0Word + c] = std::bit_cast<uint32_t>(v.pos[c]);
            r[layout.param_word(0) + c] = std::bit_cast<uint32_t>(v.param[c]);
        }
        r[layout.pos1_word + 2] = v.layer;
    }
    void prim(uint32_t block, uint32_t thread, uint32_t a, uint32_t b, uint32_t c = 0,
              bool null = false) {
        uint32_t* r = record(block, thread);
        r[kNggRecordFlagsWord] |= kNggFlagPrim;
        r[kNggRecordPrimWord] = a | b << 10 | c << 20 | (null ? 1u << 31 : 0u);
    }
};

HandVertex at(float x, float y, uint32_t layer, float tag) {
    HandVertex v;
    v.pos = {x, y, 0, 1};
    v.layer = layer;
    v.param = {tag, tag, tag, 1};
    return v;
}

// A full-screen triangle pair on `layer`, as threads 0..3 and primitives 0..1 of `block`.
void quad(HandBlocks& h, uint32_t block, uint32_t layer) {
    h.vertex(block, 0, at(-1, -1, layer, 0.5f));
    h.vertex(block, 1, at(1, -1, layer, 0.5f));
    h.vertex(block, 2, at(-1, 1, layer, 0.5f));
    h.vertex(block, 3, at(1, 1, layer, 0.5f));
    h.prim(block, 0, 0, 1, 2);
    h.prim(block, 1, 2, 1, 3);
}

NggLayerRoute first_route() {
    const auto routes = available_routes();
    return routes.empty() ? NggLayerRoute::None : routes.front();
}

// One triangle whose corners carry layers 3, 4 and 5. The layer is the provoking vertex's: corner
// 0's by default, corner 2's under PROVOKING_VTX_LAST. Taking it from corner 0 regardless is wrong.
TEST(NggRasterCommit, TheLayerComesFromTheProvokingVertex) {
    const auto routes = available_routes();
    if (routes.empty()) GTEST_SKIP() << "no layer route on this device";
    for (NggLayerRoute route : routes) {
        SCOPED_TRACE(static_cast<int>(route));
        HandBlocks h;
        h.add_block(3, 1);
        h.vertex(0, 0, at(-1, -1, 3, 0.25f));
        h.vertex(0, 1, at(3, -1, 4, 0.25f));
        h.vertex(0, 2, at(-1, 3, 5, 0.25f));
        h.prim(0, 0, 0, 1, 2);
        const auto layer_hit = [&](bool last, bool from_first) {
            NggRasterCommitConfig config = base_config(route);
            config.provoking_vertex_last = last;
            config.layer_from_first_corner_for_test = from_first;
            const Rendered rendered = render(config, h.words, h.blocks);
            EXPECT_TRUE(rendered.ok) << rendered.refusal;
            std::vector<uint32_t> hit;
            if (rendered.ok)
                for (uint32_t layer = 0; layer < kLayers; ++layer)
                    if (covered_pixels(rendered.result, layer)) hit.push_back(layer);
            return hit;
        };
        EXPECT_EQ(layer_hit(false, false), std::vector<uint32_t>{3});
        EXPECT_EQ(layer_hit(true, false), std::vector<uint32_t>{5});
        // The control: the first corner under PROVOKING_VTX_LAST draws on the wrong layer.
        EXPECT_EQ(layer_hit(true, true), std::vector<uint32_t>{3});
    }
}

// One full-screen triangle with a different PARAM per corner and a FLAT fragment input, drawn with
// back faces culled. Under PROVOKING_VTX_LAST the flat value must be corner 2's; the unrotated
// control gives corner 0's. The triangle is front-facing as exported, so it must still draw after
// the rotation: a winding-reversing reorder would cull it.
TEST(NggRasterCommit, ProvokingVertexLastRotatesTheCornersAndKeepsTheWinding) {
    const NggLayerRoute route = first_route();
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    HandBlocks h;
    h.add_block(3, 1);
    // With the flipped viewport, (-1,-1) (3,-1) (-1,3) is clockwise on screen, which the
    // pipeline's counter-clockwise front face plus y-flip makes front-facing.
    h.vertex(0, 0, at(-1, -1, 0, 0.125f));
    h.vertex(0, 1, at(3, -1, 0, 0.25f));
    h.vertex(0, 2, at(-1, 3, 0, 0.375f));
    h.prim(0, 0, 0, 1, 2);
    const auto flat_value = [&](bool last, bool skip_rotation, VkCullModeFlags cull) {
        NggRasterCommitConfig config = base_config(route);
        config.provoking_vertex_last = last;
        config.skip_provoking_rotation_for_test = skip_rotation;
        const Rendered rendered = render(config, h.words, h.blocks, 0, true, cull);
        EXPECT_TRUE(rendered.ok) << rendered.refusal;
        if (!rendered.ok || covered_pixels(rendered.result, 0) != kSize * kSize) return -2.0f;
        return rendered.result.at(0, 7, 7)[0];
    };
    // Which face is front is fixed by the target, not by the test: establish it unculled first.
    const float none = flat_value(false, false, VK_CULL_MODE_NONE);
    EXPECT_EQ(none, 0.125f) << "provoking first: corner 0's value";
    const float front = flat_value(false, false, VK_CULL_MODE_BACK_BIT);
    const float back = flat_value(false, false, VK_CULL_MODE_FRONT_BIT);
    ASSERT_TRUE((front == 0.125f) != (back == 0.125f)) << "exactly one cull mode keeps it";
    const VkCullModeFlags keep = front == 0.125f ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_FRONT_BIT;
    EXPECT_EQ(flat_value(true, false, keep), 0.375f) << "rotated: corner 2's value, still drawn";
    EXPECT_EQ(flat_value(true, true, keep), 0.125f) << "the unrotated control";
}

// Block 0: a valid quad on layer 1, and a well-formed primitive past prims_alloc. Block 1: primitive 0 indexes thread 5 >= verts_alloc 4 (a
// written vertex covering the whole screen), primitive 1 is null with indices that would also draw
// it, primitive 2 was never written. Only block 0's quad may draw; the corrupted index and the
// unwritten primitive are violations, the null primitive is not.
TEST(NggRasterCommit, ConnectivityChecksKeepCorruptRecordsFromDrawing) {
    const NggLayerRoute route = first_route();
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    HandBlocks h;
    h.add_block(4, 2);
    quad(h, 0, 1);
    // Thread 2 holds a well-formed primitive on layer 3, but it is past prims_alloc (2).
    h.vertex(0, 4, at(-1, -1, 3, 0.75f));
    h.vertex(0, 5, at(3, -1, 3, 0.75f));
    h.vertex(0, 6, at(-1, 3, 3, 0.75f));
    h.prim(0, 2, 4, 5, 6);
    h.header(0)[kNggHeaderVertsAlloc] = 7;
    h.add_block(4, 3);
    h.vertex(1, 0, at(-1, -1, 2, 0.75f));
    h.vertex(1, 1, at(3, -1, 2, 0.75f));
    h.vertex(1, 5, at(-1, 3, 2, 0.75f));   // written, but beyond verts_alloc
    h.prim(1, 0, 0, 1, 5);
    h.prim(1, 1, 0, 1, 5, /*null=*/true);
    NggRasterCommitConfig config = base_config(route);
    Rendered rendered = render(config, h.words, h.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    EXPECT_EQ(covered_pixels(rendered.result, 1), kSize * kSize);
    EXPECT_EQ(covered_pixels(rendered.result, 2), 0u);
    EXPECT_EQ(covered_pixels(rendered.result, 3), 0u) << "a slot past prims_alloc never draws";
    EXPECT_EQ(rendered.result.counters[kNggViolationConnectivity], 2u)
        << "the out-of-range index and the unwritten primitive below prims_alloc";
    EXPECT_EQ(rendered.result.counters[kNggViolationInvalidBlocks], 0u);

    // The control: without the checks, the corrupted primitive draws on layer 2.
    config.skip_connectivity_for_test = true;
    rendered = render(config, h.words, h.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    EXPECT_GT(covered_pixels(rendered.result, 2), 0u);
    EXPECT_EQ(covered_pixels(rendered.result, 3), 0u) << "prims_alloc is not a connectivity check";

    // The null bit alone: an otherwise valid primitive with bit 31 set does not draw, and is not a
    // violation.
    HandBlocks n;
    n.add_block(4, 2);
    quad(n, 0, 1);
    n.record(0, 1)[kNggRecordPrimWord] |= 1u << 31;
    config.skip_connectivity_for_test = false;
    rendered = render(config, n.words, n.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    EXPECT_GT(covered_pixels(rendered.result, 1), 0u);
    EXPECT_LT(covered_pixels(rendered.result, 1), kSize * kSize) << "half the quad is null";
    EXPECT_EQ(rendered.result.counters, (std::array<uint32_t, 3>{}));
}

// A vertex that never exported POS0 is not a vertex: a primitive naming one does not draw.
TEST(NggRasterCommit, AnIndexMustNameAWrittenVertex) {
    const NggLayerRoute route = first_route();
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    HandBlocks h;
    h.add_block(4, 2);
    quad(h, 0, 1);
    h.record(0, 3)[kNggRecordFlagsWord] &= ~kNggFlagPos0;
    const Rendered rendered = render(base_config(route), h.words, h.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    EXPECT_EQ(rendered.result.counters[kNggViolationConnectivity], 1u);
    const uint32_t covered_layer1 = covered_pixels(rendered.result, 1);
    EXPECT_GT(covered_layer1, 0u);
    EXPECT_LT(covered_layer1, kSize * kSize);
}

// A block whose header says the guest did not allocate exactly once draws nothing; it is counted
// once. A primitive whose layer is past the target's slices draws nothing and is counted.
TEST(NggRasterCommit, InvalidBlocksAndOutOfRangeLayersDrawNothing) {
    const NggLayerRoute route = first_route();
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    HandBlocks h;
    h.add_block(4, 2);
    quad(h, 0, 4);
    h.header(0)[kNggHeaderAllocRequests] = 2;
    h.add_block(4, 2);
    quad(h, 1, 5);
    h.header(1)[kNggHeaderLaunchMismatches] = 1;
    h.add_block(4, 2);
    quad(h, 2, kLayers + 8);
    h.add_block(4, 2);
    quad(h, 3, 6);
    const Rendered rendered = render(base_config(route), h.words, h.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    for (uint32_t layer = 0; layer < kLayers; ++layer)
        EXPECT_EQ(covered_pixels(rendered.result, layer), layer == 6 ? kSize * kSize : 0u)
            << "layer " << layer;
    EXPECT_EQ(rendered.result.counters[kNggViolationInvalidBlocks], 2u);
    EXPECT_EQ(rendered.result.counters[kNggViolationLayerCulled], 2u) << "both triangles";
    EXPECT_EQ(rendered.result.counters[kNggViolationConnectivity], 0u);
}

// A line list: PRIM carries two indices. One horizontal line on layer 1, one vertical on layer 2,
// one on an out-of-range layer.
TEST(NggRasterCommit, LineListsRasterizeAsLines) {
    const NggLayerRoute route = first_route();
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    HandBlocks h;
    h.add_block(6, 3);
    // Pixel row 4 has centre y = 4.5, i.e. clip y = 1 - 2 * 4.5 / 16 under the flipped viewport.
    const float row = 1.0f - 2.0f * 4.5f / kSize, column = -1.0f + 2.0f * 9.5f / kSize;
    // The endpoints lie past the target's edges, so every pixel of the row or column is crossed.
    h.vertex(0, 0, at(-1.5f, row, 1, 0.5f));
    h.vertex(0, 1, at(1.5f, row, 1, 0.5f));
    h.vertex(0, 2, at(column, -1.5f, 2, 0.5f));
    h.vertex(0, 3, at(column, 1.5f, 2, 0.5f));
    h.vertex(0, 4, at(-1, row, kLayers, 0.5f));
    h.vertex(0, 5, at(1, row, kLayers, 0.5f));
    h.prim(0, 0, 0, 1);
    h.prim(0, 1, 2, 3);
    h.prim(0, 2, 4, 5);
    const Rendered rendered =
        render(base_config(route, NggOutputTopology::LineList), h.words, h.blocks);
    ASSERT_TRUE(rendered.ok) << rendered.refusal;
    const NggRasterResult& r = rendered.result;
    for (uint32_t x = 0; x < kSize; ++x) EXPECT_TRUE(covered(r, 1, x, 4)) << "x " << x;
    for (uint32_t y = 0; y < kSize; ++y) EXPECT_TRUE(covered(r, 2, 9, y)) << "y " << y;
    EXPECT_EQ(covered_pixels(r, 1), kSize) << "one row, not a triangle";
    EXPECT_EQ(covered_pixels(r, 2), kSize) << "one column";
    EXPECT_EQ(r.counters[kNggViolationLayerCulled], 1u);
    for (uint32_t layer = 3; layer < kLayers; ++layer)
        EXPECT_EQ(covered_pixels(r, layer), 0u) << "layer " << layer;
}

}   // namespace
