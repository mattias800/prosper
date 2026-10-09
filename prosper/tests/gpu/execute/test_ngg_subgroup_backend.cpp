// The merged-NGG backend path (#3135 P4): a BackendDraw carrying an NggSubgroupDraw, submitted
// through the real backend draw and segment code (render_draws_rgba), which fills the export
// region, dispatches the subgroup shell once per wave count and draws the pass-through runs, in one
// command buffer with no CPU wait.
//
//   * Kena's LUT producer, end to end, into a retained 32-slice volume: the bytes equal the P3
//     offline result (the P2 shell and the P3 raster runners on their own device), and the backend
//     reports no violation.
//   * Two wave-count groups in one draw: a 76-vertex strip per instance plans as W=3 then W=1, so
//     the runs interleave the groups and each run must draw its own blocks.
//   * A draw between two ordinary draws: three segments, and the pixels show A, then the NGG draw,
//     then B, in that order.
//   * The zero-fill: a second frame reuses the same export slice; the shell's GS_ALLOC_REQ count
//     is an atomic add, so a stale count would invalidate every block.
//   * The violation counters are read lazily: nothing is read until the batch completes for its own
//     reasons, and an out-of-range layer is then counted.
//   * #3135 P6, indexed draws: an instanced triangle list of scattered 16- and 32-bit indices
//     covers every instance's layer, the index VALUES (not their positions) decide which half of
//     the screen one triangle covers, and each instance's primitives land on their own slice.
//   * Layered NGG depth: a depth-only VS-only draw whose layer addresses a depth array, replayed
//     per slice, leaves layer k's exact depth in slice k, and two such draws split one pass with no
//     colour carried.
#include "fixtures/render_runner.h"

#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/ngg_raster_runner.h"
#include "fixtures/ngg_subgroup_runner.h"
#include "fixtures/test_data.h"
#include "gpu/diagnostics/draw_disposition.hpp"
#include "gpu/execute/ngg_depth_slices.hpp"
#include "gpu/execute/ngg_draw_indices.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/ngg_raster_commit.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test;

namespace {

constexpr uint32_t kSize = 16;
constexpr float kClear[4] = {-1.0f, -1.0f, -1.0f, -1.0f};

const RenderVkCtx* backend() {
    const RenderVkCtx& ctx = render_vk_ctx();
    return ctx.ok && ctx.queue_supports_compute && ctx.image_view_2d_on_3d ? &ctx : nullptr;
}

// The route this backend would choose for a layered triangle draw.
NggLayerRoute backend_route(const RenderVkCtx& ctx) {
    NggLayerRouteQuery query;
    query.topology = NggOutputTopology::TriangleList;
    query.layer_from_pos1 = true;
    query.shader_output_layer = ctx.shader_output_layer_enabled;
    query.geometry_shader = ctx.geometry_shader_enabled;
    std::string why;
    return select_ngg_layer_route(query, &why);
}

struct KenaInputs {
    std::vector<uint32_t> linked;
    ShaderResourceTable table;
};

const KenaInputs& kena_inputs() {
    static const KenaInputs inputs = [] {
        KenaInputs in;
        in.linked = ngg::kena_linked(tests_root(__FILE__) / "data");
        in.table = ngg::kena_resources(4);
        return in;
    }();
    return inputs;
}

std::shared_ptr<const NggSubgroupDraw>
kena_shape_draw(const RenderVkCtx& ctx, const NggDrawShape& shape, const ShaderResourceTable& table,
                uint32_t slices, std::string* why, std::optional<NggLayerRoute> route = {}) {
    const KenaInputs& in = kena_inputs();
    NggSubgroupDrawRequest request;
    request.linked_code = in.linked.data();
    request.dwords = in.linked.size();
    request.resources = &table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.limits = ngg::kena_limits();
    request.shape = shape;
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = slices;
    request.raster.route = route ? *route : backend_route(ctx);
    request.raster.count_violations = ngg_backend_counts_violations(ctx);
    request.push_constants.assign(ngg::kKenaUserSgprs, 0u);
    request.diagnostic = {RecompileDiagnosticStage::Vertex, 0x5009440000ull};
    return build_ngg_subgroup_draw(request, why);
}

std::shared_ptr<const NggSubgroupDraw> kena_draw(const RenderVkCtx& ctx, uint32_t vertices,
                                                 uint32_t instances, uint32_t slices,
                                                 std::string* why,
                                                 std::optional<NggLayerRoute> route = {}) {
    NggDrawShape shape;
    shape.topology = NggInputTopology::TriangleStrip;
    shape.vertex_count = vertices;
    shape.instance_count = instances;
    return kena_shape_draw(ctx, shape, kena_inputs().table, slices, why, route);
}

// The guest's convention: a negative viewport height puts clip y = +1 on the top row.
ResolvedPipelineState flipped_state() {
    ResolvedPipelineState state;
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // the default 0 is a point list
    state.has_viewport = true;
    state.viewport_x = 0.0f;
    state.viewport_y = static_cast<float>(kSize);
    state.viewport_w = static_cast<float>(kSize);
    state.viewport_h = -static_cast<float>(kSize);
    return state;
}

BackendDraw ngg_backend_draw(std::shared_ptr<const NggSubgroupDraw> ngg,
                             const std::vector<uint32_t>& records,
                             const ResolvedPipelineState* state) {
    BackendDraw draw;
    draw.ngg_subgroup = std::move(ngg);
    draw.fs = ngg_param_fragment(0, false);
    draw.ps = state;
    for (const auto& [binding, words] : ngg::kena_buffers(records)) {
        FrameBufferResource buffer;
        buffer.set = 0;
        buffer.binding = binding;
        buffer.dwords = words;
        draw.B.push_back(std::move(buffer));
        draw.resource_order.push_back(0x80000000u | static_cast<uint32_t>(draw.B.size() - 1u));
    }
    return draw;
}

BackendColorTarget volume_target(uint64_t id, uint32_t depth, bool readback = true) {
    BackendColorTarget target;
    target.persistent_id = id;
    target.load_existing = false;
    target.readback = readback;
    target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    target.volume_depth = depth;
    target.volume_first_slice = 0;
    target.volume_slice_count = depth;
    return target;
}

const float* texel(const std::vector<uint8_t>& bytes, uint32_t layer, uint32_t x, uint32_t y) {
    return reinterpret_cast<const float*>(bytes.data()) +
           ((static_cast<size_t>(layer) * kSize + y) * kSize + x) * 4u;
}

// Layers [0, covered) hold PARAM.xy = the pixel's normalized screen position; the rest are clear.
::testing::AssertionResult lut_layers(const std::vector<uint8_t>& bytes, uint32_t layers,
                                      uint32_t covered, float tolerance = 1e-4f) {
    if (bytes.size() != static_cast<size_t>(layers) * kSize * kSize * 16u)
        return ::testing::AssertionFailure() << "readback of " << bytes.size() << " bytes";
    for (uint32_t layer = 0; layer < layers; ++layer)
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float* p = texel(bytes, layer, x, y);
                const float sx = (static_cast<float>(x) + 0.5f) / kSize;
                const float sy = (static_cast<float>(y) + 0.5f) / kSize;
                const bool ok = layer < covered ? std::fabs(p[0] - sx) <= tolerance &&
                                                      std::fabs(p[1] - sy) <= tolerance
                                                : p[0] == -1.0f && p[1] == -1.0f;
                if (!ok)
                    return ::testing::AssertionFailure()
                           << "layer " << layer << " pixel (" << x << "," << y << ") = (" << p[0]
                           << "," << p[1] << ")";
            }
    return ::testing::AssertionSuccess();
}

struct StatsSnapshot {
    uint64_t draws, dispatches, completed, invalid, connectivity, culled;
};
StatsSnapshot stats_now() {
    auto& s = ngg_subgroup_backend_stats();
    return {s.draws.load(),          s.dispatches.load(),   s.completed.load(),
            s.invalid_blocks.load(), s.connectivity.load(), s.layer_culled.load()};
}

// The P3 offline result: the P2 shell runner, then the P3 raster runner, on their own device.
// Empty when either runner fails.
std::vector<float> offline_kena_lut(NggLayerRoute route) {
    const KenaInputs& in = kena_inputs();
    NggDrawShape shape;
    shape.topology = NggInputTopology::TriangleStrip;
    shape.vertex_count = 4;
    shape.instance_count = 32;
    const NggSubgroupPlan plan = plan_ngg_subgroups(shape, ngg::kena_limits());
    if (!plan.ok()) return {};
    NggSubgroupShellConfig shell;
    shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    shell.user_sgprs = ngg::kKenaUserSgprs;
    NggExportRecordLayout layout;
    std::string why;
    NggSubgroupDispatch dispatch;
    dispatch.spirv =
        recompile_ngg_subgroup(in.linked.data(), in.linked.size(), &in.table, shell, &layout,
                               {RecompileDiagnosticStage::Vertex, 0x5009440000ull}, &why);
    if (dispatch.spirv.empty()) return {};
    dispatch.workgroups = 32;
    dispatch.launch = ngg::launch_records(plan, ngg::kena_limits(), 1);
    dispatch.export_words = 32 * layout.block_words(1);
    dispatch.guest_buffers = ngg::kena_buffers(ngg::vertex_records(ngg::kLutQuad));
    dispatch.push_constants.assign(ngg::kKenaUserSgprs, 0u);
    const auto exports = run_ngg_subgroup(dispatch);
    if (!exports) return {};
    NggRasterCommitConfig config;
    config.layout = layout;
    config.layer_from_pos1 = true;
    config.layer_slices = 32;
    config.route = route;
    NggRasterCommitInterface published;
    NggRasterRun run;
    run.vertex = build_ngg_raster_commit_vertex(config, &published, &why);
    if (route == NggLayerRoute::ForwardingGeometry)
        run.geometry = build_ngg_layer_forward_geometry(published);
    run.fragment = ngg_param_fragment(0, false);
    run.export_words = *exports;
    run.vertex_count = ngg_raster_commit_vertex_count(config, 32);
    run.width = run.height = kSize;
    run.layers = 32;
    run.shader_output_layer = route == NggLayerRoute::ShaderOutputLayer;
    const auto result = run_ngg_raster(run);
    if (!result || result->counters != std::array<uint32_t, 3>{}) return {};
    return result->pixels;
}

// A fragment stage writing the constant (v, v, v, v) to color 0.
std::vector<uint32_t> constant_fragment(float value) {
    // ids: 1 void, 2 fn, 3 f32, 4 v4f, 5 ptr out v4, 6 output, 7 main, 8 label, 9 c, 10 vec.
    std::vector<uint32_t> m = {0x07230203u, 0x00010300u, 0u, 11u, 0u};
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> operands) {
        m.push_back(static_cast<uint32_t>(operands.size() + 1u) << 16 | code);
        m.insert(m.end(), operands);
    };
    op(17, {1});   // OpCapability Shader
    op(14, {0, 1});   // OpMemoryModel Logical GLSL450
    op(15, {4, 7, 0x6e69616du, 0u, 6});   // OpEntryPoint Fragment %main "main" %out
    op(16, {7, 7});   // OpExecutionMode OriginUpperLeft
    op(71, {6, 30, 0});   // OpDecorate %out Location 0
    op(19, {1});   // OpTypeVoid
    op(33, {2, 1});   // OpTypeFunction
    op(22, {3, 32});   // OpTypeFloat 32
    op(23, {4, 3, 4});   // OpTypeVector
    op(32, {5, 3, 4});   // OpTypePointer Output
    op(43, {3, 9, std::bit_cast<uint32_t>(value)});   // OpConstant
    op(44, {4, 10, 9, 9, 9, 9});   // OpConstantComposite
    op(59, {5, 6, 3});   // OpVariable Output
    op(54, {1, 7, 0, 2});   // OpFunction
    op(248, {8});   // OpLabel
    op(62, {6, 10});   // OpStore
    op(253, {});   // OpReturn
    op(56, {});   // OpFunctionEnd
    return m;
}

std::vector<uint32_t> full_screen_vertex() {
#include "../tools/boot_trace/refvs.inc"
    return {std::begin(kRefVs), std::end(kRefVs)};
}

// ---- Description ----------------------------------------------------------------------------------

TEST(NggSubgroupBackend, DescriptionGroupsWaveCountsAndOrdersRuns) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    std::string why;
    const auto lut = kena_draw(*ctx, 4, 32, 32, &why);
    ASSERT_TRUE(lut) << why;
    ASSERT_EQ(lut->groups.size(), 1u);
    EXPECT_EQ(lut->groups[0].waves, 1u);
    EXPECT_EQ(lut->groups[0].blocks, 32u);
    EXPECT_EQ(lut->groups[0].launch_words.size(), 32u * 64u * kNggLaunchWordsPerLane);
    EXPECT_EQ(lut->runs.size(), 1u);
    // Binding 2 (the direct V# at s8) is declared but every load goes through the fetch-PC
    // entries: only ACCESSED bindings are guest bindings, as the frontend only builds those.
    EXPECT_EQ(lut->guest_bindings, (std::vector<uint32_t>{3, 4, 5, 6, 7, 8, 9}));

    const auto strip = kena_draw(*ctx, 76, 2, 2, &why);
    ASSERT_TRUE(strip) << why;
    ASSERT_EQ(strip->groups.size(), 2u) << "a W=3 and a W=1 group";
    EXPECT_LT(strip->groups[0].waves, strip->groups[1].waves) << "ascending: the dispatch order";
    ASSERT_EQ(strip->runs.size(), 4u);
    // Plan order alternates the groups, and each group's blocks are drawn in order.
    EXPECT_NE(strip->runs[0].group, strip->runs[1].group);
    EXPECT_EQ(strip->runs[0].group, strip->runs[2].group);
    EXPECT_EQ(strip->runs[0].first_block, 0u);
    EXPECT_EQ(strip->runs[2].first_block, 1u);
    EXPECT_EQ(strip->runs[3].first_block, 1u);

    NggSubgroupDrawRequest bad;
    EXPECT_FALSE(build_ngg_subgroup_draw(bad, &why));
    EXPECT_NE(why.find("reason=ngg-draw-plan"), std::string::npos) << why;
}

// The builder's own refusals, each beside an accepted twin.
TEST(NggSubgroupBackend, DescriptionRefusals) {
    const KenaInputs& in = kena_inputs();
    NggSubgroupDrawRequest request;
    request.linked_code = in.linked.data();
    request.dwords = in.linked.size();
    request.resources = &in.table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.limits = ngg::kena_limits();
    request.shape.topology = NggInputTopology::TriangleStrip;
    request.shape.vertex_count = 4;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = 32;
    request.raster.route = NggLayerRoute::ForwardingGeometry;
    request.push_constants.assign(ngg::kKenaUserSgprs, 0u);
    std::string why;
    ASSERT_TRUE(build_ngg_subgroup_draw(request, &why)) << why;

    NggSubgroupDrawRequest words = request;
    words.push_constants.pop_back();
    EXPECT_FALSE(build_ngg_subgroup_draw(words, &why));
    EXPECT_EQ(why, "reason=ngg-draw-push-constants");

    NggSubgroupDrawRequest lines = request;
    lines.raster.topology = NggOutputTopology::LineList;
    lines.raster.route = NggLayerRoute::InterpolationGeometry;
    lines.interpolation_geometry = [](const NggRasterCommitInterface&) {
        return std::vector<uint32_t>{1u};
    };
    EXPECT_FALSE(build_ngg_subgroup_draw(lines, &why));
    EXPECT_EQ(why, "reason=ngg-interpolation-geometry-needs-triangles");
    // No layer read, but the pixel stage still needs the interpolation stage: the raster commit
    // itself has no route to refuse here, so the builder must.
    lines.raster.layer_from_pos1 = false;
    lines.raster.route = NggLayerRoute::None;
    EXPECT_FALSE(build_ngg_subgroup_draw(lines, &why));
    EXPECT_EQ(why, "reason=ngg-interpolation-geometry-needs-triangles");

    NggSubgroupDrawRequest no_stage = request;
    no_stage.raster.route = NggLayerRoute::InterpolationGeometry;
    EXPECT_FALSE(build_ngg_subgroup_draw(no_stage, &why)) << "no interpolation stage supplied";
    EXPECT_EQ(why, "reason=ngg-draw-geometry-unavailable");
}

// Plain storage buffers in set 0 are the only guest resources the backend binds for the shell.
TEST(NggSubgroupBackend, ShellGuestBindingReflection) {
    const auto module = [](std::initializer_list<std::vector<uint32_t>> instructions) {
        std::vector<uint32_t> m = {0x07230203u, 0x00010300u, 0u, 32u, 0u};
        for (const auto& instruction : instructions)
            m.insert(m.end(), instruction.begin(), instruction.end());
        return m;
    };
    const auto op = [](uint32_t code, std::initializer_list<uint32_t> operands) {
        std::vector<uint32_t> words = {static_cast<uint32_t>(operands.size() + 1u) << 16 | code};
        words.insert(words.end(), operands);
        return words;
    };
    // %3 = struct { uint }, %4 = StorageBuffer pointer to it, %5 = an array of it, %6 its pointer.
    const auto buffer = [&](uint32_t set, uint32_t binding, bool array, uint32_t storage) {
        return module({op(71, {10, 34, set}), op(71, {10, 33, binding}), op(21, {2, 32, 0}),
                       op(30, {3, 2}), op(43, {2, 7, 2}), op(28, {5, 3, 7}),
                       op(32, {4, storage, 3}), op(32, {6, storage, 5}),
                       op(59, {array ? 6u : 4u, 10, storage})});
    };
    std::vector<uint32_t> bindings;
    ASSERT_TRUE(ngg_shell_guest_bindings(buffer(0, 5, false, 12), &bindings));
    EXPECT_EQ(bindings, std::vector<uint32_t>{5});
    EXPECT_TRUE(ngg_shell_guest_bindings(buffer(2, 1, false, 12), &bindings)) << "the shell's set";
    EXPECT_TRUE(bindings.empty());
    EXPECT_FALSE(ngg_shell_guest_bindings(buffer(1, 5, false, 12), nullptr)) << "set 1";
    EXPECT_FALSE(ngg_shell_guest_bindings(buffer(0, 5, true, 12), nullptr)) << "descriptor array";
    EXPECT_FALSE(ngg_shell_guest_bindings(buffer(0, 5, false, 2), nullptr)) << "uniform block";
    EXPECT_FALSE(ngg_shell_guest_bindings(module({op(25, {9, 2, 1, 0, 0, 0, 1, 0})}), nullptr))
        << "an image";
}

// ---- Kena end to end ------------------------------------------------------------------------------

TEST(NggSubgroupBackend, KenaLutMatchesTheOfflineResultByteForByte) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    const NggLayerRoute route = backend_route(*ctx);
    if (route == NggLayerRoute::None) GTEST_SKIP() << "no layer route on this device";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 32, 32, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    const BackendDraw draw = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    const BackendColorTarget target = volume_target(0x4e4747340001ull, 32);
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    const StatsSnapshot after = stats_now();
    ASSERT_EQ(bytes.size(), 32u * kSize * kSize * 16u);
    EXPECT_TRUE(lut_layers(bytes, 32, 32));
    EXPECT_EQ(after.draws - before.draws, 1u);
    EXPECT_EQ(after.dispatches - before.dispatches, 1u) << "one wave-count group";
    EXPECT_EQ(after.completed - before.completed, 1u) << "read once the direct batch completed";
    EXPECT_EQ(after.invalid - before.invalid, 0u);
    EXPECT_EQ(after.connectivity - before.connectivity, 0u);
    EXPECT_EQ(after.culled - before.culled, 0u);

    const std::vector<float> reference = offline_kena_lut(route);
    ASSERT_FALSE(reference.empty()) << "the P3 offline reference did not run";
    ASSERT_EQ(reference.size() * 4u, bytes.size());
    EXPECT_EQ(std::memcmp(reference.data(), bytes.data(), bytes.size()), 0)
        << "the backend's 32-layer target differs from the P3 offline result";
}

// ---- Two wave-count groups --------------------------------------------------------------------------

// A strip of horizontal bands covering the whole screen: rows y_k, k = 0..37, at x = -1 and +1.
std::vector<std::array<float, 2>> band_strip() {
    std::vector<std::array<float, 2>> positions;
    for (uint32_t row = 0; row < 38; ++row) {
        const float y = -1.0f + 2.0f * static_cast<float>(row) / 37.0f;
        positions.push_back({-1.0f, y});
        positions.push_back({1.0f, y});
    }
    return positions;
}

TEST(NggSubgroupBackend, TwoWaveCountGroupsInOneDrawCoverEveryBand) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 76, 2, 2, &why);
    ASSERT_TRUE(ngg) << why;
    ASSERT_EQ(ngg->groups.size(), 2u);
    const ResolvedPipelineState state = flipped_state();
    const BackendDraw draw = ngg_backend_draw(ngg, ngg::vertex_records(band_strip()), &state);
    const BackendColorTarget target = volume_target(0x4e4747340002ull, 2);
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    const StatsSnapshot after = stats_now();
    // 74 thin triangles per layer interpolate PARAM less exactly than the LUT's two; coverage and
    // the value are what matter (a missing band reads -1, a misplaced one a wrong row).
    EXPECT_TRUE(lut_layers(bytes, 2, 2, 1e-3f));
    EXPECT_EQ(after.dispatches - before.dispatches, 2u) << "one dispatch per wave count";
    EXPECT_EQ(after.invalid - before.invalid, 0u);
    EXPECT_EQ(after.connectivity - before.connectivity, 0u);
    EXPECT_EQ(after.culled - before.culled, 0u);
}

// ---- Segment ordering -------------------------------------------------------------------------------

// A (all channels 0.25, layer 0), then the Kena LUT (every layer), then B (blue and alpha only,
// 0.75, layer 0). Layer 0 must read (PARAM.x, PARAM.y, 0.75, 0.75): R and G from the NGG draw over
// A, B and A from B over the NGG draw. Every other layer is the LUT alone.
TEST(NggSubgroupBackend, DrawBetweenOrdinaryDrawsKeepsSubmissionOrder) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    ResolvedPipelineState blue_alpha = flipped_state();
    blue_alpha.color_write_mask = 0xC;
    BackendDraw a;
    a.vs = full_screen_vertex();
    a.fs = constant_fragment(0.25f);
    BackendDraw b = a;
    b.fs = constant_fragment(0.75f);
    b.ps = &blue_alpha;
    std::vector<BackendDraw> draws = {
        a, ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state), b};
    const BackendColorTarget target = volume_target(0x4e4747340003ull, 4);
    const auto bytes = render_draws_rgba(draws, kSize, kSize, nullptr, kClear, true, &target);
    ASSERT_EQ(bytes.size(), 4u * kSize * kSize * 16u);
    EXPECT_EQ(backend_color_target_stats().writes, 3u) << "the NGG draw is a segment of its own";
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            const float* p = texel(bytes, 0, x, y);
            const float sx = (static_cast<float>(x) + 0.5f) / kSize;
            const float sy = (static_cast<float>(y) + 0.5f) / kSize;
            ASSERT_NEAR(p[0], sx, 1e-4f)
                << "R: the NGG draw must follow A (" << x << "," << y << ")";
            ASSERT_NEAR(p[1], sy, 1e-4f)
                << "G: the NGG draw must follow A (" << x << "," << y << ")";
            ASSERT_EQ(p[2], 0.75f) << "B: B must follow the NGG draw (" << x << "," << y << ")";
            ASSERT_EQ(p[3], 0.75f) << "A: B must follow the NGG draw (" << x << "," << y << ")";
        }
    for (uint32_t layer = 1; layer < 4; ++layer)
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float* p = texel(bytes, layer, x, y);
                ASSERT_NEAR(p[0], (static_cast<float>(x) + 0.5f) / kSize, 1e-4f) << layer;
                ASSERT_NEAR(p[1], (static_cast<float>(y) + 0.5f) / kSize, 1e-4f) << layer;
            }
}

// ---- Zero-fill -------------------------------------------------------------------------------------

// Frame 2 reuses frame 1's export slice. Without the fill, every block would carry frame 1's
// GS_ALLOC_REQ count plus its own (2) and be refused as invalid: nothing would draw.
TEST(NggSubgroupBackend, SecondFrameReusesTheSliceAndSeesNoStaleExport) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto first = kena_draw(*ctx, 4, 32, 32, &why);
    const auto second = kena_draw(*ctx, 4, 16, 32, &why);
    ASSERT_TRUE(first && second) << why;
    const ResolvedPipelineState state = flipped_state();
    const auto records = ngg::vertex_records(ngg::kLutQuad);
    const BackendColorTarget target1 = volume_target(0x4e4747340004ull, 32);
    const BackendColorTarget target2 = volume_target(0x4e4747340005ull, 32);
    const auto frame1 = render_draws_rgba({ngg_backend_draw(first, records, &state)}, kSize, kSize,
                                          nullptr, kClear, true, &target1);
    EXPECT_TRUE(lut_layers(frame1, 32, 32));
    auto& stats = ngg_subgroup_backend_stats();
    const uint64_t buffer1 = stats.last_export_buffer.load();
    const uint64_t offset1 = stats.last_export_offset.load();
    const size_t chunks = NggScratchRing::instance().chunk_count();
    const StatsSnapshot before = stats_now();
    const auto frame2 = render_draws_rgba({ngg_backend_draw(second, records, &state)}, kSize, kSize,
                                          nullptr, kClear, true, &target2);
    const StatsSnapshot after = stats_now();
    EXPECT_EQ(stats.last_export_buffer.load(), buffer1) << "the same export slice is reused";
    EXPECT_EQ(stats.last_export_offset.load(), offset1);
    EXPECT_EQ(NggScratchRing::instance().chunk_count(), chunks) << "no new scratch per frame";
    EXPECT_TRUE(lut_layers(frame2, 32, 16)) << "16 instances: layers 0..15 only";
    EXPECT_EQ(after.invalid - before.invalid, 0u) << "a stale GS_ALLOC_REQ count leaked";
}

// ---- Lazy counters ---------------------------------------------------------------------------------

// 32 instances into a 16-slice target: instances 16..31 name layers out of range and are culled
// and counted. The count is read only when the batch completes: recording into an open batch reads
// nothing; completing it reads the count.
TEST(NggSubgroupBackend, ViolationCountersAreReadOnlyWhenTheBatchCompletes) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    if (!ngg_backend_counts_violations(*ctx)) GTEST_SKIP() << "no vertexPipelineStoresAndAtomics";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 32, 16, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    const BackendDraw draw = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    const BackendColorTarget open_target = volume_target(0x4e4747340006ull, 16, false);
    BackendSubmissionBatch batch;
    const StatsSnapshot before = stats_now();
    (void)render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &open_target, nullptr,
                            nullptr, nullptr, &batch, false);
    const StatsSnapshot recorded = stats_now();
    EXPECT_TRUE(batch.pending()) << "the batch stays open: nothing was submitted or waited on";
    EXPECT_EQ(recorded.draws - before.draws, 1u);
    EXPECT_EQ(recorded.completed - before.completed, 0u) << "read before completion";
    EXPECT_EQ(recorded.culled - before.culled, 0u);
    const auto result = batch.submit_and_wait(ctx->dev, ctx->queue, false);
    ASSERT_EQ(result.submit_result, VK_SUCCESS);
    ASSERT_EQ(result.wait_result, VK_SUCCESS);
    batch.complete();
    const StatsSnapshot done = stats_now();
    EXPECT_EQ(done.completed - before.completed, 1u);
    EXPECT_GE(done.culled - before.culled, 32u) << "16 out-of-range layers x 2 triangles";
    EXPECT_EQ(done.connectivity - before.connectivity, 0u);
    EXPECT_EQ(done.invalid - before.invalid, 0u);

    // The same draw read back immediately: layers 0..15 complete, nothing drawn past the target.
    const BackendColorTarget target = volume_target(0x4e4747340007ull, 16);
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    EXPECT_TRUE(lut_layers(bytes, 16, 16));
}

// ---- P5: the backend half of live admission ---------------------------------------------------------

// The forwarding geometry route through the real backend (RADV picks ShaderOutputLayer, so the
// tests above never assemble a pipeline with the GS): the same bytes as the P3 offline result.
TEST(NggSubgroupBackend, ForwardingGeometryRouteMatchesTheOfflineResult) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (!ctx->geometry_shader_enabled) GTEST_SKIP() << "no geometry shaders";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 32, 32, &why, NggLayerRoute::ForwardingGeometry);
    ASSERT_TRUE(ngg) << why;
    ASSERT_EQ(ngg->route, NggLayerRoute::ForwardingGeometry);
    ASSERT_FALSE(ngg->groups[0].stages->raster_geometry.empty());
    const ResolvedPipelineState state = flipped_state();
    const BackendDraw draw = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    const BackendColorTarget target = volume_target(0x4e4747340010ull, 32);
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    EXPECT_EQ(stats_now().draws - before.draws, 1u);
    ASSERT_EQ(bytes.size(), 32u * kSize * kSize * 16u);
    EXPECT_TRUE(lut_layers(bytes, 32, 32));
    const std::vector<float> reference = offline_kena_lut(NggLayerRoute::ForwardingGeometry);
    ASSERT_EQ(reference.size() * 4u, bytes.size());
    EXPECT_EQ(std::memcmp(reference.data(), bytes.data(), bytes.size()), 0);
}

// A draw the device cannot run is dropped before render_draws_rgba splits the batch, counted under
// backend/ngg-subgroup, and the ordinary draws around it render in one pass.
TEST(NggSubgroupBackend, DeviceRefusedDrawIsDroppedBeforeTheSplitAndCounted) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    auto refused = std::make_shared<NggSubgroupDraw>(*ngg);
    refused->lds_bytes = 1u << 30;   // more workgroup memory than any device has
    ASSERT_STREQ(ngg_device_refusal(*refused, ngg_host_capabilities(*ctx)),
                 "ngg-backend-lds-limit");
    const ResolvedPipelineState state = flipped_state();
    ResolvedPipelineState blue_alpha = flipped_state();
    blue_alpha.color_write_mask = 0xC;
    BackendDraw a;
    a.vs = full_screen_vertex();
    a.fs = constant_fragment(0.25f);
    BackendDraw b = a;
    b.fs = constant_fragment(0.75f);
    b.ps = &blue_alpha;
    const std::vector<BackendDraw> draws = {
        a, ngg_backend_draw(refused, ngg::vertex_records(ngg::kLutQuad), &state), b};
    const BackendColorTarget target = volume_target(0x4e4747340011ull, 4);
    namespace perf = prosper::diagnostics::perf;
    const size_t reason = static_cast<size_t>(perf::DropReason::BackendNggSubgroup);
    const uint64_t ledger_before = perf::ledger().drop_reasons[reason].load();
    const uint64_t refused_before = ngg_subgroup_backend_stats().refused.load();
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba(draws, kSize, kSize, nullptr, kClear, true, &target);
    EXPECT_EQ(perf::ledger().drop_reasons[reason].load() - ledger_before, 1u);
    EXPECT_EQ(ngg_subgroup_backend_stats().refused.load() - refused_before, 1u);
    EXPECT_EQ(stats_now().draws - before.draws, 0u) << "no prelude recorded";
    EXPECT_EQ(backend_color_target_stats().writes, 1u) << "A and B share one pass: no split";
    ASSERT_EQ(bytes.size(), 4u * kSize * kSize * 16u);
    const float* p = texel(bytes, 0, 3, 5);
    EXPECT_EQ(p[0], 0.25f);
    EXPECT_EQ(p[1], 0.25f);
    EXPECT_EQ(p[2], 0.75f);
    EXPECT_EQ(p[3], 0.75f);
    EXPECT_EQ(texel(bytes, 1, 3, 5)[0], -1.0f) << "the refused LUT draw drew nothing";
}

// A refusal the device check cannot see (set 2 taken) is still asked before the split, so the
// NGG draw is dropped from the call and the call's clear survives (#4634 N5): before, the
// expansion refused the NGG segment after the split and the clear went with it.
TEST(NggSubgroupBackend, StructureRefusalBeforeTheSplitKeepsTheClear) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    BackendDraw lut = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    FrameBufferResource taken;
    taken.set = kNggRasterDescriptorSet;
    taken.binding = 1;
    taken.dwords = {0u};
    lut.B.push_back(taken);
    lut.resource_order.push_back(0x80000000u | static_cast<uint32_t>(lut.B.size() - 1u));
    EXPECT_STREQ(ngg_backend_draw_structure_refusal(lut), "ngg-backend-set2-taken");
    ResolvedPipelineState blue_alpha = flipped_state();
    blue_alpha.color_write_mask = 0xC;
    BackendDraw b;
    b.vs = full_screen_vertex();
    b.fs = constant_fragment(0.75f);
    b.ps = &blue_alpha;
    const BackendColorTarget target = volume_target(0x4e4747340013ull, 4);
    const auto bytes = render_draws_rgba({lut, b}, kSize, kSize, nullptr, kClear, true, &target);
    ASSERT_EQ(bytes.size(), 4u * kSize * kSize * 16u);
    const float* p = texel(bytes, 0, 3, 5);
    EXPECT_EQ(p[0], -1.0f) << "the call's clear survived the dropped NGG draw";
    EXPECT_EQ(p[2], 0.75f) << "B drew";
}

// A split this call cannot make safely: transient depth (persist_depth_stencil false, the
// PROSPER_DUMP_DRAWSTEPS diagnostic's call) and, separately, no persistent colour target (each
// split would read the pass back with a CPU wait). A lone NGG draw needs no split and runs.
TEST(NggSubgroupBackend, UnsafeSplitsDropTheNggDraw) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    BackendDraw a;
    a.vs = full_screen_vertex();
    a.fs = constant_fragment(0.25f);
    const BackendDraw lut = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    std::vector<BackendDraw> kept;

    const BackendColorTarget target = volume_target(0x4e4747340012ull, 4);
    const std::vector<BackendDraw> pair = {a, lut};
    const auto host = ngg_host_capabilities(*ctx);
    EXPECT_STREQ(ngg_backend_draw_refusal(lut, host, 2, false, &target),
                 "ngg-backend-transient-depth-split");
    EXPECT_STREQ(ngg_backend_draw_refusal(lut, host, 2, true, nullptr),
                 "ngg-backend-readback-split");
    EXPECT_EQ(ngg_backend_draw_refusal(lut, host, 2, true, &target), nullptr);
    EXPECT_EQ(ngg_backend_draw_refusal(lut, host, 1, false, nullptr), nullptr) << "no split";
    EXPECT_EQ(ngg_backend_draw_refusal(a, host, 2, false, nullptr), nullptr) << "not NGG";
    // An MRT call carries slots 1+ across a split by readback whatever their identities.
    EXPECT_STREQ(ngg_backend_draw_refusal(lut, host, 2, true, &target, 2),
                 "ngg-backend-readback-split");
    BackendMrtOutputs mrt;
    mrt.color_count = 3;
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, &target, &mrt, nullptr, kept).size(), 1u);
    std::vector<uint8_t> out1;
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, &target, nullptr, &out1, kept).size(), 1u);
    mrt.color_count = 1;
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, &target, &mrt, nullptr, kept).data(),
              pair.data());
    EXPECT_EQ(ngg_admit_backend_draws(pair, false, &target, nullptr, nullptr, kept).size(), 1u)
        << "transient depth across a split";
    ASSERT_EQ(kept.size(), 1u);
    EXPECT_FALSE(kept[0].ngg_subgroup);
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, nullptr, nullptr, nullptr, kept).size(), 1u)
        << "no persistent colour target: a readback per split";
    BackendColorTarget unnamed = target;
    unnamed.persistent_id = 0;
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, &unnamed, nullptr, nullptr, kept).size(), 1u);
    EXPECT_EQ(ngg_admit_backend_draws(pair, true, &target, nullptr, nullptr, kept).data(),
              pair.data())
        << "persistent depth and colour: nothing dropped, the caller's vector itself";

    const std::vector<BackendDraw> lone = {lut};
    EXPECT_EQ(ngg_admit_backend_draws(lone, false, &target, nullptr, nullptr, kept).data(),
              lone.data())
        << "a lone NGG draw is not split";
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba(lone, kSize, kSize, nullptr, kClear, false, &target);
    EXPECT_EQ(stats_now().draws - before.draws, 1u);
    EXPECT_TRUE(lut_layers(bytes, 4, 4));
    const StatsSnapshot dropped = stats_now();
    (void)render_draws_rgba(pair, kSize, kSize, nullptr, kClear, false, &target);
    EXPECT_EQ(stats_now().draws - dropped.draws, 0u) << "the split under transient depth";
}

// #4643: a merged-NGG draw in a two-target VOLUME pass splits with both slots on the GPU. Each slot
// is a retained volume -- it has no transient fallback -- so the later segment LOADs it and no
// segment reads a slot back (CLAUDE.md P1); the same pass with a 2D slot 1 keeps the refusal.
TEST(NggSubgroupBackend, VolumeMrtSplitKeepsTheNggDrawAndBothSlots) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    BackendDraw a;
    a.vs = full_screen_vertex();
    a.fs = constant_fragment(0.25f);
    const BackendDraw lut = ngg_backend_draw(ngg, ngg::vertex_records(ngg::kLutQuad), &state);
    BackendColorTarget target = volume_target(0x4e4747340031ull, 4, false);
    target.persistent_id1 = 0x4e4747340032ull;
    target.load_existing1 = false;
    target.readback1 = false;
    target.format1 = target.format;
    target.volume_slots[1] = {4, 0, 4, 0};
    const auto host = ngg_host_capabilities(*ctx);
    EXPECT_EQ(ngg_backend_draw_refusal(lut, host, 2, true, &target, 2), nullptr);
    BackendColorTarget flat = target;
    flat.volume_slots[1] = {};
    EXPECT_STREQ(ngg_backend_draw_refusal(lut, host, 2, true, &flat, 2),
                 "ngg-backend-readback-split")
        << "a 2D slot 1 may be transient, so it would be carried by readback";

    const StatsSnapshot before = stats_now();
    BackendMrtOutputs mrt;
    mrt.color_count = 2;
    // want_color_readback=true, as the live renderer passes it: only the per-slot flags keep
    // a slot from being copied back, so this arm sees a split that reads slot 1 back.
    (void)render_draws_rgba({a, lut}, kSize, kSize, nullptr, kClear, true, &target, nullptr, kClear,
                            nullptr, nullptr, true, &mrt, true);
    // What catches a split that reads slot 1 back: its bytes would return as the next segment's
    // slot-1 seed, and a seeded volume slot is refused (volume-seeded), so the NGG segment would
    // record nothing and these two counts would fall.
    EXPECT_EQ(stats_now().draws - before.draws, 1u) << "the NGG draw ran in its own segment";
    EXPECT_EQ(backend_color_target_stats().writes, 2u) << "two segments, both recorded";
    EXPECT_EQ(backend_color_target_stats().retained_slots, 0x3u) << "both slots retained";
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(readback_persistent_color_target(target.persistent_id, kSize, kSize, target.format,
                                                 bytes, error, 4))
        << error;
    EXPECT_TRUE(lut_layers(bytes, 4, 4));
    // Slot 1: cleared by the first segment, LOADed (not cleared again, not seeded) by the second.
    ASSERT_TRUE(readback_persistent_color_target(target.persistent_id1, kSize, kSize,
                                                 target.format1, bytes, error, 4))
        << error;
    EXPECT_TRUE(lut_layers(bytes, 4, 0)) << "slot 1 holds its clear in every slice";
}

// Set 0 rides on the FIRST run draw only: it alone feeds the shell's guest set.
TEST(NggSubgroupBackend, OnlyTheFirstRunCarriesSetZero) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 76, 2, 2, &why);
    ASSERT_TRUE(ngg) << why;
    ASSERT_EQ(ngg->runs.size(), 4u);
    const ResolvedPipelineState state = flipped_state();
    const std::vector<BackendDraw> in = {
        ngg_backend_draw(ngg, ngg::vertex_records(band_strip()), &state)};
    std::vector<BackendDraw> out;
    std::shared_ptr<NggSubgroupBackendBatch> batch;
    std::string refusal;
    ASSERT_TRUE(NggSubgroupBackendBatch::expand(*ctx, in, out, batch, refusal)) << refusal;
    ASSERT_EQ(out.size(), 4u);
    const auto set_count = [](const BackendDraw& d, uint32_t set) {
        size_t n = 0;
        for (const auto& r : d.R) n += r.set == set;
        for (const auto& r : d.B) n += r.set == set;
        return n;
    };
    EXPECT_EQ(set_count(out[0], 0), set_count(in[0], 0)) << "the first run keeps all of set 0";
    for (size_t i = 1; i < out.size(); ++i) {
        EXPECT_EQ(set_count(out[i], 0), 0u) << "run " << i;
        EXPECT_GE(set_count(out[i], kNggRasterDescriptorSet), 1u) << "run " << i;
        for (uint32_t token : out[i].resource_order)
            EXPECT_LT(token & 0x7fffffffu,
                      (token & 0x80000000u) ? out[i].B.size() : out[i].R.size());
    }
    EXPECT_EQ(out[0].vs_shared, ngg->groups[ngg->runs[0].group].stages->raster_vertex)
        << "the pass-through stage is shared, not copied";
}

// All or nothing: a prelude records only when armed and every run draw it owns is ready; when it
// cannot, record() turns every one of its ready runs off and counts each.
TEST(NggSubgroupBackend, APartialNggDrawRecordsNothing) {
    struct Fake {
        bool ok = true;
    };
    const std::vector<int> owner = {-1, 0, 0, 1};
    std::vector<Fake> draws(4);
    const auto ready = [&](bool armed, int prelude) {
        return NggSubgroupBackendBatch::prelude_ready(armed, prelude, owner,
                                                      std::span<Fake>(draws));
    };
    EXPECT_TRUE(ready(true, 0));
    EXPECT_TRUE(ready(true, 1));
    EXPECT_FALSE(ready(false, 0)) << "unarmed";
    EXPECT_FALSE(ready(true, 2)) << "a prelude with no run draw";
    draws[2].ok = false;
    EXPECT_FALSE(ready(true, 0)) << "one of its two runs failed setup";
    EXPECT_TRUE(ready(true, 1)) << "another prelude is unaffected";

    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_draw(*ctx, 76, 2, 2, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    const std::vector<BackendDraw> in = {
        ngg_backend_draw(ngg, ngg::vertex_records(band_strip()), &state)};
    std::vector<BackendDraw> out;
    std::shared_ptr<NggSubgroupBackendBatch> batch;
    std::string refusal;
    ASSERT_TRUE(NggSubgroupBackendBatch::expand(*ctx, in, out, batch, refusal)) << refusal;
    std::vector<Fake> runs(out.size());
    const uint64_t census =
        prosper::gpu::draw_disposition_census().dropped(prosper::gpu::DrawDrop::NggSubgroup);
    BackendSubmissionBatch submission;
    {
        // record()'s drops are in-pass: a pass scope (as render_draw_pass_rgba has) reports them on
        // this thread, instead of leaving them to charge the next pass here.
        prosper::gpu::DrawDispositionPassScope pass(runs.size());
        // Never captured, so never armed: nothing is recorded and every run is turned off.
        batch->record(VK_NULL_HANDLE, submission, std::span<Fake>(runs));
    }
    for (const Fake& run : runs) EXPECT_FALSE(run.ok);
    EXPECT_EQ(prosper::gpu::draw_disposition_census().dropped(prosper::gpu::DrawDrop::NggSubgroup) -
                  census,
              runs.size());
    EXPECT_FALSE(submission.pending());
}

// The shell pipeline cache: keyed by the hash computed when the stages were built, hit by an equal
// module in another allocation, and bounded.
TEST(NggSubgroupBackend, ShellPipelineCacheIsKeyedOnceAndBounded) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    std::string why;
    const auto ngg = kena_draw(*ctx, 4, 4, 4, &why);
    ASSERT_TRUE(ngg) << why;
    const NggSubgroupStages& stages = *ngg->groups[0].stages;
    const uint32_t push = static_cast<uint32_t>(ngg->push_constants.size());
    auto& stats = ngg_shell_pipeline_cache_stats();
    const auto first = ngg_shell_pipeline(*ctx, stages, ngg->guest_bindings, push, false);
    ASSERT_TRUE(first);
    const uint64_t creations = stats.creations, hits = stats.hits;
    EXPECT_EQ(ngg_shell_pipeline(*ctx, stages, ngg->guest_bindings, push, false), first);
    NggSubgroupStages copy = stages;
    copy.shell = std::make_shared<const std::vector<uint32_t>>(*stages.shell);
    EXPECT_EQ(ngg_shell_pipeline(*ctx, copy, ngg->guest_bindings, push, false), first)
        << "an equal module in another allocation";
    EXPECT_EQ(stats.creations, creations);
    EXPECT_EQ(stats.hits, hits + 2);

    // More distinct keys than the cache holds, from the portable variant alone: a device without
    // native Wave64 (CI's lavapipe) gets no second variant per key, so the count cannot rely on it.
    // The keys differ by shell hash, not by push size: a push range past the device limit is invalid
    // Vulkan (VUID-VkPushConstantRange-size-00298), which the validation scan reports.
    const uint64_t evictions = stats.evictions;
    const uint32_t keys = static_cast<uint32_t>(kNggShellPipelineCacheEntries) + 8u;
    for (uint32_t i = 1; i <= keys; ++i) {
        NggSubgroupStages other = stages;
        other.shell_hash = stages.shell_hash + i;
        (void)ngg_shell_pipeline(*ctx, other, ngg->guest_bindings, push, false);
    }
    EXPECT_LE(stats.entries, kNggShellPipelineCacheEntries);
    EXPECT_GT(stats.evictions, evictions);
    EXPECT_TRUE(first->pipeline) << "an evicted entry a holder still owns stays valid";
}

// ---- P6: indexed draws ------------------------------------------------------------------------------
//
// Kena's LUT producer driven by an INDEX BUFFER instead of its strip: the shapes Kena submits past
// New Game (#3135 P6) are small instanced triangle lists into a 64-slice volume. Eight vertex
// records; the quad's corners sit at the scattered indices 6 (1,-1), 1 (1,1), 4 (-1,-1) and 3 (-1,1).
// The other records are decoys placed so that reading index POSITIONS instead of index VALUES draws
// the complementary half: records 0, 1, 2 are the upper-left triangle, and 3, 4, 5 a degenerate one.
// Each record's PARAM is its own normalized screen position, so a covered pixel must read its own
// position whichever records drew it -- coverage is what each arm discriminates.

const std::vector<std::array<float, 2>>& scattered_records() {
    static const std::vector<std::array<float, 2>> positions = {
        {-1, 1}, {1, 1}, {-1, -1}, {-1, 1}, {-1, -1}, {-1, 1}, {1, -1}, {-1, -1}};
    return positions;
}

const ShaderResourceTable& scattered_table() {
    static const ShaderResourceTable table = ngg::kena_resources(8);
    return table;
}

std::shared_ptr<const std::vector<uint32_t>> decoded(const std::vector<uint32_t>& values,
                                                     uint32_t element_bytes) {
    std::vector<unsigned char> bytes(values.size() * element_bytes);
    for (size_t i = 0; i < values.size(); ++i) {
        if (element_bytes == 2u) {
            const uint16_t narrow = static_cast<uint16_t>(values[i]);
            std::memcpy(bytes.data() + 2u * i, &narrow, 2u);
        } else {
            std::memcpy(bytes.data() + 4u * i, &values[i], 4u);
        }
    }
    return decode_ngg_draw_indices(bytes.data(), element_bytes,
                                   static_cast<uint32_t>(values.size()), {})
        .indices;
}

std::shared_ptr<const NggSubgroupDraw>
kena_indexed_draw(const RenderVkCtx& ctx, std::shared_ptr<const std::vector<uint32_t>> indices,
                  uint32_t instances, uint32_t slices, std::string* why) {
    NggDrawShape shape;
    shape.topology = NggInputTopology::TriangleList;
    shape.indices = std::move(indices);
    shape.vertex_count = shape.indices ? static_cast<uint32_t>(shape.indices->size()) : 0u;
    shape.instance_count = instances;
    return kena_shape_draw(ctx, shape, scattered_table(), slices, why);
}

// The quad as two triangles of scattered 16-bit and 32-bit indices, 5 instances into an 8-slice
// volume: layers 0..4 complete, 5..7 clear, and the two encodings give identical bytes. Without the
// index mapping the planner would run records 0..5 and only the upper-left half would draw.
TEST(NggSubgroupBackend, IndexedListDrawsItsIndexedVerticesOnEveryInstanceLayer) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    const std::vector<uint32_t> quad = {6, 1, 4, 4, 1, 3};
    const ResolvedPipelineState state = flipped_state();
    const auto records = ngg::vertex_records(scattered_records());
    std::vector<uint8_t> by_width[2];
    for (uint32_t w = 0; w < 2u; ++w) {
        std::string why;
        const auto ngg = kena_indexed_draw(*ctx, decoded(quad, w ? 4u : 2u), 5, 8, &why);
        ASSERT_TRUE(ngg) << why;
        ASSERT_EQ(ngg->plan.subgroups.size(), 5u);
        EXPECT_EQ(ngg->plan.subgroups[0].es_threads(), 4u) << "6 references, 4 ES lanes";
        const BackendColorTarget target = volume_target(0x4e4747340010ull + w, 8);
        const StatsSnapshot before = stats_now();
        by_width[w] = render_draws_rgba({ngg_backend_draw(ngg, records, &state)}, kSize, kSize,
                                        nullptr, kClear, true, &target);
        const StatsSnapshot after = stats_now();
        EXPECT_TRUE(lut_layers(by_width[w], 8, 5)) << (w ? "32-bit" : "16-bit") << " indices";
        EXPECT_EQ(after.invalid - before.invalid, 0u);
        EXPECT_EQ(after.connectivity - before.connectivity, 0u);
        EXPECT_EQ(after.culled - before.culled, 0u);
    }
    EXPECT_EQ(by_width[0], by_width[1]) << "16- and 32-bit encodings of one list";
}

// Which records a primitive reads is what decides coverage. One triangle, indices (6, 1, 4), is
// the lower-right half of the screen (clip y < x); the records at positions 0, 1, 2 are the
// upper-left half. Pixels on either side of the diagonal (one-pixel margin) are checked.
TEST(NggSubgroupBackend, IndexValuesDecideWhichHalfIsCovered) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    // Two slices, not one: the vertex stage writes gl_Layer, which a one-layer framebuffer leaves
    // undefined (the validation layer's Undefined-Value-Layer-Written). Slice 1 stays clear.
    const auto ngg = kena_indexed_draw(*ctx, decoded({6, 1, 4}, 2), 1, 2, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    const BackendColorTarget target = volume_target(0x4e4747340012ull, 2);
    const auto bytes =
        render_draws_rgba({ngg_backend_draw(ngg, ngg::vertex_records(scattered_records()), &state)},
                          kSize, kSize, nullptr, kClear, true, &target);
    ASSERT_EQ(bytes.size(), 2u * kSize * kSize * 16u);
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x)
            ASSERT_EQ(texel(bytes, 1, x, y)[0], -1.0f) << "slice 1 (" << x << "," << y << ")";
    uint32_t covered = 0, clear = 0;
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) / kSize;
            const float sy = (static_cast<float>(y) + 0.5f) / kSize;
            const float cx = 2.0f * sx - 1.0f, cy = 1.0f - 2.0f * sy;   // clip y = +1 on top
            const float* p = texel(bytes, 0, x, y);
            if (cy < cx - 0.13f) {
                EXPECT_NEAR(p[0], sx, 1e-4f) << "lower-right (" << x << "," << y << ")";
                EXPECT_NEAR(p[1], sy, 1e-4f) << "lower-right (" << x << "," << y << ")";
                ++covered;
            } else if (cy > cx + 0.13f) {
                EXPECT_EQ(p[0], -1.0f) << "upper-left must stay clear (" << x << "," << y << ")";
                ++clear;
            }
        }
    EXPECT_EQ(covered, 105u) << "x + y >= 17";
    EXPECT_EQ(clear, 105u) << "x + y <= 13";
}

// Slice routing per primitive: each primitive's layer is its instance plus the layer base the
// prolog reads from the constant buffer at binding 5 (word 0, pc 12). Base 2, 5 instances, into an
// 8-slice volume: layers 2..6 complete, 0, 1 and 7 clear.
TEST(NggSubgroupBackend, IndexedInstancesRouteToTheirOwnSlices) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    if (backend_route(*ctx) == NggLayerRoute::None) GTEST_SKIP() << "no layer route";
    std::string why;
    const auto ngg = kena_indexed_draw(*ctx, decoded({6, 1, 4, 4, 1, 3}, 2), 5, 8, &why);
    ASSERT_TRUE(ngg) << why;
    const ResolvedPipelineState state = flipped_state();
    BackendDraw draw = ngg_backend_draw(ngg, ngg::vertex_records(scattered_records()), &state);
    bool based = false;
    for (FrameBufferResource& buffer : draw.B)
        if (buffer.set == 0 && buffer.binding == 5) {
            buffer.dwords[0] = 2;
            based = true;
        }
    ASSERT_TRUE(based);
    const BackendColorTarget target = volume_target(0x4e4747340013ull, 8);
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    ASSERT_EQ(bytes.size(), 8u * kSize * kSize * 16u);
    for (uint32_t layer = 0; layer < 8u; ++layer) {
        const bool drawn = layer >= 2u && layer < 7u;
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                const float* p = texel(bytes, layer, x, y);
                if (drawn) {
                    ASSERT_NEAR(p[0], (static_cast<float>(x) + 0.5f) / kSize, 1e-4f) << layer;
                    ASSERT_NEAR(p[1], (static_cast<float>(y) + 0.5f) / kSize, 1e-4f) << layer;
                } else {
                    ASSERT_EQ(p[0], -1.0f) << "layer " << layer << " must stay clear";
                }
            }
    }
}

// ---- P7: NGG without a GS -----------------------------------------------------------------------------
//
// A primitive shader with no GS (VGT_SHADER_STAGES_EN 0x2000), the launch Kena's culling VS programs
// read: GS_ALLOC_REQ from s3's counts, PRIM = the three ES slots from v0/v1 (scaled by ITEMSIZE 4),
// POS0 from VertexID (v5) bit 0 -> x and bit 1 -> y (each -1 or +1), PARAM0 = (0.25, 0.5, 0, 1).
// Assembled with llvm-mc -mcpu=gfx1030; the same words as test_ngg_indexed_realize's kVsOnly.
const uint32_t kVsOnlyPrimitiveShader[] = {
    0xBEFE04C1u, 0x9394FF03u, 0x00040018u, 0xBF068014u, 0xBF840007u, 0x9395FF03u, 0x00080008u,
    0x8716FF03u, 0x000000FFu, 0x8F158C15u, 0x887C1615u, 0xBF900009u, 0xBE9703C1u, 0xD7650013u,
    0x00010017u, 0xD548000Au, 0x02390500u, 0xD548000Bu, 0x02392500u, 0xD548000Cu, 0x02390501u,
    0x3416168Au, 0x34181894u, 0xD7720009u, 0x0432170Au, 0xF8000941u, 0x00000009u, 0x361A0A81u,
    0x7E1A0D0Du, 0xD54B000Du, 0x03CDE90Du, 0xD548000Eu, 0x02050305u, 0x7E1C0D0Eu, 0xD54B000Eu,
    0x03CDE90Eu, 0x7E1E0280u, 0x7E2002F2u, 0xF80008CFu, 0x100F0E0Du, 0x7E2202FFu, 0x3E800000u,
    0x7E2402F0u, 0xF800020Fu, 0x100F1211u, 0xBF810000u,
};

// The same program also exporting POS1.z = 0 (layer 0) or 1 (layer 1), as Kena's culling VS programs
// export POS1.z under USE_VTX_RENDER_TARGET_INDX into 2D scene targets.
const uint32_t kVsOnlyLayer0[] = {
    0xBEFE04C1u, 0x9394FF03u, 0x00040018u, 0xBF068014u, 0xBF840007u, 0x9395FF03u, 0x00080008u,
    0x8716FF03u, 0x000000FFu, 0x8F158C15u, 0x887C1615u, 0xBF900009u, 0xBE9703C1u, 0xD7650013u,
    0x00010017u, 0xD548000Au, 0x02390500u, 0xD548000Bu, 0x02392500u, 0xD548000Cu, 0x02390501u,
    0x3416168Au, 0x34181894u, 0xD7720009u, 0x0432170Au, 0xF8000941u, 0x00000009u, 0x361A0A81u,
    0x7E1A0D0Du, 0xD54B000Du, 0x03CDE90Du, 0xD548000Eu, 0x02050305u, 0x7E1C0D0Eu, 0xD54B000Eu,
    0x03CDE90Eu, 0x7E1E0280u, 0x7E2002F2u, 0xF80000CFu, 0x100F0E0Du, 0x7E280280u, 0xF80008D4u,
    0x00140000u, 0x7E2202FFu, 0x3E800000u, 0x7E2402F0u, 0xF800020Fu, 0x100F1211u, 0xBF810000u,
};
const uint32_t kVsOnlyLayer1[] = {
    0xBEFE04C1u, 0x9394FF03u, 0x00040018u, 0xBF068014u, 0xBF840007u, 0x9395FF03u, 0x00080008u,
    0x8716FF03u, 0x000000FFu, 0x8F158C15u, 0x887C1615u, 0xBF900009u, 0xBE9703C1u, 0xD7650013u,
    0x00010017u, 0xD548000Au, 0x02390500u, 0xD548000Bu, 0x02392500u, 0xD548000Cu, 0x02390501u,
    0x3416168Au, 0x34181894u, 0xD7720009u, 0x0432170Au, 0xF8000941u, 0x00000009u, 0x361A0A81u,
    0x7E1A0D0Du, 0xD54B000Du, 0x03CDE90Du, 0xD548000Eu, 0x02050305u, 0x7E1C0D0Eu, 0xD54B000Eu,
    0x03CDE90Eu, 0x7E1E0280u, 0x7E2002F2u, 0xF80000CFu, 0x100F0E0Du, 0x7E280281u, 0xF80008D4u,
    0x00140000u, 0x7E2202FFu, 0x3E800000u, 0x7E2402F0u, 0xF800020Fu, 0x100F1211u, 0xBF810000u,
};

// Kena's culling-VS partition registers: onchip 0x10020040, GE_CNTL 0x8040, GE_MAX_OUTPUT 64,
// GS_MAX_VERT_OUT 0, ITEMSIZE 4, read as VS-only.
NggSubgroupLimits vs_only_limits() {
    NggSubgroupLimits l = decode_ngg_subgroup_limits(0x10020040u, 0x8040u, 0x40u, 0u, 4u);
    l.vs_only = true;
    return l;
}

std::shared_ptr<const NggSubgroupDraw>
vs_only_draw(std::vector<uint32_t> indices, std::string* why,
             std::span<const uint32_t> program = kVsOnlyPrimitiveShader,
             const RenderVkCtx* layered = nullptr) {
    static const ShaderResourceTable none;
    NggSubgroupDrawRequest request;
    request.linked_code = program.data();
    request.dwords = program.size();
    request.resources = &none;
    request.limits = vs_only_limits();
    request.shape.topology = NggInputTopology::TriangleList;
    request.shape.vertex_count = static_cast<uint32_t>(indices.size());
    request.shape.indices = std::make_shared<const std::vector<uint32_t>>(std::move(indices));
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.route = NggLayerRoute::None;
    request.raster.count_violations = false;
    if (layered) {   // the layer addresses a one-slice 2D target (#3135 P7 admission)
        request.raster.layer_from_pos1 = true;
        request.raster.layer_slices = 1;
        request.raster.route = NggLayerRoute::None;   // one slice: cull only, no gl_Layer
        request.raster.count_violations = ngg_backend_counts_violations(*layered);
    }
    request.diagnostic = {RecompileDiagnosticStage::Vertex, 0x512e920000ull};
    return build_ngg_subgroup_draw(request, why);
}

// One triangle, indices (1, 3, 2) = (+1,-1), (+1,+1), (-1,+1): the half of the screen with
// clip x + y > 0, i.e. pixel x > y. The pixels a one-pixel margin either side of the diagonal are
// exact: PARAM0 (0.25, 0.5, 0, 1) above, the clear below. The VS-only partition is what lets the
// draw exist at all: read as merged, GS_MAX_VERT_OUT 0 leaves no primitive limit and the plan is
// refused.
TEST(NggSubgroupBackend, VsOnlyPrimitiveShaderDrawsItsTriangleIntoA2DTarget) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    std::string why;
    const auto ngg = vs_only_draw({1, 3, 2}, &why);
    ASSERT_TRUE(ngg) << why;
    ASSERT_EQ(ngg->plan.subgroups.size(), 1u);
    EXPECT_EQ(ngg->plan.subgroups[0].es_vertex, (std::vector<uint32_t>{1, 3, 2}));
    const ResolvedPipelineState state = flipped_state();
    BackendDraw draw;
    draw.ngg_subgroup = ngg;
    draw.fs = ngg_param_fragment(0, false);
    draw.ps = &state;
    BackendColorTarget target;
    target.persistent_id = 0x4e4747340070ull;
    target.load_existing = false;
    target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    const StatsSnapshot before = stats_now();
    const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    const StatsSnapshot after = stats_now();
    ASSERT_EQ(bytes.size(), static_cast<size_t>(kSize) * kSize * 16u);
    EXPECT_EQ(after.draws - before.draws, 1u);
    EXPECT_EQ(after.invalid - before.invalid, 0u) << "the GS_ALLOC_REQ counts were accepted";
    EXPECT_EQ(after.connectivity - before.connectivity, 0u);
    uint32_t covered = 0, clear = 0;
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            const float* p = texel(bytes, 0, x, y);
            if (x >= y + 2u) {
                EXPECT_EQ(p[0], 0.25f) << "covered (" << x << "," << y << ")";
                EXPECT_EQ(p[1], 0.5f) << "covered (" << x << "," << y << ")";
                EXPECT_EQ(p[2], 0.0f) << "covered (" << x << "," << y << ")";
                EXPECT_EQ(p[3], 1.0f) << "covered (" << x << "," << y << ")";
                ++covered;
            } else if (y >= x + 2u) {
                EXPECT_EQ(p[0], -1.0f) << "clear (" << x << "," << y << ")";
                ++clear;
            }
        }
    EXPECT_EQ(covered, 105u);
    EXPECT_EQ(clear, 105u);

    NggSubgroupLimits merged = vs_only_limits();
    merged.vs_only = false;
    NggDrawShape triangle;
    triangle.topology = NggInputTopology::TriangleList;
    triangle.vertex_count = 3;
    EXPECT_EQ(plan_ngg_subgroups(triangle, merged).refusal, "ngg-limits-unusable")
        << "control: the same registers read as merged cannot be planned";
}

// The upper-right half (x >= y + 2) holds PARAM0 and the lower-left half (y >= x + 2) the clear.
::testing::AssertionResult upper_right_triangle(const std::vector<uint8_t>& bytes, bool drawn) {
    if (bytes.size() != static_cast<size_t>(kSize) * kSize * 16u)
        return ::testing::AssertionFailure() << "readback of " << bytes.size() << " bytes";
    for (uint32_t y = 0; y < kSize; ++y)
        for (uint32_t x = 0; x < kSize; ++x) {
            const float* p = texel(bytes, 0, x, y);
            const bool inside = x >= y + 2u;
            if (!inside && y < x + 2u) continue;   // the diagonal margin
            const bool ok = inside && drawn
                                ? p[0] == 0.25f && p[1] == 0.5f && p[2] == 0.0f && p[3] == 1.0f
                                : p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f && p[3] == -1.0f;
            if (!ok)
                return ::testing::AssertionFailure()
                       << "(" << x << "," << y << ") = (" << p[0] << "," << p[1] << "," << p[2]
                       << "," << p[3] << ")";
        }
    return ::testing::AssertionSuccess();
}

// Kena's culling VS programs export POS1.z into 2D targets. Admitted as a one-slice target: layer 0
// draws exactly the unlayered picture, and layer 1 -- a slice the view does not have -- is culled
// and counted, not drawn at slice 0.
TEST(NggSubgroupBackend, VsOnlyLayerAddressesAOneSliceTarget) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    const ResolvedPipelineState state = flipped_state();
    for (const bool second_layer : {false, true}) {
        std::string why;
        const auto ngg = vs_only_draw({1, 3, 2}, &why,
                                      second_layer ? std::span<const uint32_t>(kVsOnlyLayer1)
                                                   : std::span<const uint32_t>(kVsOnlyLayer0),
                                      ctx);
        ASSERT_TRUE(ngg) << why;
        BackendDraw draw;
        draw.ngg_subgroup = ngg;
        draw.fs = ngg_param_fragment(0, false);
        draw.ps = &state;
        BackendColorTarget target;
        target.persistent_id = 0x4e4747340071ull + (second_layer ? 1u : 0u);
        target.load_existing = false;
        target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        const StatsSnapshot before = stats_now();
        const auto bytes = render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
        const StatsSnapshot after = stats_now();
        EXPECT_TRUE(upper_right_triangle(bytes, !second_layer))
            << (second_layer ? "layer 1" : "layer 0");
        EXPECT_EQ(after.invalid - before.invalid, 0u);
        if (ngg_backend_counts_violations(*ctx))
            EXPECT_EQ(after.culled - before.culled, second_layer ? 1u : 0u)
                << "the out-of-range layer is counted";
    }
}

// ---- #4808: a Wave32 passthrough NGG VS --------------------------------------------------------------
//
// The shape of Yakuza Kiwami's NGG VS programs (VGT_SHADER_STAGES_EN 0x02402000: PRIMGEN_EN, GS_W32_EN,
// PRIMGEN_PASSTHRU_EN), synthetic: the wave index from s3[27:24], the subgroup's primitive and vertex
// counts from s2[30:22] and s2[20:12], an s_barrier, GS_ALLOC_REQ from wave 0 only, the subgroup
// thread id as mbcnt_lo(-1, wave * 32), `exp prim v0` (the launch's packed primitive, unmodified) on
// threads below the primitive count, and on threads below the vertex count POS0 from VertexID (v5)
// as kVsOnlyPrimitiveShader places it, PARAM0 = (0.25, 0.5, 0, 1). EXEC is written as
// s_mov_b32 exec_lo, -1 (Wave32). PARAM0.y is built the way Yakuza builds a descriptor word: VCC as
// scalar data, `s_bfe_u64 vcc, s[8:9], 1:32` of s[8:9] = 0x7e000000 set just before it (0.5f), read
// back by `s_or_b32 s21, vcc_lo, 0` -- after the branches, where only the local same-block proof
// (rdna2_emit_cfg.cpp) keeps VCC data. Assembled with llvm-mc -mcpu=gfx1030 (branch offsets by hand).
const uint32_t kWave32PassthroughVs[] = {
    0x9394FF03u, 0x00040018u, 0xBEFE03C1u, 0x9380FF02u, 0x00090016u, 0x9381FF02u, 0x0009000Cu,
    0xBF8A0000u, 0xBF071480u, 0xBF850004u, 0x8F158C00u, 0x887C1501u, 0xBF800000u, 0xBF900009u,
    0x8F158514u, 0xD7650001u, 0x00002AC1u, 0x7DA80200u, 0xBF880002u, 0xF8000941u, 0x00000000u,
    0xBF8CFF0Fu, 0xBEFE03C1u, 0x7DA80201u, 0xBF880018u, 0x36040A81u, 0x7E040D02u, 0xD54B0002u,
    0x03CDE902u, 0xD5480003u, 0x02050305u, 0x7E060D03u, 0xD54B0003u, 0x03CDE903u, 0x7E080280u,
    0x7E0C02F2u, 0xF80008CFu, 0x06040302u, 0x7E0E02FFu, 0x3E800000u, 0xBE8803FFu, 0x7E000000u,
    0xBE890380u, 0x94EAFF08u, 0x00200001u, 0x8815806Au, 0x7E100215u, 0xF800020Fu, 0x06040807u,
    0xBF810000u,
};

// 33 primitives: 32 degenerate (1, 1, 1) that fill wave 0, then (1, 3, 2) -- the upper-right
// triangle -- as thread 32, lane 0 of wave 1. It is drawn only if wave 1 knows its index (s3), the
// subgroup's primitive count reaches it (s2 = 33, summed over both waves' s3), and its v0 holds the
// packed passthrough primitive. Each control breaks exactly one of those and leaves the target clear.
TEST(NggSubgroupBackend, Wave32PassthroughVsDrawsTheTriangleItsSecondWaveExports) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    std::vector<uint32_t> indices;
    for (uint32_t k = 0; k < 32; ++k) indices.insert(indices.end(), {1u, 1u, 1u});
    indices.insert(indices.end(), {1u, 3u, 2u});
    static const ShaderResourceTable none;
    const auto build = [&](bool passthrough, uint32_t lanes, bool native, std::string* why) {
        NggSubgroupDrawRequest request;
        request.linked_code = kWave32PassthroughVs;
        request.dwords = std::size(kWave32PassthroughVs);
        request.resources = &none;
        request.limits = vs_only_limits();
        request.limits.wave_lanes = lanes;
        request.limits.passthrough = passthrough;
        request.shell.native_wave64 = native;
        request.shape.topology = NggInputTopology::TriangleList;
        request.shape.vertex_count = static_cast<uint32_t>(indices.size());
        request.shape.indices = std::make_shared<const std::vector<uint32_t>>(indices);
        request.raster.topology = NggOutputTopology::TriangleList;
        request.raster.route = NggLayerRoute::None;
        request.raster.count_violations = false;
        request.diagnostic = {RecompileDiagnosticStage::Vertex, 0x512e920100ull};
        return build_ngg_subgroup_draw(request, why);
    };
    const ResolvedPipelineState state = flipped_state();
    const auto render = [&](const std::shared_ptr<const NggSubgroupDraw>& ngg, uint64_t id) {
        BackendDraw draw;
        draw.ngg_subgroup = ngg;
        draw.fs = ngg_param_fragment(0, false);
        draw.ps = &state;
        BackendColorTarget target;
        target.persistent_id = id;
        target.load_existing = false;
        target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        return render_draws_rgba({draw}, kSize, kSize, nullptr, kClear, true, &target);
    };
    const bool native32 = ngg_host_capabilities(*ctx).native_wave32;
    for (const bool native : {false, true}) {
        if (native && !native32) continue;
        std::string why;
        const auto ngg = build(true, 32, native, &why);
        ASSERT_TRUE(ngg) << why;
        ASSERT_EQ(ngg->plan.subgroups.size(), 1u);
        EXPECT_EQ(ngg->plan.subgroups[0].waves, 2u) << "33 threads are two Wave32 waves";
        EXPECT_EQ(ngg->wave_lanes, 32u);
        const StatsSnapshot before = stats_now();
        const auto bytes = render(ngg, 0x4e4747320001ull + (native ? 1u : 0u));
        const StatsSnapshot after = stats_now();
        EXPECT_TRUE(upper_right_triangle(bytes, true)) << (native ? "native" : "portable");
        EXPECT_EQ(after.invalid - before.invalid, 0u) << "one GS_ALLOC_REQ, from wave 0";
        EXPECT_EQ(after.connectivity - before.connectivity, 0u);
    }
    // Controls. Read as Wave64, s_mov_b32 exec_lo, -1 leaves EXEC_HI undefined, and the program is
    // refused rather than run with half an EXEC.
    std::string why;
    EXPECT_FALSE(build(true, 64, false, &why)) << "control: the Wave64 reading of a Wave32 program";
    EXPECT_NE(why.find("ngg-abi-exec-read-before-write"), std::string::npos) << why;
    // Without passthrough, v0 holds ITEMSIZE-scaled offsets (0 | 4 << 16 for slots 0, 1) that the
    // program exports as a primitive naming threads 0, 256 and 0: no triangle survives.
    const auto offsets = build(false, 32, false, &why);
    ASSERT_TRUE(offsets) << why;
    EXPECT_TRUE(upper_right_triangle(render(offsets, 0x4e4747320004ull), false))
        << "control: offsets instead of the packed primitive";
}

// ---- Layered depth: a depth-only draw replayed per slice -----------------------------------------
//
// The VS-only program above with its layer and depth taken from the vertex: VertexID (v5) bits 0
// and 1 place the corner as before, layer = VertexID >> 2 is exported in POS1.z, and POS0.z =
// (layer + 1) / 8. Assembled with llvm-mc -mcpu=gfx1030.
const uint32_t kVsOnlyDepthLayers[] = {
    0xBEFE04C1u, 0x9394FF03u, 0x00040018u, 0xBF068014u, 0xBF840007u, 0x9395FF03u, 0x00080008u,
    0x8716FF03u, 0x000000FFu, 0x8F158C15u, 0x887C1615u, 0xBF900009u, 0xBE9703C1u, 0xD7650013u,
    0x00010017u, 0xD548000Au, 0x02390500u, 0xD548000Bu, 0x02392500u, 0xD548000Cu, 0x02390501u,
    0x3416168Au, 0x34181894u, 0xD7720009u, 0x0432170Au, 0xF8000941u, 0x00000009u, 0x361A0A81u,
    0x7E1A0D0Du, 0xD54B000Du, 0x03CDE90Du, 0xD548000Eu, 0x02050305u, 0x7E1C0D0Eu, 0xD54B000Eu,
    0x03CDE90Eu, 0x2C280A82u, 0x4A1E2881u, 0x7E1E0D0Fu, 0x101E1EFFu, 0x3E000000u, 0x7E2002F2u,
    0xF80000CFu, 0x100F0E0Du, 0xF80008D4u, 0x00140000u, 0x7E2202FFu, 0x3E800000u, 0x7E2402F0u,
    0xF800020Fu, 0x100F1211u, 0xBF810000u,
};

// One depth-only draw compiled to select its layer at draw time, as ngg_live_draw builds it. Its
// per-slice replays are this same description with BackendDraw::ngg_layer_select = k.
std::shared_ptr<const NggSubgroupDraw> depth_slice_draw(const RenderVkCtx& ctx,
                                                        std::vector<uint32_t> indices,
                                                        uint32_t slices, std::string* why) {
    static const ShaderResourceTable none;
    NggSubgroupDrawRequest request;
    request.linked_code = kVsOnlyDepthLayers;
    request.dwords = std::size(kVsOnlyDepthLayers);
    request.resources = &none;
    request.limits = vs_only_limits();
    request.shape.topology = NggInputTopology::TriangleList;
    request.shape.vertex_count = static_cast<uint32_t>(indices.size());
    request.shape.indices = std::make_shared<const std::vector<uint32_t>>(std::move(indices));
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = slices;
    request.raster.route = NggLayerRoute::None;
    request.raster.layer_select = true;
    request.raster.count_violations = ngg_backend_counts_violations(ctx);
    request.diagnostic = {RecompileDiagnosticStage::Vertex, 0x512e930000ull};
    return build_ngg_subgroup_draw(request, why);
}

// Depth-only state for one slice of the array at `base`: Z test LESS with writes (a fresh image
// starts at the far value 1.0), CB_TARGET_MASK 0, DB_DEPTH_VIEW naming exactly `slice`.
ResolvedPipelineState depth_slice_state(uint64_t base, uint32_t slice) {
    ResolvedPipelineState state = flipped_state();
    state.depth_test_enable = true;
    state.depth_write_enable = true;
    state.depth_compare_op = VK_COMPARE_OP_LESS;
    state.depth_read_base = base;
    state.depth_write_base = base;
    // The guest's whole-array view, slices 1..4, narrowed as expand_ngg_depth_slices narrows it.
    const uint32_t array_view = 1u | (4u << prosper::agc::Pm4::DB_DEPTH_VIEW_SLICE_MAX_SHIFT);
    state.db_depth_view = depth_view_for_slice(array_view, slice);
    state.color_write_mask = 0;
    state.color1_write_mask = 0;
    return state;
}

// Two depth-only draws into a four-slice depth array (guest slices 1..4), each replayed per slice
// in one pass per slice, as the renderer groups them. Draw A covers the upper-right half on layers
// 0..3 plus one primitive on layer 4, which the view does not have; draw B the lower-left half on
// layers 0..3. Every slice k then holds exactly (k + 1) / 8 in both halves:
//   * the layer chose the slice: an unselected replay puts layer 0's 0.125 in every slice;
//   * the pass split between A and B carried depth: a lost segment leaves A's half at 1.0;
//   * the pass wrote no colour, so its split needs no colour target and reads nothing back.
// The layer-4 primitive is culled and counted once, in the slice-0 replay.
TEST(NggSubgroupBackend, LayeredDepthOnlyDrawLandsEachLayerInItsSlice) {
    const RenderVkCtx* ctx = backend();
    if (!ctx) GTEST_SKIP() << "no backend device";
    constexpr uint32_t kSlices = 4, kFirst = 1;
    constexpr uint64_t kDepth = 0x4e474734d000ull;
    std::vector<uint32_t> upper, lower;
    for (uint32_t layer = 0; layer < kSlices; ++layer) {
        const uint32_t v = 4u * layer;
        upper.insert(upper.end(), {v + 1u, v + 3u, v + 2u});
        lower.insert(lower.end(), {v + 0u, v + 1u, v + 2u});
    }
    upper.insert(upper.end(), {17u, 19u, 18u});   // layer 4
    std::string why;
    const auto a = depth_slice_draw(*ctx, upper, kSlices, &why);
    ASSERT_TRUE(a) << why;
    const auto b = depth_slice_draw(*ctx, lower, kSlices, &why);
    ASSERT_TRUE(b) << why;

    std::array<ResolvedPipelineState, kSlices> states;
    const StatsSnapshot before = stats_now();
    for (uint32_t k = 0; k < kSlices; ++k) {
        states[k] = depth_slice_state(kDepth, kFirst + k);
        std::vector<BackendDraw> pass(2);
        for (uint32_t i = 0; i < 2; ++i) {
            pass[i].ngg_subgroup = i ? b : a;
            pass[i].ngg_layer_select = k;
            pass[i].fs = ngg_param_fragment(0, false);
            pass[i].ps = &states[k];
        }
        ASSERT_TRUE(backend_draws_leave_colour(pass));
        (void)render_draws_rgba(pass, kSize, kSize, nullptr, nullptr, true, nullptr, nullptr,
                                nullptr, nullptr, nullptr, true, nullptr, false);
    }
    const StatsSnapshot after = stats_now();
    EXPECT_EQ(after.draws - before.draws, 2u * kSlices) << "no replay was dropped";
    EXPECT_EQ(after.invalid - before.invalid, 0u);
    if (ngg_backend_counts_violations(*ctx))
        EXPECT_EQ(after.culled - before.culled, 1u) << "layer 4 is culled, counted once";

    std::vector<float> depth;
    std::string error;
    ASSERT_EQ(read_persistent_ds_depth_array(kDepth, kSize, kSize, kFirst, kSlices, depth, error),
              PersistentDsDepthArrayStatus::Ready)
        << error;
    ASSERT_EQ(depth.size(), static_cast<size_t>(kSlices) * kSize * kSize);
    for (uint32_t k = 0; k < kSlices; ++k) {
        const float expected = static_cast<float>(k + 1u) / 8.0f;
        uint32_t upper_px = 0, lower_px = 0;
        for (uint32_t y = 0; y < kSize; ++y)
            for (uint32_t x = 0; x < kSize; ++x) {
                if (x < y + 2u && y < x + 2u) continue;   // the diagonal margin
                const float z = depth[(static_cast<size_t>(k) * kSize + y) * kSize + x];
                ASSERT_EQ(z, expected) << "slice " << kFirst + k << " (" << x << "," << y << ")";
                (x >= y + 2u ? upper_px : lower_px)++;
            }
        EXPECT_EQ(upper_px, 105u);
        EXPECT_EQ(lower_px, 105u);
    }

    // The colour attachment of a colourless split restarts from the caller's clear in every
    // segment, so the final readback is that clear -- exactly what one unsplit draw returns, and
    // not the default clear a segment without the caller's arguments would show.
    const float kColour[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<BackendDraw> split_pass(2);
    for (uint32_t i = 0; i < 2; ++i) {
        split_pass[i].ngg_subgroup = i ? b : a;
        split_pass[i].fs = ngg_param_fragment(0, false);
        split_pass[i].ps = &states[0];
    }
    const auto split = render_draws_rgba(split_pass, kSize, kSize, nullptr, kColour, true);
    const auto whole = render_draws_rgba({split_pass[0]}, kSize, kSize, nullptr, kColour, true);
    const auto unset = render_draws_rgba({split_pass[0]}, kSize, kSize, nullptr, nullptr, true);
    ASSERT_FALSE(split.empty());
    EXPECT_EQ(split, whole) << "every segment starts from the caller's clear";
    EXPECT_NE(split, unset) << "control: the clear is visible in the readback";

    // Control: the same pair writing colour has no colour target to carry it across the split,
    // and is dropped by name rather than drawn with a readback per segment.
    ResolvedPipelineState coloured = states[0];
    coloured.color_write_mask = 0xfu;
    BackendDraw pair_a, pair_b;
    pair_a.ngg_subgroup = a;
    pair_b.ngg_subgroup = b;
    pair_a.ps = pair_b.ps = &coloured;
    pair_a.fs = pair_b.fs = ngg_param_fragment(0, false);
    const auto host = ngg_host_capabilities(*ctx);
    EXPECT_STREQ(ngg_backend_draw_refusal(
                     pair_a, host, 2, true, nullptr, 1,
                     backend_draws_leave_colour(std::vector<BackendDraw>{pair_a, pair_b})),
                 "ngg-backend-readback-split");
    EXPECT_EQ(ngg_backend_draw_refusal(pair_a, host, 2, true, nullptr, 1, true), nullptr)
        << "control: the colourless call splits safely";
    EXPECT_STREQ(ngg_backend_draw_refusal(pair_a, host, 2, false, nullptr, 1, true),
                 "ngg-backend-transient-depth-split")
        << "transient depth is never carried, colour or not";
}

// The rule the split uses: a draw leaves colour only when no slot's write mask is set and it has
// no other colour route; and the contract each segment of such a call renders under.
TEST(NggSubgroupBackend, OnlyAMaskedCallSplitsWithoutCarryingColour) {
    ResolvedPipelineState state;
    state.color_write_mask = 0;
    BackendDraw draw;
    draw.ps = &state;
    EXPECT_TRUE(backend_draw_leaves_colour(draw));
    ResolvedPipelineState slot0 = state;
    slot0.color_write_mask = 0x1u;
    ResolvedPipelineState slot1 = state;
    slot1.color1_write_mask = 0x8u;
    ResolvedPipelineState slot5 = state;
    slot5.color_targets[5].write_mask = 0x2u;
    for (const ResolvedPipelineState* writes : {&slot0, &slot1, &slot5}) {
        BackendDraw w = draw;
        w.ps = writes;
        EXPECT_FALSE(backend_draw_leaves_colour(w));
        EXPECT_FALSE(backend_draws_leave_colour(std::vector<BackendDraw>{draw, w}))
            << "one writer makes the call coloured";
    }
    // A decoded fast-clear value is register state, not a write: the backend never acts on it per
    // draw (Kena's shadow passes carry one with CB_TARGET_MASK 0).
    ResolvedPipelineState cleared = state;
    cleared.has_clear_color = true;
    cleared.has_clear_color1 = true;
    cleared.color_targets[3].has_clear = true;
    BackendDraw clear_value = draw;
    clear_value.ps = &cleared;
    EXPECT_TRUE(backend_draw_leaves_colour(clear_value));
    BackendDraw stateless;
    EXPECT_FALSE(backend_draw_leaves_colour(stateless)) << "no state: a draw writes RGBA";
    EXPECT_FALSE(backend_draws_leave_colour(std::span<const BackendDraw>{}))
        << "an empty call proves nothing";

    // The segment contract of a colourless call: a later segment starts as the first does (it
    // loads only what the caller asked the whole call to load) and no non-final segment copies
    // anything out; the final one reads back what the caller asked for.
    BackendColorTarget target;
    target.persistent_id = 0x4e47473400d0ull;
    target.load_existing = false;
    target.readback = true;
    BackendMrtOutputs mrt;
    mrt.color_count = 3;
    const auto later = split_segment_contract(&target, &mrt, false, false, {}, true);
    EXPECT_FALSE(later.target.load_existing) << "a later segment restarts, it does not load";
    EXPECT_FALSE(later.target.readback || later.target.readback1 || later.target.readback_slots[2])
        << "a non-final colourless segment carries nothing through the CPU";
    const auto last = split_segment_contract(&target, &mrt, false, true, {}, true);
    EXPECT_TRUE(last.target.readback) << "the final segment answers the caller";
    EXPECT_FALSE(last.target.load_existing);
    const auto coloured = split_segment_contract(&target, &mrt, false, false, {}, false);
    EXPECT_TRUE(coloured.target.load_existing && coloured.target.readback1)
        << "control: a coloured split loads and carries";
}

}   // namespace
