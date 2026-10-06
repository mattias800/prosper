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
#include "fixtures/render_runner.h"

#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/ngg_raster_runner.h"
#include "fixtures/ngg_subgroup_runner.h"
#include "fixtures/test_data.h"
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
#include <string>
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

std::shared_ptr<const NggSubgroupDraw> kena_draw(const RenderVkCtx& ctx, uint32_t vertices,
                                                 uint32_t instances, uint32_t slices,
                                                 std::string* why) {
    const KenaInputs& in = kena_inputs();
    NggSubgroupDrawRequest request;
    request.linked_code = in.linked.data();
    request.dwords = in.linked.size();
    request.resources = &in.table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.limits = ngg::kena_limits();
    request.shape.topology = NggInputTopology::TriangleStrip;
    request.shape.vertex_count = vertices;
    request.shape.instance_count = instances;
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = slices;
    request.raster.route = backend_route(ctx);
    request.raster.count_violations = ngg_backend_counts_violations(ctx);
    request.push_constants.assign(ngg::kKenaUserSgprs, 0u);
    request.diagnostic = {RecompileDiagnosticStage::Vertex, 0x5009440000ull};
    return build_ngg_subgroup_draw(request, why);
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
    EXPECT_EQ(lut->guest_bindings, (std::vector<uint32_t>{2, 3, 4, 5, 6, 7, 8, 9}));

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

}   // namespace
