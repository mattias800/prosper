// A graphics launch mode belongs to the producing draw and cache identity. Retaining it must
// preserve today's emitted words; later arithmetic consumers must never borrow another mode.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/gpu_capture_bundle.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/graphics_float_mode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace {
namespace P = prosper::agc::Pm4;
alignas(256) constexpr uint32_t vertex[] = {0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u,
                                            0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u,
                                            0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u,
                                            0xf80008cfu, 0x04030201u, 0xbf810000u};
alignas(256) constexpr uint32_t pixel[] = {0x7e000280u, 0xf800180fu, 0u, 0xbf810000u};
alignas(256) constexpr uint32_t rejected_vertex[] = {0x7e000280u, 0xf80008c1u, 0u,
                                                     0xbf810000u}; // unsupported partial POS0
constexpr std::array modes{
    GraphicsFloatMode{}, GraphicsFloatMode{true, 0, false}, GraphicsFloatMode{true, 0x30, false},
    GraphicsFloatMode{true, 0x30, true}, GraphicsFloatMode{true, 0xff, true}};
GpuState launch(GraphicsFloatMode mode, bool ngg) {
    GpuState state;
    for (const auto entry :
         {std::pair{P::SPI_SHADER_PGM_LO_ES, vertex}, std::pair{P::SPI_SHADER_PGM_LO_PS, pixel}}) {
        const uint64_t address = reinterpret_cast<uint64_t>(entry.second);
        state.sh[entry.first] = uint32_t(address >> 8);
        state.sh[entry.first + 1] = uint32_t(address >> 40) & 255;
    }
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = 15;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL
                                    << P::CB_COLOR_CONTROL_MODE_SHIFT;
    state.cx[P::VGT_SHADER_STAGES_EN] = ngg ? 0x2000 : 0;
    if (mode.available)
        state.sh[ngg ? P::SPI_SHADER_PGM_RSRC1_GS : P::SPI_SHADER_PGM_RSRC1_VS] =
            (uint32_t(mode.value) << 12) | (mode.ieee_mode ? 1u << 23 : 0);
    return state;
}
size_t reader(uint64_t address, uint8_t* out, size_t count) {
    for (const auto entry : {std::pair{vertex, sizeof(vertex)}, std::pair{pixel, sizeof(pixel)},
                             std::pair{rejected_vertex, sizeof(rejected_vertex)}}) {
        const uint64_t base = reinterpret_cast<uint64_t>(entry.first);
        if (address < base || address - base >= entry.second) continue;
        const size_t offset = size_t(address - base), n = std::min(count, entry.second - offset);
        std::memcpy(out, reinterpret_cast<const uint8_t*>(entry.first) + offset, n);
        return n;
    }
    return 0;
}
void set32(std::vector<uint8_t>& bytes, size_t at, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.at(at + i) = uint8_t(value >> (8 * i));
}
GpuCaptureFile capture() {
    std::vector<DrawItem> draws;
    for (const auto mode : modes) {
        DrawItem draw;
        EXPECT_TRUE(
            realize_draw_item(launch(mode, true), nullptr, 3, std::size(vertex), false, draw));
        EXPECT_EQ(draw.vertex_float_mode, mode);
        draw.draw_index = draws.size();
        draw.command_order = draws.size() + 1;
        draws.push_back(std::move(draw));
    }
    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 1;
    GpuCaptureFile result;
    std::string error;
    EXPECT_TRUE(capture_draw_items(draws, metadata, reader, result, error)) << error;
    return result;
}
}   // namespace

TEST(GraphicsFloatLaunch, CopiedAndSharedProducersKeepDistinctWarmCacheIdentities) {
    clear_shader_recompile_cache();
    std::vector<uint32_t> unchanged;
    for (bool shared : {false, true})
        for (bool ngg : {false, true}) {
            std::set<uint64_t> identities;
            for (const auto mode : modes) {
                DrawItem cold, warm;
                const auto state = launch(mode, ngg);
                ASSERT_TRUE(realize_draw_item(state, nullptr, 3, std::size(vertex), false, cold,
                                              nullptr, shared));
                ASSERT_TRUE(realize_draw_item(state, nullptr, 3, std::size(vertex), false, warm,
                                              nullptr, shared));
                EXPECT_EQ(cold.vertex_float_mode, mode);
                EXPECT_EQ(warm.vertex_float_mode, mode);
                EXPECT_NE(cold.vs_identity, 0u);
                EXPECT_EQ(warm.vs_identity, cold.vs_identity);
                identities.insert(cold.vs_identity);
                if (unchanged.empty()) unchanged = cold.vs_words();
                EXPECT_EQ(cold.vs_words(), unchanged);
                EXPECT_EQ(warm.vs_words(), unchanged);
            }
            EXPECT_EQ(identities.size(), modes.size());
        }
}

TEST(GraphicsFloatLaunch, BothHashImplementationsRetainEveryCarrierField) {
    std::set<uint64_t> word_hashes, byte_hashes;
    for (const auto mode : modes) {
        ShaderCompileKey key;
        key.stage = ShaderProgramStage::Vertex;
        key.code =
            std::make_shared<const std::vector<uint32_t>>(std::begin(vertex), std::end(vertex));
        key.code_hash = hash_shader_code(*key.code);
        key.vertex_float_mode = mode;
        word_hashes.insert(ShaderCompileKeyHash::compute_impl<true>(key));
        byte_hashes.insert(ShaderCompileKeyHash::compute_impl<false>(key));
        for (const auto other_mode : modes) {
            auto other = key;
            other.vertex_float_mode = other_mode;
            EXPECT_EQ(key == other, mode == other_mode);
        }
    }
    EXPECT_EQ(word_hashes.size(), modes.size());
    EXPECT_EQ(byte_hashes.size(), modes.size());
}

TEST(GraphicsFloatLaunch, NativeAndNggCompilerCarriersAreBehaviorNeutralAndCanonical) {
    const auto expected = recompile_vertex(vertex, std::size(vertex));
    ASSERT_FALSE(expected.empty());
    const auto code = prosper::test::ngg::synthetic_program();
    const auto resources = prosper::test::ngg::synthetic_resources();
    NggSubgroupShellConfig shell;
    shell.waves = 2;
    shell.user_sgprs = 4;
    const auto ngg = recompile_ngg_subgroup(code.data(), code.size(), &resources, shell);
    ASSERT_FALSE(ngg.empty());
    for (const auto mode : modes) {
        EXPECT_EQ(recompile_vertex(vertex, std::size(vertex), nullptr, nullptr, false, 0,
                                   {RecompileDiagnosticStage::Vertex, 0}, {}, mode),
                  expected);
        shell.float_mode = mode;
        EXPECT_EQ(recompile_ngg_subgroup(code.data(), code.size(), &resources, shell), ngg);
    }
    for (const auto mode :
         {GraphicsFloatMode{false, 1, false}, GraphicsFloatMode{false, 0, true}}) {
        EXPECT_TRUE(recompile_vertex(vertex, std::size(vertex), nullptr, nullptr, false, 0,
                                     {RecompileDiagnosticStage::Vertex, 0}, {}, mode)
                        .empty());
        shell.float_mode = mode;
        EXPECT_TRUE(recompile_ngg_subgroup(code.data(), code.size(), &resources, shell).empty());
    }
}

TEST(GraphicsFloatLaunch, LinkedVertexCompilationAndCacheKeepTheProducingMode) {
    constexpr uint32_t prolog[] = {0xbfa00003u, 0xbe802006u}; // SETPRIO; SETPC s[6:7]
    std::set<uint64_t> identities;
    std::vector<uint32_t> expected;
    for (const auto mode : modes) {
        const auto direct = recompile_vertex_chain(prolog, std::size(prolog), vertex,
                                                   std::size(vertex), nullptr, nullptr, false, 0,
                                                   {RecompileDiagnosticStage::Vertex, 0}, {}, mode);
        ASSERT_FALSE(direct.empty());
        if (expected.empty()) expected = direct;
        EXPECT_EQ(direct, expected);
        uint64_t cold_id = 0, warm_id = 0;
        const auto cold = recompile_vertex_chain_cached_shared(
            prolog, std::size(prolog), vertex, std::size(vertex), nullptr, nullptr, &cold_id, 0,
            false, {}, nullptr, mode);
        const auto warm = recompile_vertex_chain_cached_shared(
            prolog, std::size(prolog), vertex, std::size(vertex), nullptr, nullptr, &warm_id, 0,
            false, {}, nullptr, mode);
        ASSERT_TRUE(cold && warm);
        EXPECT_EQ(*cold, expected);
        EXPECT_EQ(cold, warm);
        EXPECT_NE(cold_id, 0u);
        EXPECT_EQ(cold_id, warm_id);
        identities.insert(cold_id);
    }
    EXPECT_EQ(identities.size(), modes.size());
}

TEST(GraphicsFloatLaunch, CaptureBundleAndReplayRetainModeWithoutUpgradingLegacyAuthority) {
    const auto source = capture();
    ASSERT_EQ(source.draws.size(), modes.size());
    std::vector<uint8_t> bytes;
    std::string error;
    GpuCaptureFile loaded;
    ASSERT_TRUE(serialize_gpu_capture(source, bytes, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    EXPECT_EQ(loaded.format_version, 77u);
    GpuReplayFrame frame;
    ASSERT_TRUE(materialize_gpu_replay(loaded, frame, error)) << error;
    GpuCaptureBundle bundle;
    ASSERT_TRUE(append_gpu_capture_bundle(bundle, source, error)) << error;
    GpuCaptureFile full, manifest;
    ASSERT_TRUE(materialize_gpu_capture_bundle_submit(bundle, 0, full, error)) << error;
    ASSERT_TRUE(materialize_gpu_capture_bundle_manifest(bundle, 0, manifest, error)) << error;
    ASSERT_EQ(frame.items.size(), modes.size());
    for (size_t i = 0; i < modes.size(); ++i) {
        EXPECT_EQ(loaded.draws.at(i).vertex_float_mode, modes[i]);
        EXPECT_EQ(frame.items.at(i).vertex_float_mode, modes[i]);
        EXPECT_EQ(full.draws.at(i).vertex_float_mode, modes[i]);
        EXPECT_EQ(manifest.draws.at(i).vertex_float_mode, modes[i]);
    }
    // Remove the genuine append-only v77 tail: historical captures cannot invent an observed zero.
    bytes.resize(bytes.size() - 8 - 3 * modes.size());
    set32(bytes, 8, 76);
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws) EXPECT_EQ(draw.vertex_float_mode, GraphicsFloatMode{});
    ASSERT_TRUE(serialize_gpu_capture(loaded, bytes, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws) EXPECT_EQ(draw.vertex_float_mode, GraphicsFloatMode{});
}

TEST(GraphicsFloatLaunch, MalformedAndTruncatedCaptureModesRefuse) {
    auto source = capture();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(source, bytes, error)) << error;
    const size_t tail = bytes.size() - 8 - 3 * modes.size();
    GpuCaptureFile loaded;
    for (size_t end = tail; end < bytes.size(); ++end) {
        auto bad = bytes;
        bad.resize(end);
        EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error)) << end;
    }
    for (const auto field : {tail, bytes.size() - 4}) {
        auto bad = bytes;
        set32(bad, field, UINT32_MAX);
        EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error));
    }
    for (const auto field : {tail + 4, tail + 6}) {
        auto bad = bytes;
        bad.at(field) = 2;
        EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error));
    }
    auto bad = bytes;
    bad.at(tail + 5) = 1;
    EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error));
    source.draws[0].vertex_float_mode = {false, 0, true};
    EXPECT_FALSE(serialize_gpu_capture(source, bytes, error));
    GpuReplayFrame frame;
    EXPECT_FALSE(materialize_gpu_replay(source, frame, error));
}

TEST(GraphicsFloatLaunch, RefusedProducerRetainsModeAndCannotBecomeComputeAuthority) {
    const GraphicsFloatMode mode{true, 0xa5, true};
    auto state = launch(mode, true);
    const uint64_t address = reinterpret_cast<uint64_t>(rejected_vertex);
    state.sh[P::SPI_SHADER_PGM_LO_ES] = uint32_t(address >> 8);
    state.sh[P::SPI_SHADER_PGM_HI_ES] = uint32_t(address >> 40) & 255;
    DrawItem rejected;
    OperationRealizationFailure failure;
    ASSERT_FALSE(realize_draw_item(state, nullptr, 3, std::size(rejected_vertex), false, rejected,
                                   &failure));
    ASSERT_EQ(failure.reason, RealizationFailureReason::ShaderRecompile);
    EXPECT_EQ(failure.vertex_float_mode, mode);
    failure.index = 7;
    failure.command_order = 1;
    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 1;
    GpuCaptureFile source, loaded;
    std::string error;
    std::vector<uint8_t> bytes;
    ASSERT_TRUE(capture_submit_items({}, {}, {{SubmitOperationKind::Draw, 7, 1}}, metadata, reader,
                                     source, error, {}, {failure}))
        << error;
    ASSERT_TRUE(serialize_gpu_capture(source, bytes, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.failure_diagnostics.size(), 1u);
    EXPECT_EQ(loaded.failure_diagnostics[0].vertex_float_mode, mode);
    loaded.failure_diagnostics[0].kind = SubmitOperationKind::Dispatch;
    EXPECT_FALSE(serialize_gpu_capture(loaded, bytes, error));
    EXPECT_EQ(error, "invalid failed-draw graphics float mode");
}
