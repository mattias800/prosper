// A producing half-overflow fact must survive caches and replay without changing arithmetic.
// In particular, old MODE/IEEE/raw-register evidence must not become a newly observed clear bit.
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/state/graphics_float_mode.hpp"
#include "gpu/state/render_state.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <vector>

using namespace prosper::gpu;
namespace {
namespace P = prosper::agc::Pm4;
constexpr std::array<Fp16OverflowMode, 3> modes{{{}, {true, false}, {true, true}}};
alignas(256) constexpr uint32_t vertex[] = {0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u,
                                            0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u,
                                            0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u,
                                            0xf80008cfu, 0x04030201u, 0xbf810000u};
alignas(256) constexpr uint32_t pixel[] = {0x7e000280u, 0xf800180fu, 0u, 0xbf810000u};
alignas(256) constexpr uint32_t refused_pixel[] = {0xbfa40001u, 0x7e000280u, 0xf800180fu, 0u,
                                                   0xbf810000u};

GpuState launch(bool saturate, bool ngg, bool refused = false) {
    GpuState state;
    const auto program = [&](uint32_t reg, const uint32_t* words) {
        const uint64_t address = reinterpret_cast<uint64_t>(words);
        state.sh[reg] = uint32_t(address >> 8);
        state.sh[reg + 1] = uint32_t(address >> 40) & 255;
    };
    program(P::SPI_SHADER_PGM_LO_ES, vertex);
    program(P::SPI_SHADER_PGM_LO_PS, refused ? refused_pixel : pixel);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = 15;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL
                                    << P::CB_COLOR_CONTROL_MODE_SHIFT;
    state.cx[P::VGT_SHADER_STAGES_EN] = ngg ? 0x2000 : 0;
    state.sh[ngg ? 0x8a : 0x4a] = 0x00030000u | (saturate ? 0x80000000u : 0);
    // Deliberately opposite policies: copying the other stage's fact must fail.
    state.sh[0x0a] = 0x00030000u | (saturate ? 0 : 0x20000000u);
    state.sh[P::COMPUTE_PGM_RSRC1] = saturate ? 0 : 0x04000000u;
    return state;
}
size_t reader(uint64_t address, uint8_t* out, size_t count) {
    for (const auto span : {std::pair{vertex, sizeof(vertex)}, std::pair{pixel, sizeof(pixel)},
                            std::pair{refused_pixel, sizeof(refused_pixel)}}) {
        const uint64_t base = reinterpret_cast<uint64_t>(span.first);
        if (address < base || address - base >= span.second) continue;
        const size_t offset = size_t(address - base);
        const size_t n = std::min(count, span.second - offset);
        std::memcpy(out, reinterpret_cast<const uint8_t*>(span.first) + offset, n);
        return n;
    }
    return 0;
}
void set32(std::vector<uint8_t>& bytes, size_t at, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.at(at + i) = uint8_t(value >> (8 * i));
}
void checksum(std::vector<uint8_t>& bytes) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i + 8 < bytes.size(); ++i) hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    for (unsigned i = 0; i < 8; ++i) bytes.at(bytes.size() - 8 + i) = uint8_t(hash >> (8 * i));
}
std::vector<uint8_t> sidecar(const RefusedGraphicsInputs& input) {
    const auto path = prosper_test::test_scratch_dir() / "fp16-overflow.inputs";
    EXPECT_TRUE(write_refused_graphics_inputs(path, "vs", input));
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), {}};
}
} // namespace

TEST(Fp16OverflowLaunch, EachHardwareBankUsesItsOwnPublishedBit) {
    constexpr std::array stages{GraphicsHardwareStage::Vertex, GraphicsHardwareStage::Geometry,
                                GraphicsHardwareStage::Hull};
    constexpr std::array<uint32_t, 3> registers{0x4a, 0x8a, 0x10a};
    constexpr std::array<uint32_t, 3> masks{0x80000000u, 0x80000000u, 0x40000000u};
    for (size_t stage = 0; stage < stages.size(); ++stage)
        for (bool saturate : {false, true})
            for (uint32_t mode = 0; mode < 256; ++mode) {
                GpuState state;
                const uint32_t wrong_bit = stage == 2 ? 0x80000000u : 0x40000000u;
                state.sh[registers[stage]] =
                    (mode << 12) | 0x00800000u | wrong_bit | (saturate ? masks[stage] : 0);
                state.sh[P::COMPUTE_PGM_RSRC1] = saturate ? 0 : 0x04000000u;
                const auto actual = graphics_float_mode(state, stages[stage]);
                EXPECT_EQ(actual.fp16_overflow, (Fp16OverflowMode{true, saturate}));
                EXPECT_EQ(actual.value, mode);
                EXPECT_TRUE(actual.ieee_mode);
                EXPECT_TRUE(actual.canonical());
            }
    for (bool saturate : {false, true}) {
        GpuState state;
        state.sh[0x0a] = 0x84030000u | (saturate ? 0x20000000u : 0);
        EXPECT_EQ(extract_render_state_uncached(state).ps_float_flags.fp16_overflow,
                  (Fp16OverflowMode{true, saturate}));
        EXPECT_EQ(
            compute_fp16_overflow_from_rsrc1(true, 0xa0030000u | (saturate ? 0x04000000u : 0)),
            (Fp16OverflowMode{true, saturate}));
    }
    EXPECT_EQ(compute_fp16_overflow_from_rsrc1(false, UINT32_MAX), Fp16OverflowMode{});
    EXPECT_EQ(extract_render_state_uncached(GpuState{}).ps_float_flags.fp16_overflow,
              Fp16OverflowMode{});
    EXPECT_EQ(graphics_float_mode_from_rsrc1(UINT32_MAX).fp16_overflow, Fp16OverflowMode{});
}

TEST(Fp16OverflowLaunch, AvailabilityIsIndependentAndUnknownPayloadIsInvalid) {
    for (auto mode : modes) {
        FragmentFloatFlags fragment{true, true, true, mode};
        GraphicsFloatMode vertex_mode{true, 0xff, true, mode};
        EXPECT_TRUE(fragment.canonical());
        EXPECT_TRUE(vertex_mode.canonical());
        fragment.available = vertex_mode.available = false;
        fragment.ieee_mode = fragment.dx10_clamp = vertex_mode.ieee_mode = false;
        vertex_mode.value = 0;
        EXPECT_TRUE(fragment.canonical());
        EXPECT_TRUE(vertex_mode.canonical());
    }
    const Fp16OverflowMode malformed{false, true};
    EXPECT_FALSE(malformed.canonical());
    EXPECT_FALSE((FragmentFloatFlags{true, false, false, malformed}.canonical()));
    EXPECT_FALSE((GraphicsFloatMode{true, 0, false, malformed}.canonical()));
}

TEST(Fp16OverflowLaunch, BothKeyHashesDistinguishIndependentFactWithEqualOtherInputs) {
    for (auto stage : {ShaderProgramStage::Vertex, ShaderProgramStage::Fragment}) {
        std::set<uint64_t> words, bytes;
        std::vector<ShaderCompileKey> keys;
        for (auto mode : modes) {
            ShaderCompileKey key;
            key.stage = stage;
            key.code =
                std::make_shared<const std::vector<uint32_t>>(std::begin(pixel), std::end(pixel));
            key.code_hash = hash_shader_code(*key.code);
            if (stage == ShaderProgramStage::Fragment)
                key.fragment_float_flags.fp16_overflow = mode;
            else
                key.vertex_float_mode.fp16_overflow = mode;
            words.insert(ShaderCompileKeyHash::compute_impl<true>(key));
            bytes.insert(ShaderCompileKeyHash::compute_impl<false>(key));
            keys.push_back(key);
        }
        EXPECT_EQ(words.size(), modes.size());
        EXPECT_EQ(bytes.size(), modes.size());
        for (size_t i = 0; i < keys.size(); ++i)
            for (size_t j = 0; j < keys.size(); ++j) EXPECT_EQ(keys[i] == keys[j], i == j);
    }
}

TEST(Fp16OverflowLaunch, DirectAndLinkedCachesKeepUnknownClearAndSetWithUnchangedModules) {
    clear_shader_recompile_cache();
    constexpr uint32_t prolog[] = {0xbfa00003u, 0xbe802006u};
    for (bool linked : {false, true}) {
        std::set<uint64_t> identities;
        std::vector<uint32_t> expected;
        for (auto mode : modes) {
            GraphicsFloatMode producing{true, 0x30, false, mode};
            uint64_t cold_id = 0, warm_id = 0;
            const auto compile = [&](uint64_t& identity) {
                if (linked)
                    return recompile_vertex_chain_cached_shared(
                        prolog, std::size(prolog), vertex, std::size(vertex), nullptr, nullptr,
                        &identity, 0, false, {}, nullptr, producing);
                return recompile_graphics_shader_cached_shared(
                    ShaderProgramStage::Vertex, vertex, std::size(vertex), nullptr, nullptr,
                    nullptr, &identity, false, 0, false, {}, {}, {}, {}, {}, nullptr, nullptr,
                    nullptr, {}, producing);
            };
            const auto cold = compile(cold_id), warm = compile(warm_id);
            ASSERT_TRUE(cold && warm);
            if (expected.empty()) expected = *cold;
            EXPECT_EQ(*cold, expected);
            EXPECT_EQ(cold, warm);
            EXPECT_NE(cold_id, 0u);
            EXPECT_EQ(cold_id, warm_id);
            identities.insert(cold_id);
        }
        EXPECT_EQ(identities.size(), modes.size());
    }
}

TEST(Fp16OverflowLaunch, FragmentWarmCacheCannotBorrowUnknownClearOrSetAuthority) {
    std::set<uint64_t> identities;
    std::vector<uint32_t> expected;
    for (auto mode : modes) {
        FragmentFloatFlags flags{true, false, false, mode};
        uint64_t cold_id = 0, warm_id = 0;
        const auto compile = [&](uint64_t& id) {
            return recompile_graphics_shader_cached_shared(
                ShaderProgramStage::Fragment, pixel, std::size(pixel), nullptr, nullptr, nullptr,
                &id, false, 0, false, {}, {}, {}, flags);
        };
        const auto cold = compile(cold_id), warm = compile(warm_id);
        ASSERT_TRUE(cold && warm);
        EXPECT_EQ(cold, warm);
        EXPECT_NE(cold_id, 0u);
        EXPECT_EQ(cold_id, warm_id);
        if (expected.empty()) expected = *cold;
        EXPECT_EQ(*cold, expected);
        identities.insert(cold_id);
    }
    EXPECT_EQ(identities.size(), modes.size());
}

TEST(Fp16OverflowLaunch, CopiedAndSharedDrawsAndFailuresRetainProducingStageFacts) {
    for (bool shared : {false, true})
        for (bool ngg : {false, true}) {
            std::set<uint64_t> vertex_ids, pixel_ids;
            std::vector<uint32_t> expected_vertex, expected_pixel;
            for (bool saturate : {false, true}) {
                DrawItem cold, warm, rejected;
                const auto state = launch(saturate, ngg);
                ASSERT_TRUE(realize_draw_item(state, nullptr, 3, std::size(vertex), false, cold,
                                              nullptr, shared));
                ASSERT_TRUE(realize_draw_item(state, nullptr, 3, std::size(vertex), false, warm,
                                              nullptr, shared));
                for (const auto* item : {&cold, &warm}) {
                    EXPECT_EQ(item->vertex_float_mode.fp16_overflow,
                              (Fp16OverflowMode{true, saturate}));
                    EXPECT_EQ(item->ps_float_flags.fp16_overflow,
                              (Fp16OverflowMode{true, !saturate}));
                }
                if (expected_vertex.empty()) expected_vertex = cold.vs_words();
                if (expected_pixel.empty()) expected_pixel = cold.fs_words();
                EXPECT_EQ(cold.vs_words(), expected_vertex);
                EXPECT_EQ(cold.fs_words(), expected_pixel);
                EXPECT_EQ(warm.vs_identity, cold.vs_identity);
                EXPECT_EQ(warm.fs_identity, cold.fs_identity);
                vertex_ids.insert(cold.vs_identity);
                pixel_ids.insert(cold.fs_identity);
                OperationRealizationFailure failure;
                ASSERT_FALSE(realize_draw_item(launch(saturate, ngg, true), nullptr, 3,
                                               std::size(vertex), false, rejected, &failure,
                                               shared));
                EXPECT_EQ(failure.ps_float_flags.fp16_overflow,
                          (Fp16OverflowMode{true, !saturate}));
                EXPECT_EQ(failure.vertex_float_mode.fp16_overflow,
                          (Fp16OverflowMode{true, saturate}));
            }
            EXPECT_EQ(vertex_ids.size(), 2u);
            EXPECT_EQ(pixel_ids.size(), 2u);
        }
}

TEST(Fp16OverflowLaunch, CompileCaseAppendixAndLegacyUnknownPreserveActualReplay) {
    for (auto mode : modes) {
        FragmentCompileCase input;
        input.code.assign(std::begin(pixel), std::end(pixel));
        input.float_flags = {true, false, false, mode};
        // Opposite evidence cannot stand in for independently recorded compiler authority.
        input.launch_rsrc1 = {true, mode.saturate ? 0u : 0x20000000u};
        input.expected_produced = true;
        input.compiler = "fp16-overflow-test";
        input.interpolation = fragment_interpolation_layout(pixel, std::size(pixel));
        CompilerChoiceTrace trace;
        {
            CompilerChoiceScope recording(trace);
            input.source = recompile_fragment(pixel, std::size(pixel), nullptr, nullptr, UINT32_MAX,
                                              &input.interpolation, false, {}, {}, nullptr, {},
                                              input.float_flags);
        }
        input.choices = trace;
        ASSERT_FALSE(input.source.empty());
        auto wire = encode_fragment_compile_case(input);
        auto actual = decode_fragment_compile_case(wire);
        EXPECT_EQ(actual.float_flags.fp16_overflow, mode);
        EXPECT_EQ(replay_fragment_compile_case(actual, false), input.source);
        if (!mode.available) continue;
        auto fact_only = input;
        fact_only.float_flags.available = false;
        fact_only.launch_rsrc1 = {};
        const auto fact_only_wire = encode_fragment_compile_case(fact_only);
        EXPECT_EQ(fact_only_wire.at(8), 8u);
        EXPECT_EQ(decode_fragment_compile_case(fact_only_wire).float_flags.fp16_overflow, mode);
        ASSERT_EQ(wire.at(8), 8u);
        auto legacy = wire;
        legacy.erase(legacy.end() - 10, legacy.end() - 8);
        set32(legacy, 8, 7);
        checksum(legacy);
        actual = decode_fragment_compile_case(legacy);
        EXPECT_TRUE(actual.complete);
        EXPECT_TRUE(actual.float_flags.available);
        EXPECT_EQ(actual.float_flags.fp16_overflow, Fp16OverflowMode{});
        EXPECT_EQ(actual.launch_rsrc1, input.launch_rsrc1);
        EXPECT_EQ(replay_fragment_compile_case(actual, false), input.source);
        for (size_t field : {wire.size() - 10, wire.size() - 9}) {
            auto bad = wire;
            bad.at(field) = 2;
            checksum(bad);
            EXPECT_THROW(decode_fragment_compile_case(bad), std::runtime_error);
        }
        auto bad = wire;
        bad.at(bad.size() - 10) = 0;
        bad.at(bad.size() - 9) = 1;
        checksum(bad);
        EXPECT_THROW(decode_fragment_compile_case(bad), std::runtime_error);
    }
}

TEST(Fp16OverflowLaunch, CaptureAppendixRetainsBothStagesAndLegacyNeverInfersRawBits) {
    std::vector<DrawItem> draws;
    for (bool saturate : {false, true}) {
        DrawItem draw;
        ASSERT_TRUE(
            realize_draw_item(launch(saturate, true), nullptr, 3, std::size(vertex), false, draw));
        draw.draw_index = draws.size();
        draw.command_order = draws.size() + 1;
        draws.push_back(std::move(draw));
    }
    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 1;
    GpuCaptureFile source, actual;
    std::string error;
    ASSERT_TRUE(capture_draw_items(draws, metadata, reader, source, error)) << error;
    std::vector<uint8_t> wire;
    ASSERT_TRUE(serialize_gpu_capture(source, wire, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(wire, actual, error)) << error;
    ASSERT_EQ(actual.format_version, 78u);
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(actual, replay, error)) << error;
    for (size_t i = 0; i < draws.size(); ++i) {
        EXPECT_EQ(replay.items[i].vertex_float_mode, draws[i].vertex_float_mode);
        EXPECT_EQ(replay.items[i].ps_float_flags, draws[i].ps_float_flags);
    }
    const size_t tail = wire.size() - 16 - 4 * draws.size();
    auto legacy = wire;
    legacy.resize(tail);
    set32(legacy, 8, 77);
    ASSERT_TRUE(deserialize_gpu_capture(legacy, actual, error)) << error;
    for (const auto& draw : actual.draws) {
        EXPECT_TRUE(draw.vertex_float_mode.available);
        EXPECT_TRUE(draw.ps_float_flags.available);
        EXPECT_EQ(draw.vertex_float_mode.fp16_overflow, Fp16OverflowMode{});
        EXPECT_EQ(draw.ps_float_flags.fp16_overflow, Fp16OverflowMode{});
    }
    for (size_t field : {tail, wire.size() - 4}) {
        auto bad = wire;
        set32(bad, field, UINT32_MAX);
        EXPECT_FALSE(deserialize_gpu_capture(bad, actual, error));
    }
    for (size_t field = tail + 4; field < wire.size() - 4; ++field) {
        auto bad = wire;
        bad.at(field) = 2;
        EXPECT_FALSE(deserialize_gpu_capture(bad, actual, error));
    }
    for (size_t end = tail; end < wire.size(); ++end)
        EXPECT_FALSE(deserialize_gpu_capture(std::vector<uint8_t>(wire.begin(), wire.begin() + end),
                                             actual, error));
    for (auto& draw : source.draws) {
        draw.vertex_float_mode = {};
        draw.ps_float_flags = {};
        draw.ps_launch_rsrc1 = {};
    }
    source.draws[0].vertex_float_mode.fp16_overflow = {true, false};
    ASSERT_TRUE(serialize_gpu_capture(source, wire, error)) << error;
    ASSERT_TRUE(deserialize_gpu_capture(wire, actual, error)) << error;
    EXPECT_EQ(actual.format_version, 78u);
    EXPECT_EQ(actual.draws[0].vertex_float_mode.fp16_overflow, (Fp16OverflowMode{true, false}));
}

TEST(Fp16OverflowLaunch, RefusedSidecarVersionAndIdentityKeepIndependentFacts) {
    std::set<uint64_t> identities;
    for (auto mode : modes) {
        RefusedGraphicsInputs input;
        input.selector = RefusedGraphicsCompiler::NggSubgroup;
        input.raw = {0xbf810000u};
        input.invocation.code = input.raw;
        input.invocation.expected_reject = "synthetic terminal";
        input.vertex_float_mode = {true, 0x30, false, mode};
        input.shell.float_mode = {true, 0x30, false, mode};
        input.launch.observed = true;
        input.launch.rsrc1_available = 1;
        input.launch.rsrc1[0] = 0x80030000u;
        const auto wire = sidecar(input);
        const auto actual = decode_refused_shader_inputs(wire);
        ASSERT_TRUE(actual.graphics);
        EXPECT_EQ(actual.graphics->vertex_float_mode, input.vertex_float_mode);
        EXPECT_EQ(actual.graphics->shell.float_mode, input.shell.float_mode);
        EXPECT_EQ(refused_graphics_inputs_identity(*actual.graphics),
                  refused_graphics_inputs_identity(input));
        identities.insert(refused_graphics_inputs_identity(input));
        if (!mode.available) continue;
        ASSERT_EQ(wire.at(8), 6u);
        auto legacy = wire;
        legacy.erase(legacy.end() - 12, legacy.end() - 8);
        set32(legacy, 8, 5);
        checksum(legacy);
        const auto old = decode_refused_shader_inputs(legacy);
        ASSERT_TRUE(old.graphics);
        EXPECT_EQ(old.graphics->shell.float_mode.fp16_overflow, Fp16OverflowMode{});
        EXPECT_EQ(old.graphics->vertex_float_mode.fp16_overflow, Fp16OverflowMode{});
        // Then remove schema5's MODE/IEEE tail: existing raw GS still cannot add overflow authority.
        legacy.erase(legacy.end() - 14, legacy.end() - 8);
        set32(legacy, 8, 3);
        checksum(legacy);
        const auto older = decode_refused_shader_inputs(legacy);
        ASSERT_TRUE(older.graphics);
        EXPECT_TRUE(older.graphics->shell.float_mode.available);
        EXPECT_EQ(older.graphics->shell.float_mode.fp16_overflow, Fp16OverflowMode{});
        EXPECT_EQ(older.graphics->vertex_float_mode, GraphicsFloatMode{});
    }
    EXPECT_EQ(identities.size(), modes.size());
}

TEST(Fp16OverflowLaunch, EveryGraphicsInputSetFactIndependentlyChangesDiagnosticIdentity) {
    RefusedGraphicsInputs input;
    input.selector = RefusedGraphicsCompiler::NggSubgroup;
    input.raw = input.invocation.code = {0xbf810000u};
    std::set<uint64_t> identities;
    for (auto vertex_mode : modes)
        for (auto shell_mode : modes)
            for (auto fragment_mode : modes) {
                input.vertex_float_mode.fp16_overflow = vertex_mode;
                input.shell.float_mode.fp16_overflow = shell_mode;
                input.invocation.float_flags.fp16_overflow = fragment_mode;
                identities.insert(refused_graphics_inputs_identity(input));
                const auto wire = sidecar(input);
                const auto decoded = decode_refused_shader_inputs(wire);
                ASSERT_TRUE(decoded.graphics);
                EXPECT_EQ(decoded.graphics->vertex_float_mode, input.vertex_float_mode);
                EXPECT_EQ(decoded.graphics->shell.float_mode, input.shell.float_mode);
                EXPECT_EQ(decoded.graphics->invocation.float_flags, input.invocation.float_flags);
                if (vertex_mode.available || shell_mode.available) EXPECT_EQ(wire.at(8), 6u);
            }
    EXPECT_EQ(identities.size(), 27u);
}

TEST(Fp16OverflowLaunch, GraphicsFactAloneCannotBecomeComputeFailureAuthority) {
    DrawItem rejected;
    OperationRealizationFailure failure;
    ASSERT_FALSE(realize_draw_item(launch(true, true, true), nullptr, 3, std::size(vertex), false,
                                   rejected, &failure));
    failure.index = 7;
    failure.command_order = 1;
    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 1;
    GpuCaptureFile source;
    std::string error;
    ASSERT_TRUE(capture_submit_items({}, {}, {{SubmitOperationKind::Draw, 7, 1}}, metadata, reader,
                                     source, error, {}, {failure}))
        << error;
    for (bool vertex_fact : {false, true}) {
        auto bad = source;
        auto& diagnostic = bad.failure_diagnostics[0];
        diagnostic.kind = SubmitOperationKind::Dispatch;
        diagnostic.ps_launch_rsrc1 = {};
        diagnostic.ps_float_mode = {};
        diagnostic.ps_float_flags = {};
        diagnostic.vertex_float_mode = {};
        if (vertex_fact)
            diagnostic.vertex_float_mode.fp16_overflow = {true, false};
        else
            diagnostic.ps_float_flags.fp16_overflow = {true, false};
        std::vector<uint8_t> wire;
        EXPECT_FALSE(serialize_gpu_capture(bad, wire, error));
        EXPECT_EQ(error, vertex_fact ? "invalid failed-draw graphics float mode"
                                     : "invalid failed-draw fragment float flags");
    }
}

TEST(Fp16OverflowLaunch, FailedComputeCaptureRetainsFactAndRejectsAbsentOrForeignConfig) {
    for (auto mode : modes) {
        GpuCaptureFile source, actual;
        source.format_version = 78;
        source.operations.push_back({SubmitOperationKind::Dispatch, 7, 1, false});
        GpuCapturedOperationFailure failure;
        failure.kind = SubmitOperationKind::Dispatch;
        failure.source_index = 7;
        failure.command_order = 1;
        failure.reason = RealizationFailureReason::ShaderRecompile;
        GpuCapturedStageDiagnostic stage;
        stage.stage = ShaderProgramStage::Compute;
        stage.recompile_config_available = true;
        stage.recompile_config.local_x = 1;
        stage.recompile_config.fp16_overflow = mode;
        failure.stages.push_back(stage);
        source.failure_diagnostics.push_back(failure);
        std::vector<uint8_t> wire;
        std::string error;
        ASSERT_TRUE(serialize_gpu_capture(source, wire, error)) << error;
        ASSERT_TRUE(deserialize_gpu_capture(wire, actual, error)) << error;
        EXPECT_EQ(actual.failure_diagnostics[0].stages[0].recompile_config.fp16_overflow, mode);
        if (!mode.available) continue;
        ASSERT_EQ(actual.format_version, 78u);
        // v78: empty draw, one failed graphics pair, empty compute, one failed-compute stage.
        auto legacy = wire;
        legacy.resize(legacy.size() - 26);
        set32(legacy, 8, 77);
        ASSERT_TRUE(deserialize_gpu_capture(legacy, actual, error)) << error;
        EXPECT_EQ(actual.failure_diagnostics[0].stages[0].recompile_config.fp16_overflow,
                  Fp16OverflowMode{});
        for (size_t field : {wire.size() - 2, wire.size() - 1}) {
            auto bad = wire;
            bad.at(field) = 2;
            EXPECT_FALSE(deserialize_gpu_capture(bad, actual, error));
        }
        auto absent = source;
        absent.failure_diagnostics[0].stages[0].recompile_config_available = false;
        EXPECT_FALSE(serialize_gpu_capture(absent, wire, error));
        EXPECT_EQ(error, "invalid failed-compute half overflow presence");
        auto foreign = source;
        foreign.failure_diagnostics[0].stages[0].stage = ShaderProgramStage::Vertex;
        EXPECT_FALSE(serialize_gpu_capture(foreign, wire, error));
    }
}
