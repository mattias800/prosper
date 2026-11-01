// Retain producing RSRC2 authority independently of raw observations. Old records stay unknown;
// malformed tails cannot silently turn an absent system input into an interpolation service.
#include "build_revision.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::native_pixel_parameter;
FragmentCompileCase compile_case(bool known = true) {
    FragmentCompileCase c;
    c.compiler = std::string(prosper::embedded_build_revision()) + ':' +
                 prosper::embedded_build_source_identity();
    c.program_address = 0x5543100000ull;
    c.code = fixture::program();
    c.has_system_inputs = true;
    c.system_inputs = fixture::mapping();
    if (!known) c.system_inputs.parameter_launch = {};
    c.float_transport = {FloatTransportProfile::ExplicitNonFinite32};
    c.interpolation = fragment_interpolation_layout(c.code.data(), c.code.size(), &c.system_inputs);
    std::vector<uint32_t> source;
    {
        CompilerChoiceScope choices(c.choices);
        TerminalRejectCapture rejection;
        source = recompile_fragment(c.code.data(), c.code.size(), nullptr, &c.system_inputs,
                                    UINT32_MAX, &c.interpolation, false,
                                    {RecompileDiagnosticStage::Fragment, c.program_address}, {},
                                    nullptr, c.float_transport);
        const auto rejects = rejection.take();
        if (!rejects.empty())
            c.expected_reject = rejects.back().first + " " + rejects.back().second;
    }
    finish_fragment_compile_case(c, std::move(source));
    return c;
}
void rehash(std::vector<uint8_t>& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < bytes.size() - 8; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    for (unsigned i = 0; i < 8; ++i) bytes[bytes.size() - 8 + i] = uint8_t(hash >> (8 * i));
}
std::vector<uint32_t> stored_stage(uint32_t stage) {
    std::vector<uint32_t> words{
        0x07230203, 0x00010000, 0, 5,          0, 0x0003000e, 0,         1, 0x0005000f, stage,
        1,          0x6e69616d, 0, 0x00020013, 2, 0x00030021, 3,         2, 0x00050036, 2,
        1,          0,          3, 0x000200f8, 4, 0x000100fd, 0x00010038};
    if (stage == 4) words.insert(words.begin() + 13, {0x00030010, 1, 7});
    return words;
}
GpuCaptureFile frame(bool known = true) {
    GpuCaptureFile c;
    c.metadata.title_id = "PPSA00000";
    c.metadata.width = c.metadata.height = 8;
    for (unsigned i = 0; i < 2; ++i) {
        GpuCapturedDraw draw;
        draw.vs = stored_stage(0);
        draw.fs = stored_stage(4);
        draw.draw_index = i;
        draw.command_order = i + 1;
        draw.vertex_count = draw.raw_draw_count = 3;
        draw.has_system_inputs = true;
        draw.system_inputs = fixture::mapping(i ? 0 : 30);
        if (!known) draw.system_inputs.parameter_launch = {};
        c.draws.push_back(draw);
        c.operations.push_back({SubmitOperationKind::Draw, i, i + 1, true});
    }
    GpuCapturedOperationFailure failed;
    failed.source_index = 2;
    failed.command_order = 3;
    failed.reason = RealizationFailureReason::ShaderRecompile;
    failed.vertex_retry_config_available = failed.fragment_retry_config_available =
        failed.has_system_inputs = true;
    failed.system_inputs = fixture::mapping(32);
    if (!known) failed.system_inputs.parameter_launch = {};
    c.failure_diagnostics.push_back(failed);
    c.operations.push_back({SubmitOperationKind::Draw, 2, 3, false});
    return c;
}
} // namespace

TEST(PixelParameterInputs, SchemaNineRetainsAndReplaysTheExactProducingLayout) {
    const auto c = compile_case();
    ASSERT_TRUE(c.complete) << c.reason;
    ASSERT_TRUE(c.expected_produced) << c.expected_reject;
    const auto bytes = encode_fragment_compile_case(c);
    EXPECT_EQ(bytes.at(8), 9u);
    const auto loaded = decode_fragment_compile_case(bytes);
    EXPECT_EQ(loaded.system_inputs.parameter_launch, c.system_inputs.parameter_launch);
    EXPECT_EQ(replay_fragment_compile_case(loaded, true), c.source);
}

TEST(PixelParameterInputs, LegacyCaseKeepsUnknownEvenWithRawLaunchObservation) {
    auto c = compile_case(false);
    c.launch_rsrc1 = {true, 0x022c0311u}; // a different raw register grants no RSRC2 authority
    ASSERT_TRUE(c.complete) << c.reason;
    ASSERT_FALSE(c.expected_produced);
    const auto bytes = encode_fragment_compile_case(c);
    EXPECT_EQ(bytes.at(8), 7u);
    const auto loaded = decode_fragment_compile_case(bytes);
    EXPECT_EQ(loaded.system_inputs.parameter_launch, PixelParameterLaunch{});
    EXPECT_EQ(encode_fragment_compile_case(loaded), bytes);
    EXPECT_TRUE(replay_fragment_compile_case(loaded, true).empty());
}

TEST(PixelParameterInputs, CaseRejectsInvalidKnownnessAbsentMappingAndTruncatedTails) {
    auto c = compile_case();
    ASSERT_TRUE(c.complete) << c.reason;
    const auto original = encode_fragment_compile_case(c);
    for (const uint8_t bad_flag : {uint8_t{0}, uint8_t{2}}) {
        auto bad = original;
        bad[bad.size() - 13] = bad_flag;
        rehash(bad);
        EXPECT_THROW(decode_fragment_compile_case(bad), std::runtime_error);
    }
    for (size_t n = 1; n <= 5; ++n) {
        auto bad = original;
        bad.erase(bad.end() - 8 - n, bad.end() - 8);
        rehash(bad);
        EXPECT_THROW(decode_fragment_compile_case(bad), std::runtime_error);
    }
    c.has_system_inputs = false;
    EXPECT_THROW(encode_fragment_compile_case(c), std::runtime_error);
}

TEST(PixelParameterInputs, FrameAndFailedDrawRetainIndependentLayoutsAndZeroKnownness) {
    const auto c = frame();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    EXPECT_EQ(bytes.at(8), 81u);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.draws.size(), 2u);
    ASSERT_EQ(loaded.failure_diagnostics.size(), 1u);
    for (size_t i = 0; i < 2; ++i)
        EXPECT_EQ(loaded.draws[i].system_inputs, c.draws[i].system_inputs);
    EXPECT_TRUE(loaded.draws[1].system_inputs.parameter_launch.available);
    EXPECT_EQ(loaded.draws[1].system_inputs.parameter_launch.value, 0u);
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs, c.failure_diagnostics[0].system_inputs);
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    EXPECT_EQ(replay.items[0].system_inputs, c.draws[0].system_inputs);
}

TEST(PixelParameterInputs, LegacyFramesKeepTheirVersionAndUnknownSystemWord) {
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(frame(false), bytes, error)) << error;
    EXPECT_EQ(bytes.at(8), 71u);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws)
        EXPECT_EQ(draw.system_inputs.parameter_launch, PixelParameterLaunch{});
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs.parameter_launch, PixelParameterLaunch{});
    std::vector<uint8_t> again;
    ASSERT_TRUE(serialize_gpu_capture(loaded, again, error)) << error;
    EXPECT_EQ(again, bytes);
}

TEST(PixelParameterInputs, ScalarOnlyMappingSurvivesTheDeferredVersionedTail) {
    auto c = frame();
    for (auto& draw : c.draws) draw.system_inputs.ena = draw.system_inputs.addr = 0;
    c.failure_diagnostics[0].system_inputs.ena = c.failure_diagnostics[0].system_inputs.addr = 0;
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    EXPECT_EQ(loaded.draws[0].system_inputs, c.draws[0].system_inputs);
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs, c.failure_diagnostics[0].system_inputs);
    c.draws[0].system_inputs.parameter_launch = {};
    EXPECT_FALSE(serialize_gpu_capture(c, bytes, error));
}

TEST(PixelParameterInputs, FrameRejectsBadCountsFlagsUnknownNonzeroAndTruncation) {
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(frame(), bytes, error)) << error;
    constexpr size_t tail_size = 4 + 5 * 2 + 4 + 5;
    const auto tail = bytes.size() - tail_size;
    GpuCaptureFile loaded;
    for (const size_t offset : {tail, tail + 14, tail + 4, tail + 9, tail + 18}) {
        auto bad = bytes;
        bad[offset] = 3;
        EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error));
    }
    auto noncanonical = bytes;
    noncanonical[tail + 4] = 0;
    EXPECT_FALSE(deserialize_gpu_capture(noncanonical, loaded, error));
    for (size_t n = 1; n <= tail_size; ++n) {
        auto bad = bytes;
        bad.resize(bad.size() - n);
        EXPECT_FALSE(deserialize_gpu_capture(bad, loaded, error));
    }
    auto invalid = frame();
    invalid.draws[0].has_system_inputs = false;
    EXPECT_FALSE(serialize_gpu_capture(invalid, bytes, error));
    invalid = frame();
    invalid.failure_diagnostics[0].system_inputs.parameter_launch = {false, 60};
    EXPECT_FALSE(serialize_gpu_capture(invalid, bytes, error));
}
