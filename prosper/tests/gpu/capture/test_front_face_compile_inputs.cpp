// #5052: replay must retain the actual FRONT_FACE compile input, including its
// absence. Old records cannot acquire a mode from defaults or today's raster state.
#include "build_revision.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include <gtest/gtest.h>
#include <stdexcept>

namespace prosper::gpu {
namespace {
FragmentCompileCase compile_case(FragmentFrontFaceEncoding encoding) {
    FragmentCompileCase c;
    c.compiler = std::string(prosper::embedded_build_revision()) + ':' +
                 prosper::embedded_build_source_identity();
    c.program_address = 0x10005000;
    c.code = {0x7d040080u, 0x7e120280u, 0x021012f2u, 0x7e140280u,
              0x7e1602f2u, 0xf800180fu, 0x0b0a0908u, 0xbf810000u};
    c.has_system_inputs = true;
    c.system_inputs = {0x1000u, 0x1000u, encoding};
    c.wave32 = true;
    c.interpolation = fragment_interpolation_layout(c.code.data(), c.code.size(), &c.system_inputs);
    std::vector<uint32_t> source;
    {
        CompilerChoiceScope choices(c.choices);
        TerminalRejectCapture rejection;
        source = recompile_fragment(c.code.data(), c.code.size(), nullptr, &c.system_inputs,
                                    UINT32_MAX, &c.interpolation, c.wave32,
                                    {RecompileDiagnosticStage::Fragment, c.program_address});
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
    for (uint32_t i = 0; i < 8; ++i) bytes[bytes.size() - 8 + i] = uint8_t(hash >> (8 * i));
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
    for (uint32_t i = 0; i < 2; ++i) {
        GpuCapturedDraw draw;
        draw.vs = stored_stage(0);
        draw.fs = stored_stage(4);
        draw.draw_index = 7 + i;
        draw.command_order = 10 + i;
        draw.vertex_count = draw.raw_draw_count = 3;
        draw.has_system_inputs = true;
        draw.system_inputs = {0x1000u, 0x1000u};
        if (known) draw.system_inputs.retain_front_face_encoding(true, i ? (1u << 24) : 0);
        c.draws.push_back(draw);
        c.operations.push_back(
            {SubmitOperationKind::Draw, draw.draw_index, draw.command_order, true});
    }
    GpuCapturedOperationFailure failed;
    failed.source_index = 9;
    failed.command_order = 12;
    failed.reason = RealizationFailureReason::ShaderRecompile;
    failed.vertex_retry_config_available = failed.fragment_retry_config_available =
        failed.has_system_inputs = true;
    failed.system_inputs = {0x1000u, 0x1000u};
    if (known) failed.system_inputs.retain_front_face_encoding(true, 1u << 24);
    c.failure_diagnostics.push_back(failed);
    c.operations.push_back(
        {SubmitOperationKind::Draw, failed.source_index, failed.command_order, false});
    return c;
}
constexpr size_t tail_bytes = 4 + 2 + 4 + 1; // independent v76 count/u8/count/u8 wire contract
} // namespace

TEST(FrontFaceCompileInputs, ObservedRegisterSelectsModeWithoutGrantingDisabledInput) {
    PixelSystemInputMapping input{0x1000u, 0x1000u};
    input.retain_front_face_encoding(true, 0);
    EXPECT_EQ(input.front_face_encoding, FragmentFrontFaceEncoding::SignedFloat);
    input.retain_front_face_encoding(true, 1u << 24);
    EXPECT_EQ(input.front_face_encoding, FragmentFrontFaceEncoding::AllBits);
    input.retain_front_face_encoding(false, UINT32_MAX);
    EXPECT_EQ(input.front_face_encoding, FragmentFrontFaceEncoding::Unavailable);
    for (auto disabled :
         {PixelSystemInputMapping{0, 0x1000u}, PixelSystemInputMapping{0x1000u, 0}}) {
        disabled.retain_front_face_encoding(true, 1u << 24);
        EXPECT_EQ(disabled.front_face_encoding, FragmentFrontFaceEncoding::Unavailable);
    }
}

TEST(FrontFaceCompileInputs, CompleteCompilerCasesRoundTripAndReplayBothModes) {
    for (const auto encoding :
         {FragmentFrontFaceEncoding::SignedFloat, FragmentFrontFaceEncoding::AllBits}) {
        const auto c = compile_case(encoding);
        ASSERT_TRUE(c.complete) << c.reason;
        ASSERT_TRUE(c.expected_produced);
        const auto bytes = encode_fragment_compile_case(c);
        ASSERT_GT(bytes.size(), 9u);
        EXPECT_EQ(bytes[8], 7);
        const auto loaded = decode_fragment_compile_case(bytes);
        EXPECT_EQ(loaded.system_inputs, c.system_inputs);
        EXPECT_EQ(replay_fragment_compile_case(loaded, true), c.source);
    }
}

TEST(FrontFaceCompileInputs, LegacySchemaSixKeepsTheMissingInputMissing) {
    const auto c = compile_case(FragmentFrontFaceEncoding::Unavailable);
    ASSERT_TRUE(c.complete) << c.reason;
    ASSERT_FALSE(c.expected_produced);
    auto bytes = encode_fragment_compile_case(c);
    ASSERT_GT(bytes.size(), 9u);
    bytes.erase(bytes.end() - 9);   // remove the new field, preserving the complete official prefix
    bytes[8] = 6;
    rehash(bytes);
    const auto loaded = decode_fragment_compile_case(bytes);
    EXPECT_EQ(loaded.system_inputs.front_face_encoding, FragmentFrontFaceEncoding::Unavailable);
    EXPECT_TRUE(replay_fragment_compile_case(loaded, true).empty());
}

TEST(FrontFaceCompileInputs, InvalidCaseEncodingAndAbsentMappingAreRejected) {
    auto c = compile_case(FragmentFrontFaceEncoding::AllBits);
    ASSERT_TRUE(c.complete) << c.reason;
    auto bytes = encode_fragment_compile_case(c);
    bytes[bytes.size() - 9] = 3;
    rehash(bytes);
    EXPECT_THROW(decode_fragment_compile_case(bytes), std::runtime_error);
    c.has_system_inputs = false;
    EXPECT_THROW(encode_fragment_compile_case(c), std::runtime_error);
    c.has_system_inputs = true;
    c.system_inputs.ena = 0;
    EXPECT_THROW(encode_fragment_compile_case(c), std::runtime_error);
    c.system_inputs.ena = 0x1000u;
    c.system_inputs.addr = 0;
    EXPECT_THROW(encode_fragment_compile_case(c), std::runtime_error);
}

TEST(FrontFaceCompileInputs, FrameAndFailedDrawKeepIndependentModes) {
    const auto c = frame();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    EXPECT_EQ(bytes[8], 76);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.draws.size(), 2u);
    ASSERT_EQ(loaded.failure_diagnostics.size(), 1u);
    EXPECT_EQ(loaded.draws[0].system_inputs, c.draws[0].system_inputs);
    EXPECT_EQ(loaded.draws[1].system_inputs, c.draws[1].system_inputs);
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs, c.failure_diagnostics[0].system_inputs);
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    EXPECT_EQ(replay.items[0].system_inputs, c.draws[0].system_inputs);
    EXPECT_EQ(replay.items[1].system_inputs, c.draws[1].system_inputs);
}

TEST(FrontFaceCompileInputs, UnknownFramesKeepTheLegacyVersionAndDoNotInferMode) {
    const auto c = frame(false);
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(c, bytes, error)) << error;
    EXPECT_EQ(bytes[8], 71);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws)
        EXPECT_EQ(draw.system_inputs.front_face_encoding, FragmentFrontFaceEncoding::Unavailable);
    EXPECT_EQ(loaded.failure_diagnostics[0].system_inputs.front_face_encoding,
              FragmentFrontFaceEncoding::Unavailable);
}

TEST(FrontFaceCompileInputs, FrameEncodingCountsTruncationAndTagsAreChecked) {
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(frame(), bytes, error)) << error;
    ASSERT_GT(bytes.size(), tail_bytes);
    const size_t tail = bytes.size() - tail_bytes;
    GpuCaptureFile rejected;
    for (const size_t offset : {tail + 4, tail + 5, tail + 10}) {
        auto bad = bytes;
        bad[offset] = 3;
        EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
    }
    for (const size_t offset : {tail, tail + 6}) {
        auto bad = bytes;
        bad[offset] = 0;
        EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
    }
    for (size_t n = 1; n <= tail_bytes; ++n) {
        auto bad = bytes;
        bad.resize(bad.size() - n);
        EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
    }
    auto invalid = frame();
    invalid.draws[0].has_system_inputs = false;
    EXPECT_FALSE(serialize_gpu_capture(invalid, bytes, error));
    invalid = frame();
    invalid.failure_diagnostics[0].system_inputs.addr = 0;
    EXPECT_FALSE(serialize_gpu_capture(invalid, bytes, error));
}
}   // namespace prosper::gpu
