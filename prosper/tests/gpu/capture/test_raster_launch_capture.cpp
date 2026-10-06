// Offline launch facts must remain draw-owned observations, never defaults or launch authority.
// Exercise the actual append-only codec and materializer; no Vulkan device or game is involved.
#include "gpu/capture/gpu_capture.hpp"
#include <array>
#include <gtest/gtest.h>

using namespace prosper::gpu;
namespace {
struct Control {
    bool RasterLaunchFacts::* available;
    uint32_t RasterLaunchFacts::* value;
};
constexpr std::array<Control, 5> controls{{
    {&RasterLaunchFacts::sc_shader_control_available, &RasterLaunchFacts::sc_shader_control},
    {&RasterLaunchFacts::sc_mode_cntl_0_available, &RasterLaunchFacts::sc_mode_cntl_0},
    {&RasterLaunchFacts::sc_mode_cntl_1_available, &RasterLaunchFacts::sc_mode_cntl_1},
    {&RasterLaunchFacts::sc_aa_config_available, &RasterLaunchFacts::sc_aa_config},
    {&RasterLaunchFacts::db_shader_control_available, &RasterLaunchFacts::db_shader_control},
}};
std::vector<uint32_t> stored_stage(uint32_t stage) {
    std::vector<uint32_t> words{
        0x07230203, 0x00010000, 0, 5,          0, 0x0003000e, 0, 1, // Logical GLSL450
        0x0005000f, stage,      1, 0x6e69616d, 0, 0x00020013, 2, 0x00030021, 3,         2,
        0x00050036, 2,          1, 0,          3, 0x000200f8, 4, 0x000100fd, 0x00010038};
    if (stage == 4) words.insert(words.begin() + 13, {0x00030010, 1, 7});
    return words;
}
GpuCaptureFile capsule() {
    GpuCaptureFile capture;
    capture.metadata.title_id = "PPSA00000";
    capture.metadata.width = capture.metadata.height = 8;
    for (uint32_t i = 0; i < 2; ++i) {
        GpuCapturedDraw draw;
        draw.vs = stored_stage(0);
        draw.fs = stored_stage(4);
        draw.vertex_count = draw.raw_draw_count = 3;
        draw.draw_index = 7 + i;
        draw.command_order = 10 + i;
        auto& launch = draw.ps_raster_launch;
        launch.ps_in_control_available = launch.baryc_cntl_available = true;
        launch.input_ena_available = launch.input_addr_available = true;
        launch.ps_in_control = 0x12345678;
        launch.baryc_cntl = 0x87654321;
        launch.input_ena = 0;
        launch.input_addr = 0x300;
        if (!i) {
            for (size_t j = 0; j < controls.size(); ++j) {
                launch.*controls[j].available = true;
                launch.*controls[j].value = j == 1 ? 0u : 0x81234567u + uint32_t(j);
            }
        } else {
            launch.db_shader_control_available = true; // independent observed zero
        }
        capture.draws.push_back(draw);
        capture.operations.push_back(
            {SubmitOperationKind::Draw, draw.draw_index, draw.command_order, true});
    }
    return capture;
}
constexpr size_t record_bytes = 25; // independent wire oracle: five (u8 present, u32 raw) words
size_t tail_bytes(const GpuCaptureFile& capture) {
    return 4 + record_bytes * capture.draws.size();
}
size_t coverage_tail_bytes(const GpuCaptureFile& capture) {
    return 4 + 112 * capture.draws.size(); // u32 presence + 27 complete raw words per draw
}
}   // namespace

TEST(RasterLaunchCapture, CurrentRoundTripAndReplayKeepPerDrawRawKnownness) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    ASSERT_GT(bytes.size(), tail_bytes(capture));
    EXPECT_EQ(bytes[8], 72);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.draws.size(), 2u);
    for (size_t i = 0; i < 2; ++i)
        EXPECT_EQ(loaded.draws[i].ps_raster_launch, capture.draws[i].ps_raster_launch);
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(replay.items[i].ps_raster_launch, capture.draws[i].ps_raster_launch);
        EXPECT_FALSE(replay.items[i].raster_quads);
    }
}

TEST(RasterLaunchCapture, LegacyV70RetainsSpiButNeverInventsNewControls) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    ASSERT_GT(bytes.size(), tail_bytes(capture) + 16);
    bytes.resize(bytes.size() - coverage_tail_bytes(capture) - tail_bytes(capture));
    bytes[8] = 70; // unchanged complete official v70 prefix, only version header lowered
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    EXPECT_EQ(loaded.format_version, 70u);
    ASSERT_EQ(loaded.draws.size(), 2u);
    for (const auto& draw : loaded.draws) {
        const auto& launch = draw.ps_raster_launch;
        EXPECT_TRUE(launch.ps_in_control_available && launch.baryc_cntl_available &&
                    launch.input_ena_available && launch.input_addr_available);
        EXPECT_EQ(launch.ps_in_control, 0x12345678u);
        EXPECT_EQ(launch.baryc_cntl, 0x87654321u);
        EXPECT_EQ(launch.input_ena, 0u);
        EXPECT_EQ(launch.input_addr, 0x300u);
        for (const auto& control : controls) {
            EXPECT_FALSE(launch.*control.available);
            EXPECT_EQ(launch.*control.value, 0u);
        }
        EXPECT_TRUE(launch.canonical());
    }
}

TEST(RasterLaunchCapture, EveryNewFlagRejectsNonBooleanWireValues) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    const size_t base = bytes.size() - coverage_tail_bytes(capture) - tail_bytes(capture) + 4;
    for (size_t draw = 0; draw < capture.draws.size(); ++draw) {
        for (size_t word = 0; word < controls.size(); ++word) {
            auto malformed = bytes;
            malformed[base + record_bytes * draw + 5 * word] = 2;
            GpuCaptureFile rejected;
            EXPECT_FALSE(deserialize_gpu_capture(malformed, rejected, error));
            EXPECT_EQ(error, "invalid realized-draw raster launch controls");
        }
    }
}

TEST(RasterLaunchCapture, UnavailableRawPayloadIsRejectedOnBothSidesOfCodec) {
    for (const auto& control : controls) {
        auto capture = capsule();
        auto& launch = capture.draws[1].ps_raster_launch;
        launch.*control.available = false;
        launch.*control.value = 1;
        std::vector<uint8_t> bytes{0xab};
        std::string error;
        EXPECT_FALSE(serialize_gpu_capture(capture, bytes, error));
        EXPECT_EQ(error, "invalid realized-draw fragment entry evidence");
        EXPECT_EQ(bytes, (std::vector<uint8_t>{0xab}));
    }
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    const size_t base = bytes.size() - coverage_tail_bytes(capture) - tail_bytes(capture) + 4;
    for (size_t word = 0; word < controls.size(); ++word) {
        auto malformed = bytes;
        const size_t at = base + 5 * word;
        malformed[at] = 0;
        malformed[at + 1] = 1;
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(malformed, rejected, error));
        EXPECT_EQ(error, "invalid realized-draw raster launch controls");
    }
}

TEST(RasterLaunchCapture, DrawCountCannotBorrowAnotherDrawsControlWords) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    const size_t at = bytes.size() - coverage_tail_bytes(capture) - tail_bytes(capture);
    for (uint32_t count : {0u, 1u, 3u, UINT32_MAX}) {
        auto malformed = bytes;
        for (uint32_t byte = 0; byte < 4; ++byte)
            malformed[at + byte] = uint8_t(count >> (8 * byte));
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(malformed, rejected, error));
        EXPECT_EQ(error, "invalid realized-draw raster launch controls count");
    }
}

TEST(RasterLaunchCapture, EveryTruncatedNewTailIsRejected) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    bytes.resize(bytes.size() - coverage_tail_bytes(capture));
    bytes[8] =
        71; // the complete historical v71 frame; test its own tail, not only v72 truncation
    for (size_t removed = 1; removed <= tail_bytes(capture); ++removed) {
        auto truncated = bytes;
        truncated.resize(truncated.size() - removed);
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(truncated, rejected, error)) << removed;
        EXPECT_FALSE(error.empty());
    }
}

TEST(RasterLaunchCapture, CoverageRoundTripRetainsAllWordsAndLegacyAbsence) {
    auto capture = capsule();
    auto& coverage = capture.draws[0].ps_raster_launch.coverage;
    coverage.available = 0x07ffffffu;
    for (uint32_t index = 0; index < 27; ++index)
        coverage.words[index] = index == 5 ? 0u : 0x80001000u + index;
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.draws.size(), 2u);
    EXPECT_EQ(loaded.draws[0].ps_raster_launch.coverage, coverage);
    EXPECT_EQ(loaded.draws[1].ps_raster_launch.coverage, RasterCoverageFacts{});
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    EXPECT_EQ(replay.items[0].ps_raster_launch.coverage, coverage);
    bytes.resize(bytes.size() - coverage_tail_bytes(capture));
    bytes[8] = 71;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws)
        EXPECT_EQ(draw.ps_raster_launch.coverage, RasterCoverageFacts{});
}

TEST(RasterLaunchCapture, CoverageRejectsUnknownAbsentAndTruncatedWire) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    const size_t base = bytes.size() - coverage_tail_bytes(capture);
    for (size_t at : {base + 4 + 3, base + 8}) {
        auto corrupt = bytes;
        corrupt[at] = 0x80; // unknown presence bit, or nonzero unobserved raw payload
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(corrupt, rejected, error));
        EXPECT_EQ(error, "invalid realized-draw raster coverage");
    }
    for (size_t removed = 1; removed <= coverage_tail_bytes(capture); ++removed) {
        auto truncated = bytes;
        truncated.resize(truncated.size() - removed);
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(truncated, rejected, error)) << removed;
        EXPECT_FALSE(error.empty());
    }
    auto corrupt = bytes;
    corrupt[base] = 1; // two draws cannot consume a one-draw tail
    GpuCaptureFile rejected;
    EXPECT_FALSE(deserialize_gpu_capture(corrupt, rejected, error));
    EXPECT_EQ(error, "invalid realized-draw raster coverage count");
}

TEST(RasterLaunchCapture, CompleteNewTailCannotHideTrailingGarbage) {
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capsule(), bytes, error)) << error;
    bytes.push_back(0);
    GpuCaptureFile rejected;
    EXPECT_FALSE(deserialize_gpu_capture(bytes, rejected, error));
    EXPECT_EQ(error, "capture has trailing data");
}
