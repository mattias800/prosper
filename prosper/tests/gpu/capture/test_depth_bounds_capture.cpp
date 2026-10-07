// Capture v73: a realized draw's depth-bounds test survives serialization and replay
// materialization, so an offline replay of a frame that bounds its shadow cascades or light volumes
// partitions them exactly as the live renderer did. Only a capture that uses the test becomes v73;
// every other capture keeps its v71 bytes.
#include "gpu/capture/gpu_capture.hpp"
#include <cstring>
#include <gtest/gtest.h>

using namespace prosper::gpu;
namespace {
std::vector<uint32_t> stored_stage(uint32_t stage) {
    std::vector<uint32_t> words{
        0x07230203, 0x00010000, 0, 5,          0, 0x0003000e, 0, 1, // Logical GLSL450
        0x0005000f, stage,      1, 0x6e69616d, 0, 0x00020013, 2, 0x00030021, 3,         2,
        0x00050036, 2,          1, 0,          3, 0x000200f8, 4, 0x000100fd, 0x00010038};
    if (stage == 4) words.insert(words.begin() + 13, {0x00030010, 1, 7});
    return words;
}
GpuCaptureFile capsule(bool bounded) {
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
        if (bounded && i == 1) {
            draw.ps.depth_bounds_enable = true;
            draw.ps.depth_bounds_min = 0.25f;
            draw.ps.depth_bounds_max = 0.75f;
        }
        capture.draws.push_back(draw);
        capture.operations.push_back(
            {SubmitOperationKind::Draw, draw.draw_index, draw.command_order, true});
    }
    return capture;
}
// Independent wire oracle for the v73 tail: u32 count, then per draw (u8 enable, f32 min, f32 max).
constexpr size_t kRecordBytes = 9;
size_t tail_bytes(const GpuCaptureFile& capture) { return 4 + kRecordBytes * capture.draws.size(); }
}   // namespace

TEST(DepthBoundsCapture, RoundTripAndReplayKeepTheBounds) {
    const auto capture = capsule(true);
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    EXPECT_EQ(bytes[8], 73);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    ASSERT_EQ(loaded.draws.size(), 2u);
    EXPECT_FALSE(loaded.draws[0].ps.depth_bounds_enable);
    EXPECT_TRUE(loaded.draws[1].ps.depth_bounds_enable);
    EXPECT_EQ(loaded.draws[1].ps.depth_bounds_min, 0.25f);
    EXPECT_EQ(loaded.draws[1].ps.depth_bounds_max, 0.75f);
    GpuReplayFrame replay;
    ASSERT_TRUE(materialize_gpu_replay(loaded, replay, error)) << error;
    ASSERT_EQ(replay.items.size(), 2u);
    EXPECT_FALSE(replay.items[0].ps.depth_bounds_enable);
    EXPECT_TRUE(replay.items[1].ps.depth_bounds_enable);
    EXPECT_EQ(replay.items[1].ps.depth_bounds_min, 0.25f);
    EXPECT_EQ(replay.items[1].ps.depth_bounds_max, 0.75f);
}

TEST(DepthBoundsCapture, AnUnboundedCaptureStaysV71) {
    const auto capture = capsule(false);
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    EXPECT_EQ(bytes[8], 71);
    GpuCaptureFile loaded;
    ASSERT_TRUE(deserialize_gpu_capture(bytes, loaded, error)) << error;
    for (const auto& draw : loaded.draws) EXPECT_FALSE(draw.ps.depth_bounds_enable);
}

TEST(DepthBoundsCapture, MalformedRecordsAreRefused) {
    const auto capture = capsule(true);
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    const size_t record = bytes.size() - tail_bytes(capture) + 4 + kRecordBytes;   // draw 1
    auto bad = bytes;
    bad[record] = 2;   // enable is a boolean
    GpuCaptureFile rejected;
    EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
    EXPECT_EQ(error, "invalid depth-bounds draw state");
    bad = bytes;
    const float out_of_range = 1.5f;   // the resolver clamps into [0, 1]
    std::memcpy(&bad[record + 5], &out_of_range, sizeof out_of_range);
    EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
    EXPECT_EQ(error, "invalid depth-bounds draw state");
    bad = bytes;
    bad.resize(bad.size() - 1);
    EXPECT_FALSE(deserialize_gpu_capture(bad, rejected, error));
}
