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
}   // namespace

TEST(RasterLaunchCapture, CurrentRoundTripAndReplayKeepPerDrawRawKnownness) {
    const auto capture = capsule();
    std::vector<uint8_t> bytes;
    std::string error;
    ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
    ASSERT_GT(bytes.size(), tail_bytes(capture));
    EXPECT_EQ(bytes[8], 71);
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
    bytes.resize(bytes.size() - tail_bytes(capture));
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
    const size_t base = bytes.size() - tail_bytes(capture) + 4;
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
    const size_t base = bytes.size() - tail_bytes(capture) + 4;
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
    const size_t at = bytes.size() - tail_bytes(capture);
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
    for (size_t removed = 1; removed <= tail_bytes(capture); ++removed) {
        auto truncated = bytes;
        truncated.resize(truncated.size() - removed);
        GpuCaptureFile rejected;
        EXPECT_FALSE(deserialize_gpu_capture(truncated, rejected, error)) << removed;
        EXPECT_FALSE(error.empty());
    }
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

TEST(RasterLaunchCapture, AFragmentLaunchWidthDecidesWhetherReplayNeedsWaveOwners) {
    // #4555. Replay re-derives from the ORIGINAL program whether a draw needed the owned-wave
    // path, and refuses a capsule that has no owners for one that did. That question depends on
    // the fragment launch width for a program like this one (MOUSE: P.I. For Hire's shape: the
    // loaded pair is recycled by a compare and then read by a 64-bit mask operation whose SCC
    // is branched on), and the live path now routes it natively at 64 lanes. Replay has to ask
    // with the width the capsule recorded, or every such frame grab would be refused.
    //   0  v_mov_b32 v1, 0
    //   1  s_load_dwordx4 s[16:19], s[28:29], 0xf0
    //   3  s_buffer_load_dwordx4 s[8:11], s[16:19], 0xc0
    //   5  v_cmp_*_sdwa s[16:17], 0, s10 ; s_mov_b64 vcc, s[16:17]
    //   8  s_and_b64 s[40:41], s[16:17], exec ; s_cbranch_scc1 +0
    //  10  s_cbranch_vccz +1 ; s_branch 1
    //  12  v_readfirstlane vcc_lo, v1 ; s_endpgm
    const std::vector<uint32_t> fragment{
        0x7e020280u, 0xf408040eu, 0xfa0000f0u, 0xf4280208u, 0xfa0000c0u, 0x7c1a14f9u, 0x86869080u,
        0xbeea0410u, 0x87a87e10u, 0xbf850000u, 0xbf860001u, 0xbf82fff5u, 0x7ed40501u, 0xbf810000u,
    };
    auto capture = capsule();
    capture.draws.resize(1);
    capture.operations.resize(1);
    GpuCaptureRawShaderVersion raw;
    raw.words = fragment;
    raw.content_hash = gpu_capture_hash(reinterpret_cast<const uint8_t*>(raw.words.data()),
                                        raw.words.size() * sizeof(uint32_t));
    raw.has_endpgm = true;
    capture.raw_shader_versions.push_back(raw);
    capture.draws[0].fs_raw_shader_index = 0;
    capture.draws[0].fragment_wave_config_available = true;
    const auto materialize = [&](std::string& error) {
        std::vector<uint8_t> bytes;
        GpuCaptureFile loaded;
        GpuReplayFrame replay;
        if (!serialize_gpu_capture(capture, bytes, error)) return false;
        if (!deserialize_gpu_capture(bytes, loaded, error)) return false;
        return materialize_gpu_replay(loaded, replay, error);
    };
    std::string error;
    capture.draws[0].ps_wave32 = false;
    EXPECT_TRUE(materialize(error)) << error;
    capture.draws[0].ps_wave32 = true;
    EXPECT_FALSE(materialize(error));
    EXPECT_EQ(error, "logical-wave replay lacks original exact-PC window owners stage=fs pc=1");
    // A capsule that never recorded the launch width gets the default answer, as before.
    capture.draws[0].ps_wave32 = false;
    capture.draws[0].fragment_wave_config_available = false;
    EXPECT_FALSE(materialize(error));
    EXPECT_EQ(error, "logical-wave replay lacks original exact-PC window owners stage=fs pc=1");
}
