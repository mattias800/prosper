// Videodec2 publishes linear NV12 planes. Sampling them at GFX10's default 256-byte-aligned
// pitch turns 1920-byte movie rows into stripes and reads beyond the chroma plane (#4852).
// Exercise the guest entry point and the renderer's shared pitch resolver with synthetic pixels.
#include "gpu/resources/linear_row_pitch.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/guest_texture_layout.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/video/video_backend.hpp"
#include "hle/video/videodec2_guest_abi.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper;
using namespace prosper::test::vdec;

uint64_t address(const void* ptr) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
}

class LayoutBackend final : public video::VideoBackend {
public:
    int open(const std::string&) override { return -1; }
    bool info(int, video::StreamInfo&) override { return false; }
    bool next_video(int, video::VideoFrame&) override { return false; }
    bool next_audio(int, video::AudioFrame&) override { return false; }
    bool eof(int) override { return true; }
    void close(int) override {}
    int open_decoder(uint32_t) override { return 1; }
    AuResult decode_au(int, const uint8_t*, size_t, uint8_t* dst, uint64_t bytes,
                       AuPicture& out) override {
        ++calls;
        if (no_picture) return AuResult::NoPicture;
        out.width = width;
        out.height = height;
        out.y_stride = width;
        out.uv_stride = 2 * ((width + 1) / 2);
        const uint64_t y_bytes = uint64_t{out.y_stride} * height;
        out.nv12_bytes = y_bytes + uint64_t{out.uv_stride} * ((height + 1) / 2);
        if (bytes < out.nv12_bytes) return AuResult::FrameTooSmall;
        for (uint32_t y = 0; y < height; ++y)
            std::fill_n(dst + uint64_t{y} * out.y_stride, out.y_stride,
                        static_cast<uint8_t>(20 + y));
        for (uint32_t y = 0; y < (height + 1) / 2; ++y)
            std::fill_n(dst + y_bytes + uint64_t{y} * out.uv_stride, out.uv_stride,
                        static_cast<uint8_t>(100 + y));
        return AuResult::Decoded;
    }
    uint32_t width = 1920, height = 1088;
    unsigned calls = 0;
    bool no_picture = false;
};

class Videodec2Layout : public testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        create = Hle::lookup("CNNRoRYd8XI");
        decode = Hle::lookup("852F5+q6+iM");
        destroy = Hle::lookup("jwImxXRGSKA");
        ASSERT_TRUE(create && decode && destroy);
        video::set_backend(&backend);
        VdecConfig config{};
        config.size = sizeof config;
        config.resource = config.codec = 1;
        config.profile = 100;
        config.max_level = 41;
        config.max_width = 1920;
        config.max_height = 1088;
        config.max_dpb = -1;
        config.input_depth = 4;
        config.compute_queue = address(workspace.data());
        std::array<uint64_t, 9> memory{72,
                                       65536,
                                       address(workspace.data()),
                                       65536,
                                       address(workspace.data()),
                                       65536,
                                       address(workspace.data()),
                                       65536,
                                       256};
        ASSERT_EQ(create(address(&config), address(memory.data()), address(&handle), 0, 0, 0), 0u);
        pixels.resize(video::nv12_bytes(1920, 1088) + 256, 0xcd);
        input = {sizeof input, address(au.data()), au.size(), 0, 0, 0};
        frame = {sizeof frame, address(pixels.data()), pixels.size(), 0, {}};
        output.size = sizeof output;
    }
    void TearDown() override {
        if (handle && destroy) destroy(handle, 0, 0, 0, 0, 0);
        gpu::unregister_guest_linear_texture_layouts_in(address(pixels.data()), pixels.size());
        video::set_backend(nullptr);
    }
    uint64_t submit() {
        return decode(handle, address(&input), address(&frame), address(&output), 0, 0);
    }
    gpu::ShaderResource plane(uint64_t base) {
        gpu::ShaderResource resource{};
        resource.gpu_addr = base;
        return resource;
    }
    LayoutBackend backend;
    HleFn create = nullptr, decode = nullptr, destroy = nullptr;
    uint64_t handle = 0;
    std::array<uint8_t, 65536> workspace{};
    std::array<uint8_t, 4> au{};
    std::vector<uint8_t> pixels;
    VdecInput input{};
    VdecFrame frame{};
    VdecOutput output{};
};

TEST_F(Videodec2Layout, Tight1920RowsReachBothSampledPlanes) {
    auto y = plane(frame.data);
    auto uv = plane(frame.data + 1920ull * 1088);
    ASSERT_EQ(gpu::resolved_linear_row_pitch(y, 1920, 1), 2048u)
        << "positive control: an ordinary unregistered GFX10 surface is padded";
    ASSERT_EQ(submit(), 0u);
    ASSERT_EQ(output.pictures, 1u);
    EXPECT_EQ(output.pitch_bytes, 1920u);
    const uint32_t y_pitch = gpu::resolved_linear_row_pitch(y, 1920, 1);
    const uint32_t uv_pitch = gpu::resolved_linear_row_pitch(uv, 960, 2);
    ASSERT_EQ(y_pitch, 1920u);
    ASSERT_EQ(uv_pitch, 1920u);
    EXPECT_EQ(pixels[uint64_t{y_pitch} * 1087], static_cast<uint8_t>(20 + 1087));
    EXPECT_EQ(pixels[1920ull * 1088 + uint64_t{uv_pitch} * 543], static_cast<uint8_t>(100 + 543));
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data + video::nv12_bytes(1920, 1088), 1),
              0u)
        << "allocation slack is not picture data";
}

TEST_F(Videodec2Layout, OddWidthPublishesDifferentPlanePitches) {
    backend.width = 65;
    backend.height = 33;
    ASSERT_EQ(submit(), 0u);
    EXPECT_EQ(output.pitch_bytes, 65u);
    EXPECT_EQ(gpu::resolved_linear_row_pitch(plane(frame.data), 65, 1), 65u);
    EXPECT_EQ(gpu::resolved_linear_row_pitch(plane(frame.data + 65ull * 33), 33, 2), 66u);
}

TEST_F(Videodec2Layout, ReplacingPictureRetiresPreviousPlaneBoundary) {
    backend.width = 65;
    backend.height = 33;
    ASSERT_EQ(submit(), 0u);
    const uint64_t old_uv = frame.data + 65ull * 33;
    backend.width = 1920;
    backend.height = 1088;
    ASSERT_EQ(submit(), 0u);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(old_uv, 1920), 1920u)
        << "the old UV interval must not shadow the new larger luma plane";
}

TEST_F(Videodec2Layout, SmallerPictureRetiresOldChromaLayout) {
    ASSERT_EQ(submit(), 0u);
    const uint64_t old_uv = frame.data + 1920ull * 1088;
    ASSERT_EQ(gpu::guest_linear_texture_row_pitch(old_uv, 1920), 1920u);
    backend.width = 65;
    backend.height = 33;
    ASSERT_EQ(submit(), 0u);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(old_uv, 1920), 0u)
        << "unused destination-buffer bytes must not retain the replaced picture's pitch";
    EXPECT_EQ(gpu::resolved_linear_row_pitch(plane(frame.data), 65, 1), 65u);
    EXPECT_EQ(gpu::resolved_linear_row_pitch(plane(frame.data + 65ull * 33), 33, 2), 66u);
}

TEST_F(Videodec2Layout, NoPictureDoesNotPublishLayout) {
    backend.no_picture = true;
    ASSERT_EQ(submit(), 0u);
    EXPECT_EQ(output.pictures, 0u);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data, 1920), 0u);
}

TEST_F(Videodec2Layout, UndersizedFrameDoesNotPublishLayoutOrPixels) {
    frame.data_size = 1;
    ASSERT_EQ(submit(), 0u);
    EXPECT_EQ(output.pictures, 0u);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data, 1920), 0u);
    EXPECT_EQ(pixels.front(), 0xcd);
}

TEST_F(Videodec2Layout, BadArgumentsPreserveBufferAndDoNotCallBackend) {
    EXPECT_EQ(decode(handle, 0, address(&frame), address(&output), 0, 0), 0x811d0102u);
    frame.size = 1;
    EXPECT_EQ(submit(), 0x811d0101u);
    frame.size = sizeof frame;
    frame.data_size = 0;
    EXPECT_EQ(submit(), 0x811d0106u);
    frame.data_size = pixels.size();
    frame.data = 0;
    EXPECT_EQ(submit(), 0x811d0107u);
    EXPECT_EQ(backend.calls, 0u);
    EXPECT_EQ(pixels.front(), 0xcd);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(address(pixels.data()), 1920), 0u);
}

TEST_F(Videodec2Layout, DecoderDeletionPreservesGuestOwnedPictureUntilUnmap) {
    ASSERT_EQ(submit(), 0u);
    ASSERT_EQ(destroy(handle, 0, 0, 0, 0, 0), 0u);
    handle = 0;
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data, 1920), 1920u);
    gpu::unregister_guest_linear_texture_layouts_in(frame.data, pixels.size());
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data, 1920), 0u);
    EXPECT_EQ(gpu::guest_linear_texture_row_pitch(frame.data + 1920ull * 1088, 1920), 0u);
}
} // namespace
