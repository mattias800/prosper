// test_linear_image_pitch -- which linear images the compute backend must read row by row.
//
// THE DEFECT. AvPlayer publishes 1920-wide R8 planes at a 2048-byte row pitch. The compute upload
// read the rows as tight, so each row started 128 bytes early and the padding showed as a diagonal
// band (Black Flag's warning-screen video planes came out in green stripes). The pitch is honoured
// only where the guest STATES it; it is never inferred from the 256-byte alignment (#4606).
//
// WHAT EACH TEST KILLS:
//   UnstatedPitchIsTight            the 256-byte alignment is inferred for an image nobody padded,
//                                   giving it a layout renderer targets and the presenter do not use
//   RegisteredPitchIsHonoured       an HLE registration (AvPlayer) is ignored: the original defect
//   DescriptorPitchIsHonoured       the descriptor's own pitch is ignored or loses to a registration
//   OnlyPlainLinear2dQualifies      a stated pitch leaks onto tiled, layered, mipped or compressed
//                                   images, which have their own layouts
#include "shared/compute/linear_image_pitch.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;
using prosper::frontend::compute_linear_row_pitch;

namespace {

ShaderResource r8_plane(uint32_t width) {
    ShaderResource r;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 1;
    r.width = width;
    r.height = 1080;
    r.depth = 1;
    r.img_dim = 1;
    r.declared_mip_levels = 1;
    r.tile_mode = 0;
    r.gpu_addr = 0x4227758000ull;
    return r;
}

}   // namespace

TEST(LinearImagePitch, UnstatedPitchIsTight) {
    // No inference from the GFX10 256-byte alignment: an unregistered 1920-wide plane is tight.
    EXPECT_EQ(compute_linear_row_pitch(r8_plane(1920), 1), 0u);
    ShaderResource rgba = r8_plane(100);
    rgba.num_components = 4;
    EXPECT_EQ(compute_linear_row_pitch(rgba, 4), 0u) << "400-byte rows with no stated pitch";
}

TEST(LinearImagePitch, RegisteredPitchIsHonoured) {
    constexpr uint64_t base = 0x7000000000ull;
    register_guest_linear_texture_layout(base, 1u << 20, 2048);
    ShaderResource plane = r8_plane(1920);
    plane.gpu_addr = base + 0x1000;
    EXPECT_EQ(compute_linear_row_pitch(plane, 1), 2048u) << "an AvPlayer-style registration";
    ShaderResource one_layer_array = plane;
    one_layer_array.img_dim = 5;
    EXPECT_EQ(compute_linear_row_pitch(one_layer_array, 1), 2048u);
    ShaderResource wider = r8_plane(1920);
    wider.gpu_addr = base;
    wider.num_components = 2;   // 3840-byte rows do not fit a 2048-byte registered pitch
    EXPECT_EQ(compute_linear_row_pitch(wider, 2), 0u);
    unregister_guest_linear_texture_layout(base);
    EXPECT_EQ(compute_linear_row_pitch(plane, 1), 0u) << "the pitch goes with the registration";
}

TEST(LinearImagePitch, DescriptorPitchIsHonoured) {
    ShaderResource explicit_pitch = r8_plane(1920);
    explicit_pitch.linear_row_pitch_bytes = 4096;
    EXPECT_EQ(compute_linear_row_pitch(explicit_pitch, 1), 4096u);
    static uint8_t bytes[4] = {};
    explicit_pitch.host_data = bytes;
    EXPECT_EQ(compute_linear_row_pitch(explicit_pitch, 1), 4096u)
        << "a replay resource keeps its capture-resolved pitch";

    ShaderResource tight_declared = r8_plane(1920);
    tight_declared.linear_row_pitch_bytes = 1920;
    EXPECT_EQ(compute_linear_row_pitch(tight_declared, 1), 0u)
        << "an explicit tight pitch is tight";

    constexpr uint64_t base = 0x7100000000ull;
    register_guest_linear_texture_layout(base, 1u << 20, 3072);
    ShaderResource both = r8_plane(1920);
    both.gpu_addr = base;
    both.linear_row_pitch_bytes = 4096;
    EXPECT_EQ(compute_linear_row_pitch(both, 1), 4096u) << "the descriptor outranks a registration";
    unregister_guest_linear_texture_layout(base);
}

TEST(LinearImagePitch, OnlyPlainLinear2dQualifies) {
    auto expect_tight = [](const char* why, auto&& edit) {
        ShaderResource r = r8_plane(1920);
        r.linear_row_pitch_bytes = 2048;   // a stated pitch the shape must still refuse
        edit(r);
        EXPECT_EQ(compute_linear_row_pitch(r, 1), 0u) << why;
    };
    expect_tight("tiled", [](ShaderResource& r) { r.tile_mode = 27; });
    expect_tight("cube", [](ShaderResource& r) { r.img_dim = 3; });
    expect_tight("array", [](ShaderResource& r) {
        r.img_dim = 5;
        r.depth = 4;
    });
    expect_tight("layer stride", [](ShaderResource& r) { r.layer_stride_bytes = 4096; });
    expect_tight("mip chain", [](ShaderResource& r) { r.declared_mip_levels = 3; });
    expect_tight("mip tail", [](ShaderResource& r) { r.in_mip_tail = true; });
    expect_tight("compressed", [](ShaderResource& r) { r.compression_enabled = true; });
    expect_tight("multisampled", [](ShaderResource& r) { r.sample_count = 4; });
    expect_tight("block compressed", [](ShaderResource& r) { r.format = DataFormat::Bc1; });
    expect_tight("buffer", [](ShaderResource& r) { r.cls = ResourceClass::ConstantBuffer; });
    expect_tight("zero width", [](ShaderResource& r) { r.width = 0; });
}
