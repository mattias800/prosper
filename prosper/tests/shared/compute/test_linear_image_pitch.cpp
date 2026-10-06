// test_linear_image_pitch -- which linear images the compute backend must read row by row.
//
// THE DEFECT. GFX10 pads a linear image's rows to 256 bytes (a 1920-wide R8 plane is 2048 bytes per
// row). The graphics upload honoured that; the compute upload read the rows as tight, so each row
// started 128 bytes early and the padding showed as a diagonal band. Black Flag's warning-screen
// video planes (1920x1080 R8) came out in green stripes.
//
// WHAT EACH TEST KILLS:
//   PaddedPlaneHasTheAlignedPitch   the 256-byte rule is not applied (the original defect)
//   AlignedRowsAreTight             an already-aligned width gets a pitch, repacking needlessly
//   ExplicitAndRegisteredPitchWin   a pitch the descriptor or an HLE producer states is overridden
//   OnlyPlainLinear2dQualifies      the rule leaks onto tiled, layered, mipped, compressed or
//                                   host-owned images, which have their own layouts
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

TEST(LinearImagePitch, PaddedPlaneHasTheAlignedPitch) {
    EXPECT_EQ(compute_linear_row_pitch(r8_plane(1920), 1), 2048u);
    ShaderResource rgba = r8_plane(100);
    rgba.num_components = 4;
    EXPECT_EQ(compute_linear_row_pitch(rgba, 4), 512u) << "400 bytes round up to 512";
    ShaderResource one_layer_array = r8_plane(1920);
    one_layer_array.img_dim = 5;
    EXPECT_EQ(compute_linear_row_pitch(one_layer_array, 1), 2048u);
}

TEST(LinearImagePitch, AlignedRowsAreTight) {
    EXPECT_EQ(compute_linear_row_pitch(r8_plane(1024), 1), 0u);
    EXPECT_EQ(compute_linear_row_pitch(r8_plane(3840), 1), 0u) << "3840 bytes is 15 x 256";
}

TEST(LinearImagePitch, ExplicitAndRegisteredPitchWin) {
    ShaderResource explicit_pitch = r8_plane(1920);
    explicit_pitch.linear_row_pitch_bytes = 4096;
    EXPECT_EQ(compute_linear_row_pitch(explicit_pitch, 1), 4096u);

    ShaderResource tight_declared = r8_plane(1920);
    tight_declared.linear_row_pitch_bytes = 1920;
    EXPECT_EQ(compute_linear_row_pitch(tight_declared, 1), 0u)
        << "an explicit tight pitch is tight";

    constexpr uint64_t base = 0x7000000000ull;
    register_guest_linear_texture_layout(base, 1u << 20, 3072);
    ShaderResource registered = r8_plane(1920);
    registered.gpu_addr = base + 0x1000;
    EXPECT_EQ(compute_linear_row_pitch(registered, 1), 3072u)
        << "an HLE-registered pitch outranks the rule";
    unregister_guest_linear_texture_layout(base);
    EXPECT_EQ(compute_linear_row_pitch(registered, 1), 2048u);
}

TEST(LinearImagePitch, OnlyPlainLinear2dQualifies) {
    auto expect_tight = [](const char* why, auto&& edit) {
        ShaderResource r = r8_plane(1920);
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
    static uint8_t bytes[4] = {};
    expect_tight("replay-owned bytes keep their capture's pitch",
                 [](ShaderResource& r) { r.host_data = bytes; });
    expect_tight("zero width", [](ShaderResource& r) { r.width = 0; });
}
