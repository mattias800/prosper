// Separately allocated tiled NV12 planes must preserve U and V through the shipping graphics
// upload. Broadcasting the first RG8 byte makes decoded pictures collapse to green/magenta (#4811).
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

using namespace prosper::gpu;

TEST(TiledNv12Upload, SeparatePlanesPreserveUv) {
    prosper::register_builtin_hle();
    prosper::frontend::register_live_renderer(".", false);
    constexpr uint32_t width = 8, height = 8, tile_mode = 9;
    // Synthetic fullscreen triangle and constant-coordinate texture sampler, shared with the
    // established sampled-upload fixtures. The shader exports all four sampled components.
    constexpr uint32_t vs[] = {0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
                               0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
                               0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u};
    constexpr uint32_t ps[] = {0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u, 0xf0800f08u,
                               0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u};
    const std::array<uint8_t, 8> uv = {32, 224, 32, 224, 32, 224, 32, 224};
    const std::array<uint8_t, 16> y = {};
    std::vector<uint8_t> tiled_uv(tiled_surface_bytes(2, 2, tile_mode, 0, 2));
    std::vector<uint8_t> tiled_y(tiled_surface_bytes(4, 4, tile_mode, 0, 1));
    tile_surface(tiled_uv.data(), uv.data(), 2, 2, tile_mode, 0, 2);
    tile_surface(tiled_y.data(), y.data(), 4, 4, tile_mode, 0, 1);

    ShaderResource chroma{};
    chroma.cls = ResourceClass::Texture;
    chroma.format = DataFormat::Unorm8;
    chroma.num_components = 2;
    chroma.binding = 4;
    chroma.sgpr_base = 8;
    chroma.width = chroma.height = 2;
    chroma.img_dim = 5;
    chroma.depth = 1;
    chroma.tile_mode = tile_mode;
    chroma.gpu_addr = 0x900000;
    chroma.size = uv.size();
    chroma.host_data = tiled_uv.data();
    chroma.host_data_size = tiled_uv.size();
    chroma.swizzle[0] = 4;
    chroma.swizzle[1] = 5;
    chroma.swizzle[2] = 0;
    chroma.swizzle[3] = 1;
    auto table = std::make_shared<ShaderResourceTable>();
    table->resources.push_back(chroma);
    DrawItem draw;
    draw.vs = recompile_vertex(vs, std::size(vs));
    draw.fs = recompile_fragment(ps, std::size(ps), table.get());
    ASSERT_FALSE(draw.vs.empty());
    ASSERT_FALSE(draw.fs.empty());
    draw.prt = table;
    draw.vertex_count = 3;
    draw.ps.topology = 3;
    draw.ps.color_write_mask = 15;
    draw.color0_width = width;
    draw.color0_height = height;

    // The isolated RG8 arm is a control: without a matching luma plane, established narrow
    // coverage textures still broadcast their first byte. Adding the plane must change the upload
    // contract even when the chroma address and bytes were already cached by that control draw.
    for (const bool paired : {false, true}) {
        if (paired) {
            ShaderResource luma = chroma;
            luma.binding = 5;
            luma.num_components = 1;
            luma.width = luma.height = 4;
            luma.gpu_addr = 0x700000;
            luma.size = y.size();
            luma.host_data = tiled_y.data();
            luma.host_data_size = tiled_y.size();
            table->resources.push_back(luma);
        }
        draw.color0_base = paired ? 0xb10000 : 0xb00000;
        const auto image = render_submit_items({draw}, width, height);
        ASSERT_EQ(image.size(), size_t{width} * height * 4);
        const std::array<uint8_t, 4> expected = paired ? std::array<uint8_t, 4>{32, 224, 0, 255}
                                                       : std::array<uint8_t, 4>{32, 32, 32, 32};
        for (size_t pixel = 0; pixel < size_t{width} * height; ++pixel)
            for (size_t channel = 0; channel < expected.size(); ++channel)
                EXPECT_NEAR(image[pixel * 4 + channel], expected[channel], 1)
                    << "paired=" << paired << " pixel=" << pixel << " channel=" << channel;
    }
}
