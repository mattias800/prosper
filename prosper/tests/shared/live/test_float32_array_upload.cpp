// Guest-backed Float32 array layers must reach the real graphics sampler without half narrowing.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
int main() {
    prosper::register_builtin_hle();
    constexpr uint32_t W = 8, H = 8, Layers = 3;
    const uint32_t vs[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    constexpr size_t Backing = 0x80000;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), Backing, 2, 0,
                             reinterpret_cast<uint64_t>("float-array"), 0) == 0 && guest,
          "fixture maps real guest backing");
    if (!guest) return 1;
    prosper::frontend::register_live_renderer(".", false);
    auto table = std::make_shared<ShaderResourceTable>();
    ShaderResource source{};
    source.cls = ResourceClass::Texture; source.format = DataFormat::Float32;
    source.binding = 4; source.sgpr_base = 8; source.img_dim = 5;
    source.depth = Layers; source.width = W; source.height = H;
    source.gpu_addr = guest; source.mag_filter = source.min_filter = 0;
    source.swizzle[0] = 4; source.swizzle[1] = 5;
    source.swizzle[2] = 6; source.swizzle[3] = 7;
    table->resources.push_back(source);
    DrawItem draw;
    draw.vs = recompile_vertex(vs, std::size(vs)); draw.vertex_count = 3;
    draw.ps.topology = 3; draw.ps.color_write_mask = 15;
    draw.color0_base = 0x7c000000;
    draw.color0_width = W; draw.color0_height = H; draw.prt = table;
    check(!draw.vs.empty(), "fullscreen vertex shader compiles");
    // Separate layout arms cover row padding, per-layer selected-mip offsets and packed mip tails.
    for (unsigned layout = 0; layout < 3; ++layout) {
        for (unsigned components : {1u, 2u, 4u}) {
            auto& r = table->resources[0];
            r.num_components = components;
            const size_t bpt = components * sizeof(float);
            r.tile_mode = layout == 0 ? 0 : 24;
            r.in_mip_tail = layout == 2;
            r.mip_tail_x = r.in_mip_tail ? 16 : 0;
            r.mip_tail_y = r.in_mip_tail ? 8 : 0;
            r.mip_tail_bytes = r.in_mip_tail ? 0x10000 : 0;
            r.layer_mip_offset_bytes = layout == 1 ? 0x10000 : 0;
            r.linear_row_pitch_bytes = layout == 0 ? 256 : 0;
            const size_t surface = layout == 0 ? r.linear_row_pitch_bytes * H :
                tiled_surface_bytes(W, H, r.tile_mode, 0, bpt);
            r.layer_stride_bytes = 0x20000;
            r.size = Layers * r.layer_stride_bytes;
            std::memset(reinterpret_cast<void*>(guest), 0x55, Backing);
            for (unsigned generation = 0; generation < 2; ++generation) {
                float expected[Layers][4]{};
                for (unsigned layer = 0; layer < Layers; ++layer) {
                    expected[layer][3] = 1.0f;
                    for (unsigned c = 0; c < components; ++c)
                        expected[layer][c] = 0.125f * (1 + layer) +
                            float(3 + 7 * c + generation) / 65536.0f;
                    std::vector<uint8_t> linear(W * H * bpt);
                    for (size_t texel = 0; texel < W * H; ++texel)
                        std::memcpy(linear.data() + texel * bpt, expected[layer], bpt);
                    auto* dst = reinterpret_cast<uint8_t*>(guest) + layer * r.layer_stride_bytes;
                    if (r.in_mip_tail)
                        tile_surface_level(dst, r.mip_tail_bytes, linear.data(), W, H,
                                           r.tile_mode, bpt, r.mip_tail_x, r.mip_tail_y);
                    else if (layout == 1)
                        tile_surface(dst + r.layer_mip_offset_bytes, linear.data(), W, H,
                                     r.tile_mode, 0, bpt);
                    else for (size_t y = 0; y < H; ++y)
                        std::memcpy(dst + y * r.linear_row_pitch_bytes,
                                    linear.data() + y * W * bpt, W * bpt);
                }
                notify_guest_gpu_write(guest, r.size);
                for (unsigned layer = 0; layer < Layers; ++layer) {
                  for (bool load : {false, true}) {
                    // SAMPLE_L uses v2 as layer and v3 as LOD. Amplify a mismatch against an
                    // independent exact Float32 constant before RGBA8 readback, so 16F narrowing
                    // cannot hide behind 8-bit presentation. Exact equality produces 0.5 in RGBA.
                    std::vector<uint32_t> ps{0x7e0002ffu, 0x3f000000u,
                        0x7e0202ffu, 0x3f000000u,
                        0x7e0402ffu, std::bit_cast<uint32_t>(float(layer)), 0x7e060280u,
                        0xf0900f28u, 0x00820000u};
                    if (load) {
                        ps[1] = W / 2; ps[3] = H / 2; ps[5] = layer;
                        ps[7] = 0xf0000f28u; // IMAGE_LOAD: integer [x,y,layer], no sampler
                    }
                    for (unsigned c = 0; c < 4; ++c) {
                        ps.push_back(0x080000ffu | (c << 17) | (c << 9));
                        ps.push_back(std::bit_cast<uint32_t>(expected[layer][c]));
                        ps.push_back(0x100000ffu | (c << 17) | (c << 9));
                        ps.push_back(std::bit_cast<uint32_t>(65536.0f));
                        ps.push_back(0x060000ffu | (c << 17) | (c << 9));
                        ps.push_back(std::bit_cast<uint32_t>(0.5f));
                    }
                    ps.insert(ps.end(), {0xf800000fu, 0x03020100u, 0xbf810000u});
                    draw.fs = recompile_fragment(ps.data(), ps.size(), table.get());
                    const auto report = validate_spirv_descriptor_interface(
                        draw.fs, table.get(), 1, SpirvShaderStage::Fragment);
                    bool arrayed = false;
                    for (const auto& d : report.descriptors)
                        if (d.binding == 4 && d.image_arrayed) arrayed = true;
                    check(!draw.fs.empty() && report.ok() && arrayed,
                          "Float32 shader and descriptor agree on an actual array");
                    for (unsigned reuse = 0; reuse < 2; ++reuse) {
                        const auto image = render_submit_items({draw}, W, H);
                        bool exact = image.size() == W * H * 4;
                        for (uint8_t channel : image) exact &= channel >= 127 && channel <= 128;
                        if (!exact) std::printf("layout=%u components=%u generation=%u layer=%u reuse=%u bytes=%zu first=%u,%u,%u,%u surface=%zu\n",
                            layout, components, generation, layer, reuse, image.size(),
                            image.size() > 3 ? image[0] : 0, image.size() > 3 ? image[1] : 0,
                            image.size() > 3 ? image[2] : 0, image.size() > 3 ? image[3] : 0, surface);
                        check(exact, "each guest layer samples exact Float32 values and default channels on decode/reuse");
                    }
                  }
                }
            }
        }
    }
    unmap(guest, Backing, 0, 0, 0, 0);
    std::printf("Float32 array upload: %d failures\n", failures);
    return failures ? 1 : 0;
}
