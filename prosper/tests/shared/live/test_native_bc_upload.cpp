// Native BCn sampled upload (#3873): a guest BC texture reaches Vulkan as its detiled 4x4 blocks
// instead of CPU-decoded RGBA8, and must sample the same texels the decoder produces.
//
// Every arm renders a pixel-center, point-sampled draw whose render target has the texture's own
// extent, so pixel (x,y) is exactly texel (x,y). The oracle is the CPU decoder applied to the same
// linear blocks. Three source layouts reach the three native fill paths: tight linear (the guarded
// straight copy that doubles as the cache validation source), pitch-padded linear (row-by-row copy)
// and SW_4KB_S tiled (the decoder's own block detile). The extent is deliberately not a multiple of
// four, so a wrong bufferRowLength or image extent shears the picture.
//
// The discriminator is the backend's upload byte count: ceil(w/4)*ceil(h/4)*block on the native
// path, w*h*4 on the decoder path. Run with --decoder (ctest sets PROSPER_NO_NATIVE_BC=1) for the
// negative control: same pixels, RGBA8-sized upload. A device that cannot sample BCn keeps the
// decoder; the native arm then reports SKIP rather than passing vacuously.
//
// Tolerance: BC7 interpolation is bit-exact by specification, so it allows 1/255. BC1-BC5 endpoint
// interpolation is implementation-defined within the format's precision (the decoder truncates the
// 1/3 and 1/7 weights; hardware may round), so they allow 4/255. BC6H compares the decoder's
// clamped UNORM8 against the hardware float written to a UNORM8 target, also within 4/255.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/bc_decode.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/image/exec_image.hpp"
#include "shared/live/live_renderer.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

namespace {

uint32_t lcg_state = 0x12345u;
uint8_t next_byte() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return static_cast<uint8_t>(lcg_state >> 24);
}

// Random blocks, nudged into defined encodings: BC7 needs a nonzero mode byte (mode 8 is reserved),
// and BC6H's two-bit modes (00, 01) avoid its reserved five-bit mode values. Random BC6H endpoints
// are overwhelmingly above 1.0, where both paths saturate a UNORM8 target and compare nothing, so a
// BC6H block is redrawn until most of its channels decode below 250 and the comparison has content.
std::vector<uint8_t> random_blocks(DataFormat format, size_t blocks) {
    const uint32_t bb = bc_block_bytes(format);
    std::vector<uint8_t> out(blocks * bb);
    for (size_t i = 0; i < blocks; ++i) {
        uint8_t* block = out.data() + i * bb;
        for (unsigned attempt = 0; attempt < 100000; ++attempt) {
            for (uint32_t b = 0; b < bb; ++b) block[b] = next_byte();
            if (format == DataFormat::Bc7) block[0] |= 0x40;          // mode <= 6
            if (format != DataFormat::Bc6) break;
            block[0] &= static_cast<uint8_t>(~0x02u);
            uint8_t texels[16 * 4];
            bc_decode_surface(texels, block, bb, 4, 4, format);
            unsigned unsaturated = 0;
            for (unsigned t = 0; t < 16; ++t)
                for (unsigned c = 0; c < 3; ++c) unsaturated += texels[t * 4 + c] < 250;
            if (unsaturated >= 36) break;
        }
    }
    return out;
}

// Build the SW_4KB_S tiled image of `linear` by inverting the runtime's own element detiler: detile a
// buffer whose every tiled element holds its own index, then scatter. This pins that the native path
// detiles exactly as the decoder does; the detiler's geometry itself is test_tile's subject.
std::vector<uint8_t> tile_blocks(const std::vector<uint8_t>& linear, uint32_t bw, uint32_t bh,
                                 uint32_t bb, uint32_t tile_mode) {
    const size_t tiled_bytes = tiled_elements_bytes(bw, bh, bb, tile_mode);
    std::vector<uint8_t> ids(tiled_bytes, 0);
    for (size_t e = 0; e * bb < tiled_bytes; ++e) {
        const uint64_t id = e + 1;
        std::memcpy(ids.data() + e * bb, &id, sizeof(id));
    }
    std::vector<uint8_t> map(static_cast<size_t>(bw) * bh * bb, 0);
    detile_elements(map.data(), ids.data(), ids.size(), bw, bh, bb, tile_mode);
    std::vector<uint8_t> tiled(tiled_bytes, 0);
    for (size_t l = 0; l < static_cast<size_t>(bw) * bh; ++l) {
        uint64_t id = 0;
        std::memcpy(&id, map.data() + l * bb, sizeof(id));
        if (!id) return {};
        std::memcpy(tiled.data() + (id - 1) * bb, linear.data() + l * bb, bb);
    }
    return tiled;
}

}  // namespace

int main(int argc, char** argv) {
    const bool decoder_arm = argc == 2 && std::strcmp(argv[1], "--decoder") == 0;
    prosper::register_builtin_hle();
    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    prosper::frontend::register_live_renderer(".", false);

    // Pure shape contract: only a plain single-surface 2D texture is eligible.
    {
        ShaderResource r{};
        r.cls = ResourceClass::Texture; r.format = DataFormat::Bc7; r.img_dim = 1; r.depth = 1;
        r.declared_mip_levels = 1;
        check(prosper::frontend::native_bc_sampled_format(r, false) == VK_FORMAT_BC7_UNORM_BLOCK,
              "plain 2D BC7 maps to BC7_UNORM");
        r.format = DataFormat::Bc1;
        check(prosper::frontend::native_bc_sampled_format(r, false) == VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
              "BC1 maps to the RGBA (punch-through alpha) variant the decoder implements");
        r.format = DataFormat::Bc6;
        check(prosper::frontend::native_bc_sampled_format(r, false) == VK_FORMAT_BC6H_UFLOAT_BLOCK,
              "BC6H UF16 maps to BC6H_UFLOAT");
        r.declared_mip_levels = 4;
        check(prosper::frontend::native_bc_sampled_format(r, false) == 0u &&
                  prosper::frontend::native_bc_sampled_format(r, true) != 0u,
              "a declared mip chain keeps the decoder unless explicitly allowed");
        r.declared_mip_levels = 1;
        r.img_dim = 3; r.depth = 6;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "cube keeps the decoder");
        r.img_dim = 2; r.depth = 4;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "volume keeps the decoder");
        r.img_dim = 5; r.depth = 8;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "array keeps the decoder");
        r.img_dim = 1; r.depth = 1; r.compression_enabled = true;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "DCC keeps the decoder");
        r.compression_enabled = false; r.cls = ResourceClass::StorageImage;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "storage keeps the decoder");
        r.cls = ResourceClass::Texture; r.format = DataFormat::Unorm8;
        check(prosper::frontend::native_bc_sampled_format(r, true) == 0u, "non-BC is not affected");
    }

    const bool device_native = prosper::test::backend_native_bc_sampled_supported(
        VK_FORMAT_BC7_UNORM_BLOCK);
    const bool expect_native = device_native && !decoder_arm;
    std::printf("device native BC=%d decoder_arm=%d\n", device_native, decoder_arm);
    if (!device_native && !decoder_arm)
        std::printf("SKIP native assertions: device cannot sample BCn (decoder fallback kept)\n");

    constexpr uint32_t W = 18, H = 10;           // deliberately not multiples of 4
    constexpr uint32_t BW = (W + 3) / 4, BH = (H + 3) / 4;
    const uint32_t ps_rdna[] = {
        0x100000ffu, std::bit_cast<uint32_t>(1.0f / W),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / H),
        0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    enum class Layout { TightLinear, PaddedLinear, Tiled };
    const struct { DataFormat format; uint32_t comps; int tolerance; } formats[] = {
        {DataFormat::Bc1, 4, 4}, {DataFormat::Bc2, 4, 4}, {DataFormat::Bc3, 4, 4},
        {DataFormat::Bc4, 1, 4}, {DataFormat::Bc5, 2, 4}, {DataFormat::Bc6, 3, 4},
        {DataFormat::Bc7, 4, 1},
    };
    for (const auto& f : formats) {
        for (Layout layout : {Layout::TightLinear, Layout::PaddedLinear, Layout::Tiled}) {
            const uint32_t bb = bc_block_bytes(f.format);
            const std::vector<uint8_t> linear = random_blocks(f.format, size_t(BW) * BH);
            std::vector<uint8_t> oracle(size_t(W) * H * 4, 0);
            check(bc_decode_surface(oracle.data(), linear.data(), linear.size(), W, H, f.format),
                  "oracle decodes");
            std::vector<uint8_t> source;
            uint32_t pitch = 0, tile_mode = 0;
            if (layout == Layout::TightLinear) {
                source = linear;
            } else if (layout == Layout::PaddedLinear) {
                pitch = BW * bb + 3 * bb;
                source.assign(size_t(pitch) * BH, 0xcd);   // padding must never be sampled
                for (uint32_t y = 0; y < BH; ++y)
                    std::memcpy(source.data() + size_t(y) * pitch,
                                linear.data() + size_t(y) * BW * bb, size_t(BW) * bb);
            } else {
                tile_mode = static_cast<uint32_t>(TileMode::Sw4KbS);
                source = tile_blocks(linear, BW, BH, bb, tile_mode);
                check(!source.empty(), "tiled fixture is a complete permutation");
            }
            ShaderResource r{};
            r.cls = ResourceClass::Texture;
            r.format = f.format;
            r.num_components = f.comps;
            r.binding = 4; r.sgpr_base = 8;
            r.img_dim = 1; r.depth = 1; r.declared_mip_levels = 1;
            r.width = W; r.height = H;
            r.tile_mode = tile_mode;
            r.linear_row_pitch_bytes = pitch;
            r.mag_filter = r.min_filter = 0;
            r.gpu_addr = 0x71000000;
            r.size = static_cast<uint32_t>(source.size());
            r.host_data = source.data(); r.host_data_size = source.size();
            auto table = std::make_shared<ShaderResourceTable>();
            table->resources.push_back(r);
            DrawItem draw;
            draw.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
            const PixelSystemInputMapping positions{0x300u, 0x300u};
            draw.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), table.get(), &positions);
            draw.prt = table; draw.vertex_count = 3;
            draw.ps.topology = 3; draw.ps.color_write_mask = 15;
            draw.color0_base = 0x72000000;
            draw.color0_width = W; draw.color0_height = H;
            const auto actual = render_submit_items({draw}, W, H);
            const auto stats = prosper::test::backend_texture_upload_stats();
            char label[160];
            std::snprintf(label, sizeof label, "fmt=%u layout=%d", static_cast<unsigned>(f.format),
                          static_cast<int>(layout));
            if (actual.size() != oracle.size()) {
                std::printf("  %s: rendered %zu bytes\n", label, actual.size());
                check(false, "draw renders a complete image");
                continue;
            }
            int worst = 0; size_t worst_at = 0;
            for (size_t i = 0; i < oracle.size(); ++i) {
                // Missing channels follow the hardware rule on both paths; compare what the
                // format carries (BC4: R; BC5: RG; BC6H: RGB) plus alpha where it is defined.
                const uint32_t c = static_cast<uint32_t>(i % 4);
                if (c >= f.comps && c != 3) continue;
                const int d = std::abs(int(actual[i]) - int(oracle[i]));
                if (d > worst) { worst = d; worst_at = i; }
            }
            std::printf("  %s: upload_bytes=%llu worst=%d at texel %zu ch %zu (got %u want %u)\n",
                        label, static_cast<unsigned long long>(stats.upload_bytes), worst,
                        worst_at / 4, worst_at % 4, actual[worst_at], oracle[worst_at]);
            check(worst <= f.tolerance, "sampled texels match the CPU decoder within tolerance");
            const uint64_t native_bytes = uint64_t(BW) * BH * bb;
            const uint64_t decoded_bytes = uint64_t(W) * H * 4;
            if (expect_native)
                check(stats.upload_bytes == native_bytes,
                      "native path uploads ceil(w/4)*ceil(h/4) blocks, not decoded texels");
            else
                check(stats.upload_bytes == decoded_bytes,
                      "decoder path uploads w*h RGBA8 texels");
        }
    }
    // Guest mip chain (#3873): a tiled BC texture declaring a chain uploads the guest's OWN levels,
    // because a block format cannot be blit-generated. Each level holds different random blocks, so
    // a level the backend generated from level 0 (the decoder path) cannot match it. The sampler's
    // LOD clamp pins one level per draw; an 8x8 target point-samples it at known texels.
    auto alloc = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    constexpr uint64_t kGuestBytes = 0x400000;
    constexpr uint64_t kRegion = 0x200000;   // the chain is written at guest and guest+kRegion
    uint64_t guest = 0, physical = 0;
    check(alloc && release && map && unmap &&
          alloc(0, 0x200000000ull, kGuestBytes, 0x10000, 0,
                reinterpret_cast<uint64_t>(&physical)) == 0 &&
          map(reinterpret_cast<uint64_t>(&guest), kGuestBytes, 2, 0, physical, 0x10000) == 0 && guest,
          "fixture maps real guest memory for the chain arm");
    constexpr uint32_t T = 8;                     // target extent
    const uint32_t chain_ps[] = {
        0x100000ffu, std::bit_cast<uint32_t>(1.0f / T),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / T),
        0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    for (const DataFormat format : {DataFormat::Bc7, DataFormat::Bc1}) {
        if (!guest) break;
        constexpr uint32_t CW = 512, CH = 512, MAX_MIP = 9;
        const uint32_t bb = bc_block_bytes(format);
        const uint32_t tile_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
        const uint32_t ew = CW / 4, eh = CH / 4;
        const size_t chain_bytes = tiled_mip_chain_bytes(ew, eh, bb, tile_mode, MAX_MIP);
        check(chain_bytes && chain_bytes <= kRegion, "chain fits the guest allocation");
        if (!chain_bytes || chain_bytes > kRegion) continue;
        std::memset(reinterpret_cast<void*>(guest), 0, kGuestBytes);
        std::vector<std::vector<uint8_t>> level_blocks(MAX_MIP + 1);
        uint64_t level0_offset = 0;
        bool placed = true;
        for (uint32_t level = 0; level <= MAX_MIP; ++level) {
            const TiledMipLevelLayout layout =
                tiled_mip_level_layout(ew, eh, bb, tile_mode, MAX_MIP, level);
            const uint32_t lw = std::max(CW >> level, 1u), lh = std::max(CH >> level, 1u);
            const uint32_t lbw = (lw + 3) / 4, lbh = (lh + 3) / 4;
            level_blocks[level] = random_blocks(format, size_t(lbw) * lbh);
            if (!layout.supported) { placed = false; break; }
            if (level == 0) level0_offset = layout.byte_offset;
            // Tile by inverting the runtime detiler over this level's own span (see tile_blocks).
            const size_t span = layout.in_tail ? layout.tail_block_bytes
                                               : tiled_elements_bytes(lbw, lbh, bb, tile_mode);
            std::vector<uint8_t> ids(span, 0);
            for (size_t e = 0; e * bb < span; ++e) {
                const uint64_t id = e + 1;
                std::memcpy(ids.data() + e * bb, &id, sizeof(id));
            }
            std::vector<uint8_t> where(size_t(lbw) * lbh * bb, 0);
            if (layout.in_tail)
                detile_elements_level(where.data(), ids.data(), span, lbw, lbh, bb, tile_mode,
                                      layout.tail_x, layout.tail_y);
            else
                detile_elements(where.data(), ids.data(), span, lbw, lbh, bb, tile_mode);
            uint8_t* level_base = reinterpret_cast<uint8_t*>(guest) +
                (layout.in_tail ? 0 : layout.byte_offset);
            for (size_t l = 0; l < size_t(lbw) * lbh; ++l) {
                uint64_t id = 0;
                std::memcpy(&id, where.data() + l * bb, sizeof(id));
                if (!id) { placed = false; break; }
                std::memcpy(level_base + (id - 1) * bb, level_blocks[level].data() + l * bb, bb);
            }
        }
        check(placed, "every chain level is placed through the runtime's own layout");
        if (!placed) continue;
        // A second, byte-identical copy at another address gives the same-pass arm below a decode
        // identity no earlier draw has touched.
        std::memcpy(reinterpret_cast<void*>(guest + kRegion), reinterpret_cast<void*>(guest),
                    kRegion);
        notify_guest_gpu_write(guest, kGuestBytes);
        ShaderResource r{};
        r.cls = ResourceClass::Texture;
        r.format = format;
        r.num_components = 4;
        r.binding = 4; r.sgpr_base = 8;
        r.img_dim = 1; r.depth = 1;
        r.width = CW; r.height = CH;
        r.tile_mode = tile_mode;
        r.declared_mip_levels = MAX_MIP + 1;
        r.mip_chain_element_width = ew; r.mip_chain_element_height = eh;
        r.mip_chain_bytes_per_block = bb; r.mip_chain_max_level = MAX_MIP;
        r.mip_chain_base_level = 0;
        r.mag_filter = r.min_filter = 0; r.mip_filter = 0;
        r.gpu_addr = guest + level0_offset;
        r.size = static_cast<uint32_t>(tiled_elements_bytes(ew, eh, bb, tile_mode));
        check(shader_resource_block_mip_chain_plan(r, bb).valid &&
                  shader_resource_block_mip_chain_plan(r, bb).level_count == MAX_MIP + 1,
              "the renderer's plan models this chain");
        const int tolerance = format == DataFormat::Bc7 ? 1 : 4;
        auto make_draw = [&](const ShaderResource& resource, uint64_t color_base) {
            auto table = std::make_shared<ShaderResourceTable>();
            table->resources.push_back(resource);
            DrawItem draw;
            draw.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
            const PixelSystemInputMapping positions{0x300u, 0x300u};
            draw.fs = recompile_fragment(chain_ps, std::size(chain_ps), table.get(), &positions);
            draw.prt = table; draw.vertex_count = 3;
            draw.ps.topology = 3; draw.ps.color_write_mask = 15;
            draw.color0_base = color_base;
            draw.color0_width = T; draw.color0_height = T;
            return draw;
        };
        // Worst channel error of an 8x8 point-sampled target against guest level `level`. Levels
        // smaller than one 4x4 block (2x2, 1x1) sample the top-left texels of their single block.
        auto level_worst = [&](const std::vector<uint8_t>& actual, uint32_t level) {
            const uint32_t lw = std::max(CW >> level, 1u), lh = std::max(CH >> level, 1u);
            std::vector<uint8_t> oracle(size_t(lw) * lh * 4);
            bc_decode_surface(oracle.data(), level_blocks[level].data(),
                              level_blocks[level].size(), lw, lh, format);
            if (actual.size() != size_t(T) * T * 4) return 999;
            int worst = 0;
            for (uint32_t y = 0; y < T; ++y)
                for (uint32_t x = 0; x < T; ++x) {
                    const uint32_t tx = (2 * x + 1) * lw / (2 * T);
                    const uint32_t ty = (2 * y + 1) * lh / (2 * T);
                    for (uint32_t c = 0; c < 4; ++c)
                        worst = std::max(worst, std::abs(int(actual[(y * T + x) * 4 + c]) -
                            int(oracle[(size_t(ty) * lw + tx) * 4 + c])));
                }
            return worst;
        };
        // #3883 review: the decode key does not hold the T#'s level count, so two descriptors at one
        // address that differ only in last_level share it. A reused entry must never be served to a
        // descriptor that expects a different number of packed levels -- the backend would read the
        // missing levels past the end of the cached buffer.
        if (expect_native) {
            // Same pass (one decode span): a 1-level view, then a 10-level view of the SAME
            // address. Draw B covers the whole target, so the output is B's sample of level 3.
            ShaderResource single = r;
            single.gpu_addr += kRegion;
            single.declared_mip_levels = 1;
            single.min_lod = single.max_lod = 0.0f;
            ShaderResource chained = r;
            chained.gpu_addr += kRegion;
            chained.min_lod = chained.max_lod = 3.0f;
            const auto same_pass = render_submit_items(
                {make_draw(single, 0x74000000), make_draw(chained, 0x74000000)}, T, T);
            const int worst = level_worst(same_pass, 3);
            std::printf("  chain fmt=%u same-pass 1-then-10 levels: worst=%d\n",
                        static_cast<unsigned>(format), worst);
            check(worst <= tolerance,
                  "a 1-level entry is not reused for a 10-level view of the same address (same pass)");
            // Across submits (persistent cache): a 4-level chain, then a 10-level one. Both entries
            // validate the same allocation span, so only the packed level count tells them apart.
            ShaderResource four = r;
            four.declared_mip_levels = 4;
            four.min_lod = four.max_lod = 0.0f;
            (void)render_submit_items({make_draw(four, 0x74100000)}, T, T);
            ShaderResource ten = r;
            ten.min_lod = ten.max_lod = 5.0f;
            const auto across = render_submit_items({make_draw(ten, 0x74200000)}, T, T);
            const int worst_across = level_worst(across, 5);
            std::printf("  chain fmt=%u cross-submit 4-then-10 levels: worst=%d\n",
                        static_cast<unsigned>(format), worst_across);
            check(worst_across <= tolerance,
                  "a 4-level entry is not reused for a 10-level view (persistent cache)");
        }
        for (const uint32_t level : {0u, 1u, 2u, 3u, 5u, 7u, 8u, 9u}) {
            r.min_lod = r.max_lod = static_cast<float>(level);
            prosper::frontend::reset_texture_decode_scope_stats();
            const auto actual = render_submit_items(
                {make_draw(r, 0x73000000 + level * 0x10000)}, T, T);
            const auto stats = prosper::test::backend_texture_upload_stats();
            const auto decode_stats = prosper::frontend::texture_decode_scope_stats();
            // Only the sampler's LOD clamp changed after the first draw: the guest chain did not, so
            // the persistent cache must validate it unchanged over its WHOLE span (which starts
            // below the level-0 address) and reuse it rather than decode and upload it again.
            if (expect_native && level != 0)
                check(decode_stats.decodes == 0 && stats.unique_uploads == 0,
                      "an unchanged guest chain is reused, not re-decoded");
            const int worst = level_worst(actual, level);
            std::printf("  chain fmt=%u level=%u upload_bytes=%llu worst=%d\n",
                        static_cast<unsigned>(format), level,
                        static_cast<unsigned long long>(stats.upload_bytes), worst);
            if (expect_native) {
                check(worst <= tolerance, "each LOD samples the guest's own level, not a generated one");
                if (stats.unique_uploads)
                    check(stats.upload_bytes == prosper::test::backend_texture_chain_bytes(
                              static_cast<VkFormat>(prosper::frontend::native_bc_sampled_format(r, true)),
                              CW, CH, MAX_MIP + 1),
                          "the whole guest chain is staged once");
            } else if (level == 0) {
                check(worst <= tolerance, "decoder path samples level 0 correctly");
            } else if (level >= 2 && level <= 5) {
                // A generated level is a box filter of level 0, not the guest's own random level:
                // this is what makes the native assertion above discriminating.
                check(worst > tolerance, "decoder path generates levels (they differ from the guest's)");
            }
        }
    }
    if (guest) {
        unmap(guest, kGuestBytes, 0, 0, 0, 0);
        release(physical, kGuestBytes, 0, 0, 0, 0);
    }
    std::printf("native bc upload: %d failures\n", failures);
    return failures ? 1 : 0;
}
