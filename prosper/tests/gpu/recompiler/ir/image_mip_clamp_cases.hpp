// Dynamic IMAGE_LOAD_MIP selector values through the existing admitted tiled chain. The image
// access, upload plan and source proofs remain production services, separate from the IR value.
#pragma once

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"

#include <cstddef>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <vector>

namespace prosper::test::ir::image_mip_clamp {
inline constexpr uint32_t kLanes = 64, kWidth = 64, kMaxMip = 6;
inline constexpr uint32_t kTile = static_cast<uint32_t>(prosper::gpu::TileMode::Sw64KbRX);

inline LaneInputs inputs(bool empty = false) {
    LaneInputs in = lane_inputs(0x1a730001u);
    constexpr uint32_t selectors[] = {0, 1,  2,           3,           5,          6,
                                      7, 31, 0x7fffffffu, 0x80000000u, 0xffffffffu};
    in.s2.resize(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        in.s0[lane] = in.s1[lane] = 0; // texel (0,0) exists at every level
        in.s2[lane] = selectors[lane % std::size(selectors)];
        if (empty) in.active[lane] = false;
    }
    return in;
}

// Independent hand-worked oracle: the admitted three-, six- and seven-level images stop at
// levels 2, 5 and 6. Every level's first texel is a distinct nonzero integer, including level zero.
inline uint32_t expected(const LaneInputs& in, uint32_t lane, uint32_t levels) {
    if (!in.active[lane]) return in.old_d[lane];
    const uint32_t requested = in.s2[lane];
    // Hand-worked per declared count: 3 -> 2, 6 -> 5, 7 -> 6.
    const uint32_t last = levels == 3 ? 2u : (levels == 6 ? 5u : 6u);
    const uint32_t selected = requested < last ? requested : last;
    return 0x7100u + selected;
}

// One tiled chain shape. Every value the fixture writes is `0x7100 + level`, which fits both the
// 16- and 32-bit formats, so a lane's expectation names the level it must have read.
struct Geometry {
    uint32_t width = kWidth;   // square level zero
    uint32_t texel_bytes = 4;
    prosper::gpu::DataFormat format = prosper::gpu::DataFormat::Uint32;
    uint32_t max_mip = kMaxMip;   // the allocation's final level
};
// The #5429 shape: level zero of a 64x64 R32 Sw64KbRX chain already sits in the shared tail.
inline constexpr Geometry kAllTail64R32{};
// The same tail rule with a different element size and a different extent and chain length.
inline constexpr Geometry kAllTail64R16{64, 2, prosper::gpu::DataFormat::Uint16, kMaxMip};
inline constexpr Geometry kAllTail32R32{32, 4, prosper::gpu::DataFormat::Uint32, 5};
// Level zero outside the tail: the control that always passed.
inline constexpr Geometry kNonTail256R32{256, 4, prosper::gpu::DataFormat::Uint32, kMaxMip};

inline std::vector<uint8_t> image_bytes(const Geometry& g = kAllTail64R32) {
    using namespace prosper::gpu;
    const uint32_t bpe = g.texel_bytes;
    std::vector<uint8_t> bytes(tiled_mip_chain_bytes(g.width, g.width, bpe, kTile, g.max_mip), 0);
    for (uint32_t level = 0; level <= g.max_mip; ++level) {
        const auto layout = tiled_mip_level_layout(g.width, g.width, bpe, kTile, g.max_mip, level);
        const uint32_t level_width = g.width >> level;
        const size_t texels = size_t{level_width} * level_width;
        std::vector<uint8_t> pixels(texels * bpe, 0);
        const uint32_t value = 0x7100u + level;
        for (size_t t = 0; t < texels; ++t) std::memcpy(pixels.data() + t * bpe, &value, bpe);
        if (layout.in_tail)
            tile_surface_level(bytes.data(), bytes.size(), pixels.data(), level_width, level_width,
                               kTile, bpe, layout.tail_x, layout.tail_y);
        else
            tile_surface(bytes.data() + layout.byte_offset, pixels.data(), level_width, level_width,
                         kTile, 0, bpe);
    }
    return bytes;
}

// The texture's placement comes from the production T# decode, not from a hand copy of it.
// That decode leaves a tail-packed selected level's base at the ALLOCATION base and carries the
// in-block origin beside it (`mip_tail_offset`/`mip_tail_x`/`mip_tail_y`); every consumer of a
// ShaderResource relies on that contract. The first version of this fixture added the tail's own
// byte origin to `gpu_addr`, so the backend applied the origin twice and read the wrong texels
// (#5429): it was the fixture, not the upload, that was wrong.
inline prosper::gpu::ShaderResource texture(uint32_t levels, uint64_t image,
                                            const Geometry& g = kAllTail64R32) {
    using namespace prosper::gpu;
    DecodedImageDescriptor d;
    d.base = image;
    d.width = d.height = g.width;
    d.tile_mode = kTile;
    d.type = 9;   // SQ_RSRC_IMG 2D
    d.base_level = 0;
    d.last_level = static_cast<uint8_t>(levels - 1u);
    d.max_mip = static_cast<uint8_t>(g.max_mip);
    Gen5ImageFormatInfo fi;
    fi.format = g.format;
    fi.num_components = 1;
    fi.bytes_per_block = g.texel_bytes;
    const DecodedImageView view = image_base_level_view(d, fi);
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = 1;
    t.format = g.format;
    t.num_components = 1;
    t.width = view.width;
    t.height = view.height;
    t.depth = 1;
    t.tile_mode = kTile;
    t.declared_mip_levels = levels;
    shader_resource_apply_mip_chain_provenance(t, view);
    t.gpu_addr = view.supported ? view.base : 0;
    t.size = static_cast<uint32_t>(tiled_surface_bytes(g.width, g.width, kTile, 0, g.texel_bytes));
    t.in_mip_tail = view.in_mip_tail;
    t.mip_tail_offset = view.in_mip_tail ? static_cast<uint32_t>(view.mip_offset) : 0u;
    t.mip_tail_x = view.mip_tail_x;
    t.mip_tail_y = view.mip_tail_y;
    t.mip_tail_bytes = view.mip_tail_bytes;
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4u + c;
    return t;
}

inline LaneProgram program(bool nsa, uint32_t levels, uint32_t wave, uint64_t buffer = 0,
                           uint64_t image = 0, const Geometry& g = kAllTail64R32) {
    // Consecutive v[2:4] copies the independent mip from v5 after EXEC has read v4.
    const std::vector<uint32_t> op =
        nsa ? std::vector<uint32_t>{0xf004010au, 0x00040102u, 0x00000503u}
            : std::vector<uint32_t>{vop1(1u, 4u, 261u), 0xf0040108u, 0x00040102u};
    LaneProgram p = lane_program(op, kLanes, wave, true);
    p.table.resources[0].gpu_addr = buffer;
    p.table.resources.push_back(texture(levels, image, g));
    p.config.user_sgprs.resize(32);
    return p;
}
}   // namespace prosper::test::ir::image_mip_clamp
