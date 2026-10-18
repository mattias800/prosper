// Dynamic IMAGE_LOAD_MIP selector values through the existing admitted tiled chain. The image
// access, upload plan and source proofs remain production services, separate from the IR value.
#pragma once

#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"

#include <cstddef>
#include <iterator>
#include <cstdint>
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

// Independent hand-worked oracle: the admitted three-level and seven-level images stop at
// levels 2 and 6. Every level's first texel is a distinct nonzero integer, including level zero.
inline uint32_t expected(const LaneInputs& in, uint32_t lane, uint32_t levels) {
    if (!in.active[lane]) return in.old_d[lane];
    const uint32_t requested = in.s2[lane];
    const uint32_t last = levels == 3 ? 2u : 6u;
    const uint32_t selected = requested < last ? requested : last;
    return 0x7100u + selected;
}

inline std::vector<uint8_t> image_bytes(uint32_t width = kWidth) {
    using namespace prosper::gpu;
    std::vector<uint8_t> bytes(tiled_mip_chain_bytes(width, width, 4, kTile, kMaxMip), 0);
    for (uint32_t level = 0; level <= kMaxMip; ++level) {
        const auto layout = tiled_mip_level_layout(width, width, 4, kTile, kMaxMip, level);
        const uint32_t level_width = width >> level;
        std::vector<uint32_t> pixels(size_t{level_width} * level_width, 0x7100u + level);
        if (layout.in_tail)
            tile_surface_level(bytes.data(), bytes.size(),
                               reinterpret_cast<const uint8_t*>(pixels.data()), level_width,
                               level_width, kTile, 4, layout.tail_x, layout.tail_y);
        else
            tile_surface(bytes.data() + layout.byte_offset,
                         reinterpret_cast<const uint8_t*>(pixels.data()), level_width, level_width,
                         kTile, 0, 4);
    }
    return bytes;
}

inline LaneProgram program(bool nsa, uint32_t levels, uint32_t wave, uint64_t buffer = 0,
                           uint64_t image = 0, uint32_t width = kWidth) {
    using namespace prosper::gpu;
    // Consecutive v[2:4] copies the independent mip from v5 after EXEC has read v4.
    const std::vector<uint32_t> op =
        nsa ? std::vector<uint32_t>{0xf004010au, 0x00040102u, 0x00000503u}
            : std::vector<uint32_t>{vop1(1u, 4u, 261u), 0xf0040108u, 0x00040102u};
    LaneProgram p = lane_program(op, kLanes, wave, true);
    p.table.resources[0].gpu_addr = buffer;
    ShaderResource t{};
    t.cls = ResourceClass::Texture;
    t.binding = 5;
    t.sgpr_base = 16;
    t.img_dim = 1;
    t.format = DataFormat::Uint32;
    t.num_components = 1;
    t.width = t.height = width;
    t.depth = 1;
    t.tile_mode = kTile;
    t.declared_mip_levels = levels;
    t.mip_chain_element_width = t.mip_chain_element_height = width;
    t.mip_chain_bytes_per_block = 4;
    t.mip_chain_max_level = kMaxMip;
    const auto base = tiled_mip_level_layout(width, width, 4, kTile, kMaxMip, 0);
    t.gpu_addr = image ? image + base.byte_offset : 0;
    t.size = static_cast<uint32_t>(tiled_surface_bytes(width, width, kTile, 0, 4));
    t.in_mip_tail = base.in_tail;
    t.mip_tail_x = base.tail_x;
    t.mip_tail_y = base.tail_y;
    t.mip_tail_bytes = base.tail_block_bytes;
    for (uint32_t c = 0; c < 4; ++c) t.swizzle[c] = 4u + c;
    p.table.resources.push_back(t);
    p.config.user_sgprs.resize(32);
    return p;
}
} // namespace prosper::test::ir::image_mip_clamp
