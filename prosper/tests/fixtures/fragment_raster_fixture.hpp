// Actual registered saved-live/helper original and physically supplied single-sample context.
// These are fixture launch inputs, not reset values inferred by the production consumer.
#pragma once
#include "fixtures/fragment_draw_fixture.hpp"

namespace prosper::test::fragment_raster {
namespace g = prosper::gpu;
namespace p = prosper::agc::Pm4;
inline std::vector<uint32_t> original(uint32_t position = 0) {
    return {0xbe94047eu, 0xbefe0a7eu, 0xd8d480ffu, (position << 24) | position,
            0xbf8cc07fu, 0xbefe0414u, 0xf800180fu, position * 0x01010101u,
            0xbf810000u};
}
inline std::vector<std::pair<uint32_t, uint32_t>> context() {
    return {{p::SPI_PS_INPUT_ENA, 1u << 8},
            {p::SPI_PS_INPUT_ADDR, 1u << 8},
            {p::PA_SC_SHADER_CONTROL, 0},
            {p::PA_SC_MODE_CNTL_0, 0},
            {p::PA_SC_MODE_CNTL_1, 0},
            {p::PA_SC_AA_CONFIG, 0},
            {p::DB_SHADER_CONTROL, 1u << 11},
            {p::DB_DEPTH_CONTROL, 0},
            {p::DB_RENDER_CONTROL, 0},
            {p::DB_RENDER_OVERRIDE, 0},
            {p::DB_RENDER_OVERRIDE2, 0},
            {p::DB_EQAA, 0},
            {p::PA_SU_VTX_CNTL, 1},
            {p::PA_SC_CONSERVATIVE_RASTERIZATION_CNTL, 0},
            {p::PA_SC_AA_MASK_X0Y0_X1Y0, 0x00010001u},
            {p::PA_SC_AA_MASK_X0Y1_X1Y1, 0x00010001u}};
}
inline bool realize(g::DrawItem& draw) {
    const auto words = context();
    return fragment_draw::realize(draw, fragment_draw::color_a, original(),
                                  fragment_draw::ieee_rsrc1, 15, words);
}
} // namespace prosper::test::fragment_raster
