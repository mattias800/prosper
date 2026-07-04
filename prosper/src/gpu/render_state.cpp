// render_state.cpp — see render_state.hpp.
#include "render_state.hpp"
#include "pm4_registers.hpp"
#include "vk_translate.hpp"

namespace prosper::gpu {

namespace {
namespace P = prosper::agc::Pm4;

// Read a register value from a class file (0 if unset).
uint32_t rd(const std::unordered_map<uint32_t, uint32_t>& file, uint32_t off) {
    auto it = file.find(off);
    return it == file.end() ? 0u : it->second;
}

// RDNA2 base-address pair: LO holds bits [39:8], HI holds bits [47:40]. Verified vs Kyty
// GraphicsRun.cpp (`(lo<<8) | ((hi&0xff)<<40)`).
uint64_t addr_of(uint32_t lo, uint32_t hi) {
    return (static_cast<uint64_t>(lo) << 8) | (static_cast<uint64_t>(hi & 0xffu) << 40);
}
}  // namespace

RenderState extract_render_state(const GpuState& st) {
    RenderState rs;

    // Shader program addresses (SH register file).
    rs.ps_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_PS), rd(st.sh, P::SPI_SHADER_PGM_HI_PS));
    rs.gs_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_GS), rd(st.sh, P::SPI_SHADER_PGM_HI_GS));
    rs.es_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_ES), rd(st.sh, P::SPI_SHADER_PGM_HI_ES));
    rs.hs_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_HS), rd(st.sh, P::SPI_SHADER_PGM_HI_HS));

    // Color MRT 0 (context register file).
    rs.color0_base            = addr_of(rd(st.cx, P::CB_COLOR0_BASE), rd(st.cx, P::CB_COLOR0_BASE_EXT));
    const uint32_t cinfo       = rd(st.cx, P::CB_COLOR0_INFO);
    rs.color0_format           = PM4_FIELD(cinfo, CB_COLOR0_INFO, FORMAT);
    rs.color0_number_type      = PM4_FIELD(cinfo, CB_COLOR0_INFO, NUMBER_TYPE);
    rs.color0_comp_swap        = PM4_FIELD(cinfo, CB_COLOR0_INFO, COMP_SWAP);

    // Primitive topology.
    rs.prim_type = PM4_FIELD(rd(st.cx, P::VGT_PRIMITIVE_TYPE), VGT_PRIMITIVE_TYPE, PRIM_TYPE);

    // Depth/stencil test state (decoded fields of DB_DEPTH_CONTROL).
    const uint32_t dc = rd(st.cx, P::DB_DEPTH_CONTROL);
    rs.stencil_enable = PM4_FIELD(dc, DB_DEPTH_CONTROL, STENCIL_ENABLE) != 0;
    rs.z_enable       = PM4_FIELD(dc, DB_DEPTH_CONTROL, Z_ENABLE) != 0;
    rs.z_write_enable = PM4_FIELD(dc, DB_DEPTH_CONTROL, Z_WRITE_ENABLE) != 0;
    rs.zfunc          = PM4_FIELD(dc, DB_DEPTH_CONTROL, ZFUNC);

    // Color blend state (decoded fields of CB_BLEND0_CONTROL).
    const uint32_t bc = rd(st.cx, P::CB_BLEND0_CONTROL);
    rs.blend_enable    = PM4_FIELD(bc, CB_BLEND0_CONTROL, ENABLE) != 0;
    rs.color_src_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
    rs.color_dst_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
    rs.color_comb_fcn  = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_COMB_FCN);

    // Faithful raw state registers.
    rs.db_depth_control  = dc;
    rs.cb_color_control  = rd(st.cx, P::CB_COLOR_CONTROL);
    rs.cb_blend0_control = bc;
    rs.cb_target_mask    = rd(st.cx, P::CB_TARGET_MASK);

    return rs;
}

ResolvedPipelineState resolve_pipeline_state(const RenderState& rs) {
    ResolvedPipelineState ps;
    ps.topology      = static_cast<uint32_t>(vk_topology(rs.prim_type));
    ps.color0_format = static_cast<uint32_t>(
        vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap));

    ps.depth_test_enable  = rs.z_enable;
    ps.depth_write_enable = rs.z_write_enable;
    ps.depth_compare_op   = vk_compare_op(rs.zfunc);

    ps.blend_enable            = rs.blend_enable;
    ps.src_color_blend_factor  = vk_blend_factor(rs.color_src_blend);
    ps.dst_color_blend_factor  = vk_blend_factor(rs.color_dst_blend);
    ps.color_blend_op          = vk_blend_op(rs.color_comb_fcn);

    // CB_TARGET_MASK holds a 4-bit write mask per MRT; MRT0 is bits [3:0]. RDNA2's R/G/B/A bit order
    // matches VkColorComponentFlags (R=1,G=2,B=4,A=8), so the nibble maps 1:1.
    ps.color_write_mask = rs.cb_target_mask & 0xFu;

    return ps;
}

} // namespace prosper::gpu
