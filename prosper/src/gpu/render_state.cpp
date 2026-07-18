// render_state.cpp — see render_state.hpp.
#include "render_state.hpp"
#include "pm4_registers.hpp"
#include "vk_translate.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

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

// PA_CL_VPORT_* registers hold IEEE-754 floats.
float flt(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

int32_t add_scissor_offset(uint32_t coordinate, int32_t offset, bool disabled) {
    const int64_t value = static_cast<int64_t>(coordinate) + (disabled ? 0 : offset);
    return static_cast<int32_t>(std::clamp<int64_t>(
        value, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
}

int32_t scale_scissor_boundary(int32_t value, float scale, bool lower) {
    const double scaled = static_cast<double>(value) * static_cast<double>(scale);
    const double rounded = lower ? std::floor(scaled) : std::ceil(scaled);
    return static_cast<int32_t>(std::clamp(
        rounded, static_cast<double>(std::numeric_limits<int32_t>::min()),
        static_cast<double>(std::numeric_limits<int32_t>::max())));
}

// sRGB (gamma-encoded) 8-bit sample -> linear [0,1]. VkClearColorValue's float channels are LINEAR
// for an sRGB attachment (Vulkan re-encodes on store), so an sRGB fast-clear word must be linearized
// here to round-trip to the exact stored byte. Identity at 0 and 1, so black/white are exact.
float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Decode a CB fast-clear word into an RGBA float clear color (Vulkan order: out[0]=R). Returns false
// for formats we do not decode yet, so the caller falls back to opaque black — the correct PS5
// default and never worse than the old debug blue. CONFIDENCE: HIGH for 8_8_8_8 (the live-verified
// Messenger / PPSA02664 target); the byte order matches vk_color_format's 0xA row (STD->R8G8B8A8,
// ALT->B8G8R8A8) and the sRGB rows are linearized. Other formats: MED -> left to the black fallback.
bool decode_clear_color(uint32_t format, uint32_t number_type, uint32_t comp_swap,
                        bool has_clear, uint32_t w0, float out[4]) {
    if (!has_clear) return false;
    if (format == 0xAu) {  // COLOR_8_8_8_8
        const float b0 = (w0 & 0xFFu) / 255.0f, b1 = ((w0 >> 8) & 0xFFu) / 255.0f,
                    b2 = ((w0 >> 16) & 0xFFu) / 255.0f, b3 = ((w0 >> 24) & 0xFFu) / 255.0f;
        float r, g, b, a = b3;                     // alpha is byte 3 for both swaps
        if (comp_swap == 1u) { r = b2; g = b1; b = b0; }   // ALT: B8G8R8A8 (byte0=B)
        else                            { r = b0; g = b1; b = b2; }  // STD: R8G8B8A8 (byte0=R)
        const bool is_srgb = (number_type == 6u);
        out[0] = is_srgb ? srgb_to_linear(r) : r;
        out[1] = is_srgb ? srgb_to_linear(g) : g;
        out[2] = is_srgb ? srgb_to_linear(b) : b;
        out[3] = a;                                // alpha is always linear
        return true;
    }
    return false;  // unmapped format -> caller uses opaque-black fallback
}
}  // namespace

RenderState extract_render_state(const GpuState& st) {
    RenderState rs;

    // Shader program addresses (SH register file).
    rs.ps_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_PS), rd(st.sh, P::SPI_SHADER_PGM_HI_PS));
    rs.gs_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_GS), rd(st.sh, P::SPI_SHADER_PGM_HI_GS));
    rs.es_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_ES), rd(st.sh, P::SPI_SHADER_PGM_HI_ES));
    rs.hs_addr = addr_of(rd(st.sh, P::SPI_SHADER_PGM_LO_HS), rd(st.sh, P::SPI_SHADER_PGM_HI_HS));
    for (uint32_t i = 0; i < rs.ps_input_cntl.size(); ++i) {
        const uint32_t reg = P::SPI_PS_INPUT_CNTL_0 + i;
        auto it = st.cx.find(reg);
        if (it == st.cx.end()) continue;
        rs.ps_input_cntl[i] = it->second;
        rs.ps_input_cntl_valid_mask |= 1u << i;
    }
    rs.ps_input_ena = rd(st.cx, P::SPI_PS_INPUT_ENA);
    rs.ps_input_addr = rd(st.cx, P::SPI_PS_INPUT_ADDR);

    // Color MRT 0 (context register file).
    rs.color0_base            = addr_of(rd(st.cx, P::CB_COLOR0_BASE), rd(st.cx, P::CB_COLOR0_BASE_EXT));
    const uint32_t cinfo       = rd(st.cx, P::CB_COLOR0_INFO);
    rs.color0_format           = PM4_FIELD(cinfo, CB_COLOR0_INFO, FORMAT);
    rs.color0_number_type      = PM4_FIELD(cinfo, CB_COLOR0_INFO, NUMBER_TYPE);
    rs.color0_comp_swap        = PM4_FIELD(cinfo, CB_COLOR0_INFO, COMP_SWAP);

    auto color_attrib2 = st.cx.find(P::CB_COLOR0_ATTRIB2);
    if (color_attrib2 != st.cx.end()) {
        rs.color0_has_extent = true;
        rs.color0_width  = PM4_FIELD(color_attrib2->second, CB_COLOR0_ATTRIB2, MIP0_WIDTH) + 1u;
        rs.color0_height = PM4_FIELD(color_attrib2->second, CB_COLOR0_ATTRIB2, MIP0_HEIGHT) + 1u;
    }

    // CB fast-clear color (CB_COLOR0_CLEAR_WORD0/1). Present only when the game programs a fast-clear
    // for MRT 0; the words carry the clear value in the target's pixel format (decoded in resolve).
    rs.color0_has_clear   = st.cx.count(P::CB_COLOR0_CLEAR_WORD0) || st.cx.count(P::CB_COLOR0_CLEAR_WORD1);
    rs.color0_clear_word0 = st.cx.count(P::CB_COLOR0_CLEAR_WORD0) ? rd(st.cx, P::CB_COLOR0_CLEAR_WORD0) : 0u;
    rs.color0_clear_word1 = st.cx.count(P::CB_COLOR0_CLEAR_WORD1) ? rd(st.cx, P::CB_COLOR0_CLEAR_WORD1) : 0u;

    // Color MRT 1 uses the same field layouts as MRT0, with the RDNA2 target-block stride encoded by
    // the named register constants in pm4_registers.hpp.
    rs.color1_base = addr_of(rd(st.cx, P::CB_COLOR1_BASE), rd(st.cx, P::CB_COLOR1_BASE_EXT));
    const uint32_t cinfo1 = rd(st.cx, P::CB_COLOR1_INFO);
    rs.color1_format = PM4_FIELD(cinfo1, CB_COLOR0_INFO, FORMAT);
    rs.color1_number_type = PM4_FIELD(cinfo1, CB_COLOR0_INFO, NUMBER_TYPE);
    rs.color1_comp_swap = PM4_FIELD(cinfo1, CB_COLOR0_INFO, COMP_SWAP);
    auto color1_attrib2 = st.cx.find(P::CB_COLOR1_ATTRIB2);
    if (color1_attrib2 != st.cx.end()) {
        rs.color1_has_extent = true;
        rs.color1_width = PM4_FIELD(color1_attrib2->second, CB_COLOR0_ATTRIB2, MIP0_WIDTH) + 1u;
        rs.color1_height = PM4_FIELD(color1_attrib2->second, CB_COLOR0_ATTRIB2, MIP0_HEIGHT) + 1u;
    }
    rs.color1_has_clear = st.cx.count(P::CB_COLOR1_CLEAR_WORD0) ||
                          st.cx.count(P::CB_COLOR1_CLEAR_WORD1);
    rs.color1_clear_word0 = st.cx.count(P::CB_COLOR1_CLEAR_WORD0)
        ? rd(st.cx, P::CB_COLOR1_CLEAR_WORD0) : 0u;
    rs.color1_clear_word1 = st.cx.count(P::CB_COLOR1_CLEAR_WORD1)
        ? rd(st.cx, P::CB_COLOR1_CLEAR_WORD1) : 0u;

    // Primitive topology. VGT_PRIMITIVE_TYPE (0x242) is a UCONFIG register in RDNA2 (the game sets it via
    // a Uc-class SetRegsIndirect / CreatePrimState's uc[2]), NOT a context register — read it from st.uc.
    // (Reading st.cx left it 0 -> the default PointList topology, so triangles rasterized as ~3 points.)
    rs.prim_type = PM4_FIELD(rd(st.uc, P::VGT_PRIMITIVE_TYPE), VGT_PRIMITIVE_TYPE, PRIM_TYPE);

    // Depth/stencil test state (decoded fields of DB_DEPTH_CONTROL).
    const uint32_t dc = rd(st.cx, P::DB_DEPTH_CONTROL);
    rs.stencil_enable = PM4_FIELD(dc, DB_DEPTH_CONTROL, STENCIL_ENABLE) != 0;
    rs.z_enable       = PM4_FIELD(dc, DB_DEPTH_CONTROL, Z_ENABLE) != 0;
    rs.z_write_enable = PM4_FIELD(dc, DB_DEPTH_CONTROL, Z_WRITE_ENABLE) != 0;
    rs.zfunc          = PM4_FIELD(dc, DB_DEPTH_CONTROL, ZFUNC);

    // Depth/stencil clear values (DB_DEPTH_CLEAR = IEEE-754 float, DB_STENCIL_CLEAR = 8-bit). Present
    // only when the game programmed them; the depth attachment's LOAD_OP_CLEAR must use the guest's
    // value (not a fixed 0.5) or a standard depth test wrongly rejects fragments (#371).
    rs.has_depth_clear     = st.cx.count(P::DB_DEPTH_CLEAR) != 0;
    rs.has_stencil_clear   = st.cx.count(P::DB_STENCIL_CLEAR) != 0;
    rs.depth_clear_value   = rs.has_depth_clear ? flt(rd(st.cx, P::DB_DEPTH_CLEAR)) : 1.0f;
    rs.stencil_clear_value = PM4_FIELD(rd(st.cx, P::DB_STENCIL_CLEAR), DB_STENCIL_CLEAR, CLEAR);
    rs.db_render_control   = rd(st.cx, P::DB_RENDER_CONTROL);
    rs.depth_clear_enable = PM4_FIELD(rs.db_render_control, DB_RENDER_CONTROL, DEPTH_CLEAR_ENABLE) != 0;
    rs.stencil_clear_enable = PM4_FIELD(rs.db_render_control, DB_RENDER_CONTROL, STENCIL_CLEAR_ENABLE) != 0;
    rs.depth_read_base = addr_of(rd(st.cx, P::DB_Z_READ_BASE), rd(st.cx, P::DB_Z_READ_BASE_HI));
    rs.depth_write_base = addr_of(rd(st.cx, P::DB_Z_WRITE_BASE), rd(st.cx, P::DB_Z_WRITE_BASE_HI));
    rs.stencil_read_base = addr_of(rd(st.cx, P::DB_STENCIL_READ_BASE), rd(st.cx, P::DB_STENCIL_READ_BASE_HI));
    rs.stencil_write_base = addr_of(rd(st.cx, P::DB_STENCIL_WRITE_BASE), rd(st.cx, P::DB_STENCIL_WRITE_BASE_HI));
    rs.db_depth_view = rd(st.cx, P::DB_DEPTH_VIEW);
    rs.db_render_override = rd(st.cx, P::DB_RENDER_OVERRIDE);
    rs.db_render_override2 = rd(st.cx, P::DB_RENDER_OVERRIDE2);
    rs.htile_data_base = addr_of(rd(st.cx, P::DB_HTILE_DATA_BASE), rd(st.cx, P::DB_HTILE_DATA_BASE_HI));
    rs.db_depth_size_xy = rd(st.cx, P::DB_DEPTH_SIZE_XY);
    rs.db_dfsm_control = rd(st.cx, P::DB_DFSM_CONTROL);
    rs.db_depth_info = rd(st.cx, P::DB_DEPTH_INFO);
    rs.db_z_info = rd(st.cx, P::DB_Z_INFO);
    rs.db_stencil_info = rd(st.cx, P::DB_STENCIL_INFO);
    rs.db_depth_size = rd(st.cx, P::DB_DEPTH_SIZE);
    rs.db_depth_slice = rd(st.cx, P::DB_DEPTH_SLICE);
    rs.db_htile_surface = rd(st.cx, P::DB_HTILE_SURFACE);
    rs.db_rmi_l2_cache_control = rd(st.cx, P::DB_RMI_L2_CACHE_CONTROL);
    rs.db_shader_control = rd(st.cx, P::DB_SHADER_CONTROL);
    rs.stencil_test_val_export_enable =
        PM4_FIELD(rs.db_shader_control, DB_SHADER_CONTROL, STENCIL_TEST_VAL_EXPORT_ENABLE) != 0;
    rs.stencil_op_val_export_enable =
        PM4_FIELD(rs.db_shader_control, DB_SHADER_CONTROL, STENCIL_OP_VAL_EXPORT_ENABLE) != 0;

    // Color blend state (decoded fields of CB_BLEND0_CONTROL).
    const uint32_t bc = rd(st.cx, P::CB_BLEND0_CONTROL);
    rs.blend_enable    = PM4_FIELD(bc, CB_BLEND0_CONTROL, ENABLE) != 0;
    rs.color_src_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
    rs.color_dst_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
    rs.color_comb_fcn  = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_COMB_FCN);
    // Separate alpha blend (#381): the alpha channel has its own factors/op when SEPARATE_ALPHA_BLEND
    // is set (a common UI/premultiplied pattern: color SrcAlpha/OneMinusSrcAlpha but alpha One/OneMinus).
    rs.separate_alpha_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, SEPARATE_ALPHA_BLEND) != 0;
    rs.alpha_src_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, ALPHA_SRCBLEND);
    rs.alpha_dst_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, ALPHA_DESTBLEND);
    rs.alpha_comb_fcn  = PM4_FIELD(bc, CB_BLEND0_CONTROL, ALPHA_COMB_FCN);

    const uint32_t bc1 = rd(st.cx, P::CB_BLEND1_CONTROL);
    rs.blend1_enable = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ENABLE) != 0;
    rs.color1_src_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
    rs.color1_dst_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
    rs.color1_comb_fcn = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_COMB_FCN);
    rs.separate_alpha_blend1 = PM4_FIELD(bc1, CB_BLEND0_CONTROL, SEPARATE_ALPHA_BLEND) != 0;
    rs.alpha1_src_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_SRCBLEND);
    rs.alpha1_dst_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_DESTBLEND);
    rs.alpha1_comb_fcn = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_COMB_FCN);

    // Faithful raw state registers.
    rs.db_depth_control  = dc;
    // Rasterizer cull/front-face/polygon mode (#456). Absent -> 0 -> CULL_NONE/CCW/FILL (prior default).
    rs.pa_su_sc_mode_cntl = rd(st.cx, P::PA_SU_SC_MODE_CNTL);
    // Stencil op + ref/mask registers (absent -> 0; stencil_enable already gates whether they apply).
    rs.db_stencil_control   = st.cx.count(P::DB_STENCIL_CONTROL)   ? rd(st.cx, P::DB_STENCIL_CONTROL)   : 0u;
    rs.db_stencilrefmask    = st.cx.count(P::DB_STENCILREFMASK)    ? rd(st.cx, P::DB_STENCILREFMASK)    : 0u;
    rs.db_stencilrefmask_bf = st.cx.count(P::DB_STENCILREFMASK_BF) ? rd(st.cx, P::DB_STENCILREFMASK_BF) : 0u;
    rs.cb_color_control  = rd(st.cx, P::CB_COLOR_CONTROL);
    rs.cb_blend0_control = bc;
    // CB_TARGET_MASK (per-MRT color write mask). The AGC driver defaults it to write-all when the game
    // does not program register 0x8E explicitly — and this game NEVER emits it (verified with
    // PROSPER_RESDUMP: the title/cutscene bind CB_COLOR0_BASE/INFO but no 0x8E), relying on that default.
    // Our register file starts empty, so an ABSENT CB_TARGET_MASK must resolve to write-all (0xF per MRT),
    // NOT 0 — reading 0 dropped every color draw (color_write_mask==0 skip), i.e. the "blue screen" where
    // the real art rendered only under PROSPER_FORCE_COLORWRITE. A register that IS present (even 0) is
    // honored, so a genuine depth-only pass that programs mask 0 still skips correctly.
    rs.cb_target_mask    = st.cx.count(P::CB_TARGET_MASK) ? rd(st.cx, P::CB_TARGET_MASK) : 0xFFFFFFFFu;

    // Viewport 0 transform (guest floats; all-zero when never programmed).
    rs.vport_xscale  = flt(rd(st.cx, P::PA_CL_VPORT_XSCALE));
    rs.vport_xoffset = flt(rd(st.cx, P::PA_CL_VPORT_XOFFSET));
    rs.vport_yscale  = flt(rd(st.cx, P::PA_CL_VPORT_YSCALE));
    rs.vport_yoffset = flt(rd(st.cx, P::PA_CL_VPORT_YOFFSET));
    rs.vport_zscale  = flt(rd(st.cx, P::PA_CL_VPORT_ZSCALE));
    rs.vport_zoffset = flt(rd(st.cx, P::PA_CL_VPORT_ZOFFSET));

    // AMD's rasterization contract intersects screen, window, generic, and optionally viewport
    // scissors. Window-space rectangles apply PA_SC_WINDOW_OFFSET independently unless their own
    // WINDOW_OFFSET_DISABLE bit is set. AGC initializes every rectangle to [0,16384), so missing
    // members of a partially-programmed state use that published driver default rather than zero.
    const uint32_t scissor_registers[] = {
        P::PA_SC_SCREEN_SCISSOR_TL, P::PA_SC_SCREEN_SCISSOR_BR,
        P::PA_SC_WINDOW_OFFSET, P::PA_SC_WINDOW_SCISSOR_TL, P::PA_SC_WINDOW_SCISSOR_BR,
        P::PA_SC_GENERIC_SCISSOR_TL, P::PA_SC_GENERIC_SCISSOR_BR,
        P::PA_SC_VPORT_SCISSOR_0_TL, P::PA_SC_VPORT_SCISSOR_0_BR, P::PA_SC_MODE_CNTL_0,
    };
    rs.has_scissor = std::any_of(std::begin(scissor_registers), std::end(scissor_registers),
                                 [&](uint32_t reg) { return st.cx.count(reg) != 0; });
    if (rs.has_scissor) {
        const uint32_t screen_tl = st.cx.count(P::PA_SC_SCREEN_SCISSOR_TL)
            ? rd(st.cx, P::PA_SC_SCREEN_SCISSOR_TL) : 0u;
        const uint32_t screen_br = st.cx.count(P::PA_SC_SCREEN_SCISSOR_BR)
            ? rd(st.cx, P::PA_SC_SCREEN_SCISSOR_BR) : 0x40004000u;
        const uint32_t window_offset = st.cx.count(P::PA_SC_WINDOW_OFFSET)
            ? rd(st.cx, P::PA_SC_WINDOW_OFFSET) : 0u;
        const uint32_t window_tl = st.cx.count(P::PA_SC_WINDOW_SCISSOR_TL)
            ? rd(st.cx, P::PA_SC_WINDOW_SCISSOR_TL) : 0x80000000u;
        const uint32_t window_br = st.cx.count(P::PA_SC_WINDOW_SCISSOR_BR)
            ? rd(st.cx, P::PA_SC_WINDOW_SCISSOR_BR) : 0x40004000u;
        const uint32_t generic_tl = st.cx.count(P::PA_SC_GENERIC_SCISSOR_TL)
            ? rd(st.cx, P::PA_SC_GENERIC_SCISSOR_TL) : 0x80000000u;
        const uint32_t generic_br = st.cx.count(P::PA_SC_GENERIC_SCISSOR_BR)
            ? rd(st.cx, P::PA_SC_GENERIC_SCISSOR_BR) : 0x40004000u;
        const uint32_t viewport_tl = st.cx.count(P::PA_SC_VPORT_SCISSOR_0_TL)
            ? rd(st.cx, P::PA_SC_VPORT_SCISSOR_0_TL) : 0x80000000u;
        const uint32_t viewport_br = st.cx.count(P::PA_SC_VPORT_SCISSOR_0_BR)
            ? rd(st.cx, P::PA_SC_VPORT_SCISSOR_0_BR) : 0x40004000u;
        const uint32_t mode = st.cx.count(P::PA_SC_MODE_CNTL_0)
            ? rd(st.cx, P::PA_SC_MODE_CNTL_0) : 0x2u;

        const int32_t offset_x = static_cast<int16_t>(PM4_FIELD(
            window_offset, PA_SC_WINDOW_OFFSET, WINDOW_X_OFFSET));
        const int32_t offset_y = static_cast<int16_t>(PM4_FIELD(
            window_offset, PA_SC_WINDOW_OFFSET, WINDOW_Y_OFFSET));
        const auto screen = [](uint32_t value) {
            return std::max<int32_t>(0, static_cast<int16_t>(value));
        };

        int32_t left = screen(PM4_FIELD(screen_tl, PA_SC_SCREEN_SCISSOR_TL, TL_X));
        int32_t top = screen(PM4_FIELD(screen_tl, PA_SC_SCREEN_SCISSOR_TL, TL_Y));
        int32_t right = screen(PM4_FIELD(screen_br, PA_SC_SCREEN_SCISSOR_BR, BR_X));
        int32_t bottom = screen(PM4_FIELD(screen_br, PA_SC_SCREEN_SCISSOR_BR, BR_Y));
        auto intersect = [&](uint32_t tl_x, uint32_t tl_y, uint32_t br_x, uint32_t br_y,
                             bool offset_disabled) {
            left = std::max(left, add_scissor_offset(tl_x, offset_x, offset_disabled));
            top = std::max(top, add_scissor_offset(tl_y, offset_y, offset_disabled));
            right = std::min(right, add_scissor_offset(br_x, offset_x, offset_disabled));
            bottom = std::min(bottom, add_scissor_offset(br_y, offset_y, offset_disabled));
        };
        intersect(PM4_FIELD(window_tl, PA_SC_WINDOW_SCISSOR_TL, TL_X),
                  PM4_FIELD(window_tl, PA_SC_WINDOW_SCISSOR_TL, TL_Y),
                  PM4_FIELD(window_br, PA_SC_WINDOW_SCISSOR_BR, BR_X),
                  PM4_FIELD(window_br, PA_SC_WINDOW_SCISSOR_BR, BR_Y),
                  PM4_FIELD(window_tl, PA_SC_WINDOW_SCISSOR_TL, WINDOW_OFFSET_DISABLE) != 0);
        intersect(PM4_FIELD(generic_tl, PA_SC_GENERIC_SCISSOR_TL, TL_X),
                  PM4_FIELD(generic_tl, PA_SC_GENERIC_SCISSOR_TL, TL_Y),
                  PM4_FIELD(generic_br, PA_SC_GENERIC_SCISSOR_BR, BR_X),
                  PM4_FIELD(generic_br, PA_SC_GENERIC_SCISSOR_BR, BR_Y),
                  PM4_FIELD(generic_tl, PA_SC_GENERIC_SCISSOR_TL, WINDOW_OFFSET_DISABLE) != 0);
        if (PM4_FIELD(mode, PA_SC_MODE_CNTL_0, VPORT_SCISSOR_ENABLE))
            intersect(PM4_FIELD(viewport_tl, PA_SC_VPORT_SCISSOR_0_TL, TL_X),
                      PM4_FIELD(viewport_tl, PA_SC_VPORT_SCISSOR_0_TL, TL_Y),
                      PM4_FIELD(viewport_br, PA_SC_VPORT_SCISSOR_0_BR, BR_X),
                      PM4_FIELD(viewport_br, PA_SC_VPORT_SCISSOR_0_BR, BR_Y),
                      PM4_FIELD(viewport_tl, PA_SC_VPORT_SCISSOR_0_TL,
                                WINDOW_OFFSET_DISABLE) != 0);
        rs.scissor_left = left; rs.scissor_top = top;
        rs.scissor_right = right; rs.scissor_bottom = bottom;
    }

    return rs;
}

ResolvedPipelineState resolve_pipeline_state(const RenderState& rs) {
    ResolvedPipelineState ps;
    ps.topology      = static_cast<uint32_t>(vk_topology(rs.prim_type));
    ps.color0_format = static_cast<uint32_t>(
        vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap));
    ps.color1_format = rs.color1_format ? static_cast<uint32_t>(
        vk_color_format(rs.color1_format, rs.color1_number_type, rs.color1_comp_swap)) : 0u;

    // Fast-clear color: decode the CB_COLOR0_CLEAR_WORD when the game programmed one (#309). A
    // decode miss (no fast-clear, or an unmapped format) leaves has_clear_color false and the
    // default opaque-black clear_color, which the backend uses instead of the old debug blue.
    ps.has_clear_color = decode_clear_color(
        rs.color0_format, rs.color0_number_type, rs.color0_comp_swap,
        rs.color0_has_clear, rs.color0_clear_word0, ps.clear_color);
    ps.has_clear_color1 = decode_clear_color(
        rs.color1_format, rs.color1_number_type, rs.color1_comp_swap,
        rs.color1_has_clear, rs.color1_clear_word0, ps.clear_color1);

    ps.depth_test_enable  = rs.z_enable;
    ps.depth_write_enable = rs.z_write_enable;
    ps.depth_compare_op   = vk_compare_op(rs.zfunc);
    // Depth clear value (#371): use the guest's DB_DEPTH_CLEAR when programmed. Otherwise pick a default
    // matching the resolved compare op — LESS/LESS_OR_EQUAL clear to 1.0 (far), GREATER/GREATER_OR_EQUAL
    // (reversed-Z) clear to 0.0 (near) — the overwhelmingly common convention, and never a fixed 0.5
    // (which discards every fragment on the wrong side of 0.5 under a standard depth test).
    if (rs.has_depth_clear) {
        ps.depth_clear_value = rs.depth_clear_value;
    } else {
        const uint32_t op = ps.depth_compare_op;   // VkCompareOp: LESS=1, LEQUAL=3, GREATER=4, GEQUAL=6
        ps.depth_clear_value = (op == 4u || op == 6u) ? 0.0f : 1.0f;
    }
    ps.stencil_clear_value = rs.stencil_clear_value;
    ps.has_depth_clear = rs.has_depth_clear;
    ps.has_stencil_clear = rs.has_stencil_clear;
    ps.db_render_control = rs.db_render_control;
    ps.depth_clear_enable = rs.depth_clear_enable;
    ps.stencil_clear_enable = rs.stencil_clear_enable;
    ps.depth_read_base = rs.depth_read_base;
    ps.depth_write_base = rs.depth_write_base;
    ps.stencil_read_base = rs.stencil_read_base;
    ps.stencil_write_base = rs.stencil_write_base;
    ps.db_depth_view = rs.db_depth_view;
    ps.db_render_override = rs.db_render_override;
    ps.db_render_override2 = rs.db_render_override2;
    ps.htile_data_base = rs.htile_data_base;
    ps.db_depth_size_xy = rs.db_depth_size_xy;
    ps.db_dfsm_control = rs.db_dfsm_control;
    ps.db_depth_info = rs.db_depth_info;
    ps.db_z_info = rs.db_z_info;
    ps.db_stencil_info = rs.db_stencil_info;
    ps.db_depth_size = rs.db_depth_size;
    ps.db_depth_slice = rs.db_depth_slice;
    ps.db_htile_surface = rs.db_htile_surface;
    ps.db_rmi_l2_cache_control = rs.db_rmi_l2_cache_control;
    ps.db_shader_control = rs.db_shader_control;
    ps.stencil_test_val_export_enable = rs.stencil_test_val_export_enable;
    ps.stencil_op_val_export_enable = rs.stencil_op_val_export_enable;

    // Stencil: compare func from DB_DEPTH_CONTROL (STENCILFUNC / _BF), ops from DB_STENCIL_CONTROL,
    // ref/masks from DB_STENCILREFMASK[_BF]. [0]=front, [1]=back. Only meaningful when stencil_enable.
    ps.stencil_enable = rs.stencil_enable;
    if (rs.stencil_enable) {
        const uint32_t dc2 = rs.db_depth_control, sc = rs.db_stencil_control;
        const uint32_t rm = rs.db_stencilrefmask, rmb = rs.db_stencilrefmask_bf;
        ps.stencil_compare_op[0]    = vk_compare_op(PM4_FIELD(dc2, DB_DEPTH_CONTROL, STENCILFUNC));
        ps.stencil_compare_op[1]    = vk_compare_op(PM4_FIELD(dc2, DB_DEPTH_CONTROL, STENCILFUNC_BF));
        ps.stencil_fail_op[0]       = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILFAIL));
        ps.stencil_pass_op[0]       = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZPASS));
        ps.stencil_depth_fail_op[0] = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZFAIL));
        ps.stencil_fail_op[1]       = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILFAIL_BF));
        ps.stencil_pass_op[1]       = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZPASS_BF));
        ps.stencil_depth_fail_op[1] = vk_stencil_op(PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZFAIL_BF));
        ps.stencil_ref[0]           = PM4_FIELD(rm,  DB_STENCILREFMASK,    STENCILTESTVAL);
        ps.stencil_op_val[0]        = PM4_FIELD(rm,  DB_STENCILREFMASK,    STENCILOPVAL);
        ps.stencil_compare_mask[0]  = PM4_FIELD(rm,  DB_STENCILREFMASK,    STENCILMASK);
        ps.stencil_write_mask[0]    = PM4_FIELD(rm,  DB_STENCILREFMASK,    STENCILWRITEMASK);
        ps.stencil_ref[1]           = PM4_FIELD(rmb, DB_STENCILREFMASK_BF, STENCILTESTVAL_BF);
        ps.stencil_op_val[1]        = PM4_FIELD(rmb, DB_STENCILREFMASK_BF, STENCILOPVAL_BF);
        ps.stencil_compare_mask[1]  = PM4_FIELD(rmb, DB_STENCILREFMASK_BF, STENCILMASK_BF);
        ps.stencil_write_mask[1]    = PM4_FIELD(rmb, DB_STENCILREFMASK_BF, STENCILWRITEMASK_BF);
        // #466: ONES(2)/REPLACE_TEST(3)/REPLACE_OP(4) all resolve to VK REPLACE but write DIFFERENT
        // values — ONES writes the constant 0xFF, REPLACE_TEST writes STENCILTESTVAL, REPLACE_OP writes
        // STENCILOPVAL. The backend feeds stencil_op_val as the Vk REPLACE `reference` (does_replace path),
        // so set it to what the face's replace op actually writes. Vulkan has ONE reference per face, so a
        // face mixing replace variants can't be fully represented — precedence ONES > REPLACE_TEST (the
        // common case is a single replace op per face, e.g. an ALWAYS-compare stencil-prime that writes
        // 0xFF). Runs BEFORE the #377 back-face mirror so a mirrored back inherits the corrected value.
        const uint32_t raw_ops[2][3] = {
            { PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILFAIL),    PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZPASS),    PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZFAIL) },
            { PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILFAIL_BF), PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZPASS_BF), PM4_FIELD(sc, DB_STENCIL_CONTROL, STENCILZFAIL_BF) },
        };
        for (int fb = 0; fb < 2; fb++)
            for (int k = 0; k < 3; k++) ps.raw_stencil_op[fb][k] = raw_ops[fb][k] & 0xFu;
        for (int fb = 0; fb < 2; fb++) {
            bool ones = false, rtest = false;
            for (int k = 0; k < 3; k++) { uint32_t o = raw_ops[fb][k] & 0xFu; if (o == 2u) ones = true; else if (o == 3u) rtest = true; }
            if (ones)       ps.stencil_op_val[fb] = 0xFFu;                // ONES writes the constant 0xFF
            else if (rtest) ps.stencil_op_val[fb] = ps.stencil_ref[fb];  // REPLACE_TEST writes STENCILTESTVAL
            // else REPLACE_OP (or no replace): keep STENCILOPVAL, already set.
        }
        // DB_DEPTH_CONTROL.BACKFACE_ENABLE == 0 means the FRONT state applies to BOTH faces (the _BF
        // registers are ignored). Sourcing back from _BF regardless left back faces with the
        // unprogrammed _BF defaults — STENCILFUNC_BF=0 -> VK_COMPARE_OP_NEVER, dropping every back
        // face out of a two-sided stencil pass. Match RDNA2 (Kyty GraphicsRender: else back = front). #377
        if (PM4_FIELD(dc2, DB_DEPTH_CONTROL, BACKFACE_ENABLE) == 0) {
            ps.stencil_compare_op[1]    = ps.stencil_compare_op[0];
            ps.stencil_fail_op[1]       = ps.stencil_fail_op[0];
            ps.stencil_pass_op[1]       = ps.stencil_pass_op[0];
            ps.stencil_depth_fail_op[1] = ps.stencil_depth_fail_op[0];
            ps.stencil_ref[1]           = ps.stencil_ref[0];
            ps.stencil_op_val[1]        = ps.stencil_op_val[0];
            ps.stencil_compare_mask[1]  = ps.stencil_compare_mask[0];
            ps.stencil_write_mask[1]    = ps.stencil_write_mask[0];
        }
    }

    ps.blend_enable            = rs.blend_enable;
    ps.src_color_blend_factor  = vk_blend_factor(rs.color_src_blend);
    ps.dst_color_blend_factor  = vk_blend_factor(rs.color_dst_blend);
    ps.color_blend_op          = vk_blend_op(rs.color_comb_fcn);
    // Alpha blend factors/op (#381): use the separate ALPHA_* fields when SEPARATE_ALPHA_BLEND is set,
    // else mirror the color factors (RDNA2 behavior — Kyty GraphicsRender). The backend reads these
    // directly, so it no longer has to reuse the color factors for the alpha channel (which corrupted
    // stored dst-alpha on any target later read back — RTT composites, alpha-test/text layers).
    if (rs.separate_alpha_blend) {
        ps.src_alpha_blend_factor = vk_blend_factor(rs.alpha_src_blend);
        ps.dst_alpha_blend_factor = vk_blend_factor(rs.alpha_dst_blend);
        ps.alpha_blend_op         = vk_blend_op(rs.alpha_comb_fcn);
    } else {
        ps.src_alpha_blend_factor = ps.src_color_blend_factor;
        ps.dst_alpha_blend_factor = ps.dst_color_blend_factor;
        ps.alpha_blend_op         = ps.color_blend_op;
    }

    ps.blend1_enable = rs.blend1_enable;
    ps.src_color_blend_factor1 = vk_blend_factor(rs.color1_src_blend);
    ps.dst_color_blend_factor1 = vk_blend_factor(rs.color1_dst_blend);
    ps.color_blend_op1 = vk_blend_op(rs.color1_comb_fcn);
    if (rs.separate_alpha_blend1) {
        ps.src_alpha_blend_factor1 = vk_blend_factor(rs.alpha1_src_blend);
        ps.dst_alpha_blend_factor1 = vk_blend_factor(rs.alpha1_dst_blend);
        ps.alpha_blend_op1 = vk_blend_op(rs.alpha1_comb_fcn);
    } else {
        ps.src_alpha_blend_factor1 = ps.src_color_blend_factor1;
        ps.dst_alpha_blend_factor1 = ps.dst_color_blend_factor1;
        ps.alpha_blend_op1 = ps.color_blend_op1;
    }

    // CB_TARGET_MASK holds a 4-bit write mask per MRT; MRT0 is bits [3:0]. RDNA2's R/G/B/A bit order
    // matches VkColorComponentFlags (R=1,G=2,B=4,A=8), so the nibble maps 1:1.
    ps.color_write_mask = rs.cb_target_mask & 0xFu;
    ps.color1_write_mask = (rs.cb_target_mask >> 4) & 0xFu;

    // Viewport: hardware maps screen = offset + scale * ndc; Vulkan maps px = (x + w/2) + ndc * (w/2).
    // Equating the two: x = xoffset - xscale, w = 2*xscale (same for y). A guest with GNM's +Y-up NDC
    // programs a NEGATIVE yscale, which lands here as a negative-height Vulkan viewport (core 1.1) —
    // the exact hardware Y orientation, with no flip heuristics. Depth: hw z = zoffset + zscale * ndc_z
    // and Vulkan z = min + ndc_z * (max - min), so min = zoffset, max = zoffset + zscale (clamped to
    // Vulkan's required [0,1]; a zero zscale/zoffset pair means "never programmed" -> default 0..1).
    if (rs.vport_xscale != 0.0f && rs.vport_yscale != 0.0f) {
        ps.has_viewport = true;
        ps.viewport_x   = rs.vport_xoffset - rs.vport_xscale;
        ps.viewport_w   = 2.0f * rs.vport_xscale;
        ps.viewport_y   = rs.vport_yoffset - rs.vport_yscale;
        ps.viewport_h   = 2.0f * rs.vport_yscale;
        if (rs.vport_zscale != 0.0f || rs.vport_zoffset != 0.0f) {
            ps.min_depth = std::clamp(rs.vport_zoffset, 0.0f, 1.0f);
            ps.max_depth = std::clamp(rs.vport_zoffset + rs.vport_zscale, 0.0f, 1.0f);
        }
    }

    ps.has_scissor = rs.has_scissor;
    if (rs.has_scissor) {
        ps.scissor_left = rs.scissor_left;
        ps.scissor_top = rs.scissor_top;
        ps.scissor_right = std::max(rs.scissor_right, rs.scissor_left);
        ps.scissor_bottom = std::max(rs.scissor_bottom, rs.scissor_top);
    }

    // Rasterizer cull/front-face/polygon mode from PA_SU_SC_MODE_CNTL (#456). CULL_FRONT[0]/CULL_BACK[1]
    // -> VkCullModeFlags (FRONT_BIT=1, BACK_BIT=2). AMD FACE[2] and VkFrontFace have the same encoding:
    // 0 = CCW is front, 1 = CW is front. The guest's negative Y scale is already represented by the
    // Vulkan viewport and does not change this field mapping. POLY_MODE[4:3]==0 -> FILL; otherwise the
    // front-face
    // PTYPE[7:5] (0=points,1=lines,2=triangles) selects VkPolygonMode (POINT=2, LINE=1, FILL=0).
    const uint32_t su = rs.pa_su_sc_mode_cntl;
    ps.cull_mode  = (PM4_FIELD(su, PA_SU_SC_MODE_CNTL, CULL_FRONT) ? 1u : 0u)
                  | (PM4_FIELD(su, PA_SU_SC_MODE_CNTL, CULL_BACK)  ? 2u : 0u);
    ps.front_face = PM4_FIELD(su, PA_SU_SC_MODE_CNTL, FACE);
    if (PM4_FIELD(su, PA_SU_SC_MODE_CNTL, POLY_MODE) == 0u) {
        ps.polygon_mode = 0u;   // FILL
    } else {
        switch (PM4_FIELD(su, PA_SU_SC_MODE_CNTL, POLYMODE_FRONT_PTYPE)) {
            case 0u:  ps.polygon_mode = 2u; break;   // points -> VK_POLYGON_MODE_POINT
            case 1u:  ps.polygon_mode = 1u; break;   // lines  -> VK_POLYGON_MODE_LINE
            default:  ps.polygon_mode = 0u; break;   // triangles / other -> FILL
        }
    }

    return ps;
}

void scale_resolved_render_area(ResolvedPipelineState& ps, float scale_x, float scale_y) {
    if (ps.has_viewport) {
        ps.viewport_x *= scale_x; ps.viewport_w *= scale_x;
        ps.viewport_y *= scale_y; ps.viewport_h *= scale_y;
    }
    if (ps.has_scissor) {
        ps.scissor_left = scale_scissor_boundary(ps.scissor_left, scale_x, true);
        ps.scissor_top = scale_scissor_boundary(ps.scissor_top, scale_y, true);
        ps.scissor_right = scale_scissor_boundary(ps.scissor_right, scale_x, false);
        ps.scissor_bottom = scale_scissor_boundary(ps.scissor_bottom, scale_y, false);
        ps.scissor_right = std::max(ps.scissor_right, ps.scissor_left);
        ps.scissor_bottom = std::max(ps.scissor_bottom, ps.scissor_top);
    }
}

} // namespace prosper::gpu
