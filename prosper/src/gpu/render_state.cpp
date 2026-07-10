// render_state.cpp — see render_state.hpp.
#include "render_state.hpp"
#include "pm4_registers.hpp"
#include "vk_translate.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

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
bool decode_clear_color(const RenderState& rs, float out[4]) {
    if (!rs.color0_has_clear) return false;
    const uint32_t w0 = rs.color0_clear_word0;
    if (rs.color0_format == 0xAu) {  // COLOR_8_8_8_8
        const float b0 = (w0 & 0xFFu) / 255.0f, b1 = ((w0 >> 8) & 0xFFu) / 255.0f,
                    b2 = ((w0 >> 16) & 0xFFu) / 255.0f, b3 = ((w0 >> 24) & 0xFFu) / 255.0f;
        float r, g, b, a = b3;                     // alpha is byte 3 for both swaps
        if (rs.color0_comp_swap == 1u) { r = b2; g = b1; b = b0; }   // ALT: B8G8R8A8 (byte0=B)
        else                            { r = b0; g = b1; b = b2; }  // STD: R8G8B8A8 (byte0=R)
        const bool is_srgb = (rs.color0_number_type == 6u);
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

    // Color MRT 0 (context register file).
    rs.color0_base            = addr_of(rd(st.cx, P::CB_COLOR0_BASE), rd(st.cx, P::CB_COLOR0_BASE_EXT));
    const uint32_t cinfo       = rd(st.cx, P::CB_COLOR0_INFO);
    rs.color0_format           = PM4_FIELD(cinfo, CB_COLOR0_INFO, FORMAT);
    rs.color0_number_type      = PM4_FIELD(cinfo, CB_COLOR0_INFO, NUMBER_TYPE);
    rs.color0_comp_swap        = PM4_FIELD(cinfo, CB_COLOR0_INFO, COMP_SWAP);

    // CB fast-clear color (CB_COLOR0_CLEAR_WORD0/1). Present only when the game programs a fast-clear
    // for MRT 0; the words carry the clear value in the target's pixel format (decoded in resolve).
    rs.color0_has_clear   = st.cx.count(P::CB_COLOR0_CLEAR_WORD0) || st.cx.count(P::CB_COLOR0_CLEAR_WORD1);
    rs.color0_clear_word0 = st.cx.count(P::CB_COLOR0_CLEAR_WORD0) ? rd(st.cx, P::CB_COLOR0_CLEAR_WORD0) : 0u;
    rs.color0_clear_word1 = st.cx.count(P::CB_COLOR0_CLEAR_WORD1) ? rd(st.cx, P::CB_COLOR0_CLEAR_WORD1) : 0u;

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

    // Color blend state (decoded fields of CB_BLEND0_CONTROL).
    const uint32_t bc = rd(st.cx, P::CB_BLEND0_CONTROL);
    rs.blend_enable    = PM4_FIELD(bc, CB_BLEND0_CONTROL, ENABLE) != 0;
    rs.color_src_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
    rs.color_dst_blend = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
    rs.color_comb_fcn  = PM4_FIELD(bc, CB_BLEND0_CONTROL, COLOR_COMB_FCN);

    // Faithful raw state registers.
    rs.db_depth_control  = dc;
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

    return rs;
}

ResolvedPipelineState resolve_pipeline_state(const RenderState& rs) {
    ResolvedPipelineState ps;
    ps.topology      = static_cast<uint32_t>(vk_topology(rs.prim_type));
    ps.color0_format = static_cast<uint32_t>(
        vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap));

    // Fast-clear color: decode the CB_COLOR0_CLEAR_WORD when the game programmed one (#309). A
    // decode miss (no fast-clear, or an unmapped format) leaves has_clear_color false and the
    // default opaque-black clear_color, which the backend uses instead of the old debug blue.
    ps.has_clear_color = decode_clear_color(rs, ps.clear_color);

    ps.depth_test_enable  = rs.z_enable;
    ps.depth_write_enable = rs.z_write_enable;
    ps.depth_compare_op   = vk_compare_op(rs.zfunc);

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

    // CB_TARGET_MASK holds a 4-bit write mask per MRT; MRT0 is bits [3:0]. RDNA2's R/G/B/A bit order
    // matches VkColorComponentFlags (R=1,G=2,B=4,A=8), so the nibble maps 1:1.
    ps.color_write_mask = rs.cb_target_mask & 0xFu;

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

    return ps;
}

} // namespace prosper::gpu
