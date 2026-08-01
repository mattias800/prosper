// render_state.cpp — see render_state.hpp.
#include "render_state.hpp"
#include "pm4_registers.hpp"
#include "vk_translate.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>

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

// CB_COLOR_CONTROL.MODE selects what the color block DOES with a rasterized primitive, and only
// NORMAL blends the bound pixel shader's color export into the target. prosper models three of the
// other values as their own operation: DISABLE suppresses color writes, RESOLVE becomes cb_resolve,
// and DCC_DECOMPRESS keeps the AGC helper-program handling in gpu_execute.hpp. Every REMAINING
// value of the 3-bit field is reported here. ELIMINATE_FAST_CLEAR(2) is a color-block metadata
// operation that hardware performs INSTEAD of shading and prosper still runs as an ordinary color
// draw — that gap is #1588. Values 4, 5 and 7 are grouped with it because they are simply "not one
// of the four prosper models", NOT because their operation is known: 4 and 5 are decompress modes
// in the published enum, 7 is not defined there, and no title here exercises any of them.
bool cb_mode_is_unmodeled_metadata_operation(uint32_t mode) {
    return mode != P::CB_COLOR_CONTROL_MODE_DISABLE &&
           mode != P::CB_COLOR_CONTROL_MODE_NORMAL &&
           mode != P::CB_COLOR_CONTROL_MODE_RESOLVE &&
           mode != P::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS;
}

std::atomic<uint64_t> g_unmodeled_cb_mode_counts[P::CB_COLOR_CONTROL_MODE_MASK + 1u];

void report_unmodeled_cb_color_mode(uint32_t mode) {
    const uint64_t count =
        ++g_unmodeled_cb_mode_counts[mode & P::CB_COLOR_CONTROL_MODE_MASK];
    // This line used to dedupe on a bitmask of modes already seen, so a whole run emitted exactly
    // one line per distinct value and "once" was indistinguishable from "hundreds of thousands".
    // #1588 therefore had to be sized from a separate PROSPER_COLORSTATETRACE run — the same
    // under-reporting recorded as instruments #9 and #13 in docs/GAME_COMPAT_ORCHESTRATION.md.
    // Print at powers of two so the volume stays bounded on a title that does this every frame, and
    // carry the count in EVERY line, so a printed number is that occurrence's exact ordinal rather
    // than an unreadable cap. It is NOT the total: the highest ordinal printed is a lower bound on
    // it, and because draws realize on parallel workers the lines can also arrive out of order. Use
    // unmodeled_cb_color_mode_count() for the total — that is what makes per-title exposure
    // measurable for #1706. The atomic pre-increment gives every caller a distinct value, so no
    // power-of-two line is duplicated or lost when threads race here.
    if ((count & (count - 1u)) == 0u)
        fprintf(stderr, "[gpu] resolve_pipeline_state: CB_COLOR_CONTROL.MODE=%u is an unmodeled "
                        "color-block metadata operation -> still executed as an ordinary color "
                        "draw (#1588; count=%llu)\n",
                mode, static_cast<unsigned long long>(count));
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

uint64_t unmodeled_cb_color_mode_count(uint32_t mode) {
    return g_unmodeled_cb_mode_counts[mode & P::CB_COLOR_CONTROL_MODE_MASK].load();
}

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
    rs.ps_wave32 = PM4_FIELD(rd(st.cx, P::SPI_PS_IN_CONTROL),
                             SPI_PS_IN_CONTROL, PS_W32_EN) != 0;

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

    // Diagnostic (PROSPER_MSAA_LOG): report MSAA sample/fragment programming and the CB color mode so we
    // can tell whether a title uses MSAA + hardware resolve (CB_COLOR_CONTROL.MODE=3), which prosper renders
    // single-sample and does not yet resolve. Gated; no default behavior change.
    if (getenv("PROSPER_MSAA_LOG")) {
        const uint32_t cattr    = st.cx.count(P::CB_COLOR0_ATTRIB)   ? rd(st.cx, P::CB_COLOR0_ATTRIB)   : 0u;
        const uint32_t aacfg    = st.cx.count(P::PA_SC_AA_CONFIG)    ? rd(st.cx, P::PA_SC_AA_CONFIG)    : 0u;
        const uint32_t modecntl = st.cx.count(P::PA_SC_MODE_CNTL_0)  ? rd(st.cx, P::PA_SC_MODE_CNTL_0)  : 0u;
        const uint32_t ctrl     = st.cx.count(P::CB_COLOR_CONTROL)   ? rd(st.cx, P::CB_COLOR_CONTROL)   : 0u;
        const uint32_t nsamp  = PM4_FIELD(cattr, CB_COLOR0_ATTRIB, NUM_SAMPLES);
        const uint32_t aa_ns  = PM4_FIELD(aacfg, PA_SC_AA_CONFIG, MSAA_NUM_SAMPLES);
        const uint32_t msaaen = PM4_FIELD(modecntl, PA_SC_MODE_CNTL_0, MSAA_ENABLE);
        const uint32_t cbmode = PM4_FIELD(ctrl, CB_COLOR_CONTROL, MODE);
        const uint64_t c1base  = addr_of(rd(st.cx, P::CB_COLOR1_BASE), rd(st.cx, P::CB_COLOR1_BASE_EXT));
        const uint32_t tmask   = st.cx.count(P::CB_TARGET_MASK) ? rd(st.cx, P::CB_TARGET_MASK) : 0xffffffffu;
        fprintf(stderr, "[msaa] color0=0x%llx fmt=%u %ux%u ATTRIB.NUM_SAMPLES=%u(=%ux) "
                        "AA.NUM_SAMPLES=%u(=%ux) MSAA_ENABLE=%u CB_MODE=%u color1=0x%llx tgtmask=0x%x\n",
                (unsigned long long)rs.color0_base, rs.color0_format, rs.color0_width, rs.color0_height,
                nsamp, 1u << nsamp, aa_ns, 1u << aa_ns, msaaen, cbmode,
                (unsigned long long)c1base, tmask);
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

    // Complete RDNA2 MRT programming. CB_COLORn's main block has a 0xf-register stride; BASE_EXT
    // and ATTRIB2 are dense arrays. Keep the named MRT0/MRT1 fields above for source compatibility,
    // while every new consumer uses this hardware-sized array.
    for (uint32_t slot = 0; slot < rs.color_targets.size(); ++slot) {
        constexpr uint32_t kColorRegisterStride = 0xf;
        ColorTargetState& target = rs.color_targets[slot];
        const uint32_t base_reg = P::CB_COLOR0_BASE + slot * kColorRegisterStride;
        const uint32_t info_reg = P::CB_COLOR0_INFO + slot * kColorRegisterStride;
        const uint32_t clear0_reg = P::CB_COLOR0_CLEAR_WORD0 + slot * kColorRegisterStride;
        const uint32_t clear1_reg = clear0_reg + 1u;
        target.base = addr_of(rd(st.cx, base_reg), rd(st.cx, P::CB_COLOR0_BASE_EXT + slot));
        const uint32_t info = rd(st.cx, info_reg);
        target.format = PM4_FIELD(info, CB_COLOR0_INFO, FORMAT);
        target.number_type = PM4_FIELD(info, CB_COLOR0_INFO, NUMBER_TYPE);
        target.comp_swap = PM4_FIELD(info, CB_COLOR0_INFO, COMP_SWAP);
        const auto attrib2 = st.cx.find(P::CB_COLOR0_ATTRIB2 + slot);
        if (attrib2 != st.cx.end()) {
            target.has_extent = true;
            target.width = PM4_FIELD(attrib2->second, CB_COLOR0_ATTRIB2, MIP0_WIDTH) + 1u;
            target.height = PM4_FIELD(attrib2->second, CB_COLOR0_ATTRIB2, MIP0_HEIGHT) + 1u;
        }
        target.has_clear = st.cx.count(clear0_reg) || st.cx.count(clear1_reg);
        target.clear_word0 = st.cx.count(clear0_reg) ? rd(st.cx, clear0_reg) : 0u;
        target.clear_word1 = st.cx.count(clear1_reg) ? rd(st.cx, clear1_reg) : 0u;
    }

    // Primitive topology. VGT_PRIMITIVE_TYPE (0x242) is a UCONFIG register in RDNA2 (the game sets it via
    // a Uc-class SetRegsIndirect / CreatePrimState's uc[2]), NOT a context register — read it from st.uc.
    // (Reading st.cx left it 0 -> the default PointList topology, so triangles rasterized as ~3 points.)
    rs.prim_type = PM4_FIELD(rd(st.uc, P::VGT_PRIMITIVE_TYPE), VGT_PRIMITIVE_TYPE, PRIM_TYPE);
    rs.ge_indx_offset = rd(st.uc, P::GE_INDX_OFFSET);

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
    // #1353: on GFX10 the READ and WRITE base registers describe ONE surface — drivers program
    // both from a single address (the split is a pre-GFX10 TC-compat remnant), so a lone zero on
    // either side is never a coherent state. Blue Prince's Gen5 indirect register arenas carry
    // stale slots that parse as (DB_Z_WRITE_BASE, 0) AFTER the real pair write in the SAME array
    // (PROSPER_DBBASETRACE capture on the issue; the #1335 screen-scissor fold is the same
    // family), leaving every shadow-caster pass with (read=P, write=0) and splitting the
    // persistent-DS identity. Recover the clobbered half from its partner and log fail-visibly;
    // a genuine unbind writes both halves 0 and is untouched. Divergent NONZERO pairs are kept
    // as-is (not observed; hardware would not produce them either).
    const auto recover_pair = [](uint64_t& a, uint64_t& b, const char* which) {
        if (a == b || (a && b)) return;
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 8)
            fprintf(stderr,
                    "[render_state] lone-zero DB %s base recovered from partner "
                    "(read=0x%llx write=0x%llx, #1353 stale arena slot)\n",
                    which, (unsigned long long)a, (unsigned long long)b);
        if (!a) a = b; else b = a;
    };
    recover_pair(rs.depth_read_base, rs.depth_write_base, "Z");
    recover_pair(rs.stencil_read_base, rs.stencil_write_base, "STENCIL");
    rs.db_depth_view = rd(st.cx, P::DB_DEPTH_VIEW);
    rs.db_render_override = rd(st.cx, P::DB_RENDER_OVERRIDE);
    rs.db_render_override2 = rd(st.cx, P::DB_RENDER_OVERRIDE2);
    rs.htile_data_base = addr_of(rd(st.cx, P::DB_HTILE_DATA_BASE), rd(st.cx, P::DB_HTILE_DATA_BASE_HI));
    rs.has_db_depth_size_xy = st.cx.count(P::DB_DEPTH_SIZE_XY) != 0;
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
    rs.disable_rop3    = PM4_FIELD(bc, CB_BLEND0_CONTROL, DISABLE_ROP3) != 0;

    const uint32_t bc1 = rd(st.cx, P::CB_BLEND1_CONTROL);
    rs.blend1_enable = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ENABLE) != 0;
    rs.color1_src_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
    rs.color1_dst_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
    rs.color1_comb_fcn = PM4_FIELD(bc1, CB_BLEND0_CONTROL, COLOR_COMB_FCN);
    rs.separate_alpha_blend1 = PM4_FIELD(bc1, CB_BLEND0_CONTROL, SEPARATE_ALPHA_BLEND) != 0;
    rs.alpha1_src_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_SRCBLEND);
    rs.alpha1_dst_blend = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_DESTBLEND);
    rs.alpha1_comb_fcn = PM4_FIELD(bc1, CB_BLEND0_CONTROL, ALPHA_COMB_FCN);
    rs.disable_rop3_1 = PM4_FIELD(bc1, CB_BLEND0_CONTROL, DISABLE_ROP3) != 0;

    for (uint32_t slot = 0; slot < rs.color_targets.size(); ++slot) {
        ColorTargetState& target = rs.color_targets[slot];
        const uint32_t blend = rd(st.cx, P::CB_BLEND0_CONTROL + slot);
        target.blend_enable = PM4_FIELD(blend, CB_BLEND0_CONTROL, ENABLE) != 0;
        target.color_src_blend = PM4_FIELD(blend, CB_BLEND0_CONTROL, COLOR_SRCBLEND);
        target.color_dst_blend = PM4_FIELD(blend, CB_BLEND0_CONTROL, COLOR_DESTBLEND);
        target.color_comb_fcn = PM4_FIELD(blend, CB_BLEND0_CONTROL, COLOR_COMB_FCN);
        target.separate_alpha_blend =
            PM4_FIELD(blend, CB_BLEND0_CONTROL, SEPARATE_ALPHA_BLEND) != 0;
        target.alpha_src_blend = PM4_FIELD(blend, CB_BLEND0_CONTROL, ALPHA_SRCBLEND);
        target.alpha_dst_blend = PM4_FIELD(blend, CB_BLEND0_CONTROL, ALPHA_DESTBLEND);
        target.alpha_comb_fcn = PM4_FIELD(blend, CB_BLEND0_CONTROL, ALPHA_COMB_FCN);
        target.disable_rop3 = PM4_FIELD(blend, CB_BLEND0_CONTROL, DISABLE_ROP3) != 0;
    }

    // Faithful raw state registers.
    rs.db_depth_control  = dc;
    // Rasterizer cull/front-face/polygon mode (#456). Absent -> 0 -> CULL_NONE/CCW/FILL (prior default).
    rs.pa_su_sc_mode_cntl = rd(st.cx, P::PA_SU_SC_MODE_CNTL);
    rs.pa_su_poly_offset_front_scale  = rd(st.cx, P::PA_SU_POLY_OFFSET_FRONT_SCALE);
    rs.pa_su_poly_offset_front_offset = rd(st.cx, P::PA_SU_POLY_OFFSET_FRONT_OFFSET);
    rs.pa_su_poly_offset_back_scale   = rd(st.cx, P::PA_SU_POLY_OFFSET_BACK_SCALE);
    rs.pa_su_poly_offset_back_offset  = rd(st.cx, P::PA_SU_POLY_OFFSET_BACK_OFFSET);
    rs.pa_su_poly_offset_clamp        = rd(st.cx, P::PA_SU_POLY_OFFSET_CLAMP);
    rs.pa_su_poly_offset_db_fmt_cntl  = rd(st.cx, P::PA_SU_POLY_OFFSET_DB_FMT_CNTL);
    // Stencil op + ref/mask registers (absent -> 0; stencil_enable already gates whether they apply).
    rs.db_stencil_control   = st.cx.count(P::DB_STENCIL_CONTROL)   ? rd(st.cx, P::DB_STENCIL_CONTROL)   : 0u;
    rs.db_stencilrefmask    = st.cx.count(P::DB_STENCILREFMASK)    ? rd(st.cx, P::DB_STENCILREFMASK)    : 0u;
    rs.db_stencilrefmask_bf = st.cx.count(P::DB_STENCILREFMASK_BF) ? rd(st.cx, P::DB_STENCILREFMASK_BF) : 0u;
    // PROSPER_STENCILLOG (gated, off by default): the guest's RAW depth/stencil register dwords per
    // stencil-enabled draw — ground truth for compare/op decode questions (the translated fields
    // above can be audited against exactly what the title programmed). Dedup on change.
    if (rs.stencil_enable && getenv("PROSPER_STENCILLOG")) {
        static thread_local uint32_t last_dc, last_sc, last_rm, last_rmb, last_prim = 0xFFFFFFFFu;
        if (dc != last_dc || rs.db_stencil_control != last_sc ||
            rs.db_stencilrefmask != last_rm || rs.db_stencilrefmask_bf != last_rmb ||
            rs.prim_type != last_prim) {
            last_dc = dc; last_sc = rs.db_stencil_control;
            last_rm = rs.db_stencilrefmask; last_rmb = rs.db_stencilrefmask_bf; last_prim = rs.prim_type;
            fprintf(stderr, "[stencil-raw] dc=%08x sc=%08x rm=%08x rmb=%08x prim=%u backface_en=%u\n",
                    dc, rs.db_stencil_control, rs.db_stencilrefmask, rs.db_stencilrefmask_bf,
                    rs.prim_type, PM4_FIELD(dc, DB_DEPTH_CONTROL, BACKFACE_ENABLE));
        }
    }
    rs.has_cb_color_control = st.cx.count(P::CB_COLOR_CONTROL) != 0;
    rs.cb_color_control  = rd(st.cx, P::CB_COLOR_CONTROL);
    rs.cb_blend0_control = bc;
    // CB_TARGET_MASK (per-MRT color write mask). The AGC driver defaults it to write-all when the game
    // does not program register 0x8E explicitly — and this game NEVER emits it (verified with
    // PROSPER_RESDUMP: the title/cutscene bind CB_COLOR0_BASE/INFO but no 0x8E), relying on that default.
    // Our register file starts empty, so an ABSENT CB_TARGET_MASK must resolve to write-all (0xF per MRT),
    // NOT 0 — reading 0 dropped every color draw (color_write_mask==0 skip), i.e. the "blue screen" where
    // the real art rendered only under PROSPER_FORCE_COLORWRITE. A register that IS present (even 0) is
    // honored, so a genuine depth-only pass that programs mask 0 still skips correctly.
    rs.has_cb_target_mask = st.cx.count(P::CB_TARGET_MASK) != 0;
    rs.cb_target_mask    = st.cx.count(P::CB_TARGET_MASK) ? rd(st.cx, P::CB_TARGET_MASK) : 0xFFFFFFFFu;
    // CB_SHADER_MASK is programmed from the pixel shader's color-export contract. Like the target
    // mask, an absent register must not suppress color in synthetic/legacy command streams. The
    // resolved Vulkan write mask intersects both registers so channels the shader does not export
    // preserve their existing attachment contents (AMD RDNA2 EXP EN semantics, Table 56).
    rs.has_cb_shader_mask = st.cx.count(P::CB_SHADER_MASK) != 0;
    rs.cb_shader_mask    = st.cx.count(P::CB_SHADER_MASK) ? rd(st.cx, P::CB_SHADER_MASK) : 0xFFFFFFFFu;
    rs.spi_shader_col_format = st.cx.count(P::SPI_SHADER_COL_FORMAT)
        ? rd(st.cx, P::SPI_SHADER_COL_FORMAT) : 0u;
    rs.sx_ps_downconvert = st.cx.count(P::SX_PS_DOWNCONVERT)
        ? rd(st.cx, P::SX_PS_DOWNCONVERT) : 0u;

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
        uint32_t screen_tl = st.cx.count(P::PA_SC_SCREEN_SCISSOR_TL)
            ? rd(st.cx, P::PA_SC_SCREEN_SCISSOR_TL) : 0u;
        uint32_t screen_br = st.cx.count(P::PA_SC_SCREEN_SCISSOR_BR)
            ? rd(st.cx, P::PA_SC_SCREEN_SCISSOR_BR) : 0x40004000u;
        // #1335: a PRESENT-but-degenerate screen pair (BR <= TL on either axis) is never a
        // coherently-programmed state — no draw intends a zero-area SCREEN scissor (the same
        // contract as the window-offset recovery below). Blue Prince writes the TL thousands of
        // times per route and the BR exactly once, with 0, folded out of a foreign span inside an
        // indirect-register arena at the Day One transition; honoring it blanks every subsequent
        // draw and freezes the presented frame (full evidence chain on the issue). Real hardware
        // keeps the register's initialized full-screen value because that byte range is not
        // consumed as a register write. Substitute the full default and log fail-visibly; genuine
        // per-pass closes arrive via the WINDOW/GENERIC/VPORT rects, which stay fully honored.
        // Both halves are reset, not only the BR: a degenerate partner poisons trust in the pair
        // (the observed live TL of (0,1024) would otherwise clip 1024 of 1080 rows on its own),
        // and the init state is the only coherent recovery target.
        {
            const auto s16 = [](uint32_t v, uint32_t shift) {
                return static_cast<int32_t>(static_cast<int16_t>((v >> shift) & 0xffffu));
            };
            if (s16(screen_br, 0) <= s16(screen_tl, 0) ||
                s16(screen_br, 16) <= s16(screen_tl, 16)) {
                static std::atomic<int> logged{0};
                if (logged.fetch_add(1) < 8)
                    fprintf(stderr,
                            "[scissor] degenerate SCREEN pair recovered: tl=%08x br=%08x -> "
                            "full (#1335)\n",
                            screen_tl, screen_br);
                screen_tl = 0u;
                screen_br = 0x40004000u;
            }
        }
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

        int32_t left = 0, top = 0, right = 0, bottom = 0;
        // Intersect screen ∩ window ∩ generic ∩ (optional) viewport. `apply_offset` gates whether
        // PA_SC_WINDOW_OFFSET participates at all, so the whole combine can be re-run without it.
        auto combine = [&](bool apply_offset) -> bool {
            left = screen(PM4_FIELD(screen_tl, PA_SC_SCREEN_SCISSOR_TL, TL_X));
            top = screen(PM4_FIELD(screen_tl, PA_SC_SCREEN_SCISSOR_TL, TL_Y));
            right = screen(PM4_FIELD(screen_br, PA_SC_SCREEN_SCISSOR_BR, BR_X));
            bottom = screen(PM4_FIELD(screen_br, PA_SC_SCREEN_SCISSOR_BR, BR_Y));
            const int32_t ox = apply_offset ? offset_x : 0;
            const int32_t oy = apply_offset ? offset_y : 0;
            auto intersect = [&](uint32_t tl_x, uint32_t tl_y, uint32_t br_x, uint32_t br_y,
                                 bool offset_disabled) {
                left = std::max(left, add_scissor_offset(tl_x, ox, offset_disabled));
                top = std::max(top, add_scissor_offset(tl_y, oy, offset_disabled));
                right = std::min(right, add_scissor_offset(br_x, ox, offset_disabled));
                bottom = std::min(bottom, add_scissor_offset(br_y, oy, offset_disabled));
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
            // #1164: a window offset can push the scissor ENTIRELY off the render target while
            // leaving it non-empty in ABSOLUTE coordinates. E.g. a VPORT rect [0,2048) offset by
            // +2560 becomes [2560,4608): right>left (absolutely non-empty), but its overlap with a
            // 2048-wide target is [max(2560,0), min(4608,2048)) = [2560,2048) = empty. Judge
            // emptiness against the color0 extent so the recover below fires for that off-target
            // case too, not only the absolute-degenerate case #1161 handled. Only when the extent
            // is known (a color pass); a depth-only pass has no extent and keeps the absolute test.
            if (rs.color0_has_extent) {
                const int32_t cw = static_cast<int32_t>(rs.color0_width);
                const int32_t ch = static_cast<int32_t>(rs.color0_height);
                return std::min(right, cw) > std::max(left, 0) &&
                       std::min(bottom, ch) > std::max(top, 0);
            }
            return right > left && bottom > top;
        };

        // A non-zero PA_SC_WINDOW_OFFSET that collapses the intersection to empty has pushed an
        // offset-enabled viewport/window rectangle entirely off the render target. An emitted draw
        // never intends an empty scissor (it would clip the whole frame to nothing), so when the
        // offset is the cause, recover by re-running the combine without it. Observed on Bendy and
        // the Ink Machine (PPSA27616), whose title/menu/gameplay draws leave the VPORT scissor
        // offset-enabled while PA_SC_WINDOW_OFFSET carries a large value; the un-offset scissor is
        // the correct full-target region and the frame renders. #1164 extends the empty test to
        // "empty after clamping to the color0 extent" so in-game draws whose offset shifts the
        // scissor off-target (non-empty in absolute coords) also recover. CONFIDENCE: MED —
        // mechanism verified against the live capture; the un-offset recovery matches hardware.
        if (!combine(/*apply_offset=*/true) && (offset_x != 0 || offset_y != 0))
            combine(/*apply_offset=*/false);
        // PROSPER_SCISSORLOG (#1287 bring-up): print the raw scissor registers whenever the combined
        // rectangle is empty — Blue Prince resolves 699 of ~2,400 Day One draws to [0,0)x[0,0).
        if (right <= left || bottom <= top) {
            static const bool scissor_log = getenv("PROSPER_SCISSORLOG") != nullptr;
            // Atomic like the #1335 recovery log above: extract_render_state runs from
            // DrawRealizationPool workers, so a plain int here was a formal data race (#1345).
            static std::atomic<int> logged{0};
            if (scissor_log && logged.fetch_add(1) < 24) {
                // screen= is the EFFECTIVE pair — the #1335 degenerate-pair recovery above may
                // have substituted the full default; the recovery's own log prints the raw pair.
                fprintf(stderr,
                        "[scissor] EMPTY combined=[%d,%d)-[%d,%d) screen=%08x/%08x window=%08x/%08x "
                        "generic=%08x/%08x vport=%08x/%08x offset=%08x mode=%08x\n",
                        left, top, right, bottom, screen_tl, screen_br, window_tl, window_br,
                        generic_tl, generic_br, viewport_tl, viewport_br, window_offset, mode);
            }
        }
        rs.scissor_left = left; rs.scissor_top = top;
        rs.scissor_right = right; rs.scissor_bottom = bottom;
    }

    return rs;
}

ColorStateTraceSnapshot snapshot_color_state_trace(const RenderState& rs,
                                                   const ResolvedPipelineState& ps) {
    ColorStateTraceSnapshot out;
    out.es_addr = rs.es_addr;
    out.ps_addr = rs.ps_addr;
    out.has_cb_color_control = rs.has_cb_color_control;
    out.has_cb_target_mask = rs.has_cb_target_mask;
    out.has_cb_shader_mask = rs.has_cb_shader_mask;
    out.cb_color_control = rs.cb_color_control;
    out.cb_color_mode = PM4_FIELD(rs.cb_color_control, CB_COLOR_CONTROL, MODE);
    out.cb_target_mask = rs.cb_target_mask;
    out.cb_shader_mask = rs.cb_shader_mask;
    for (uint32_t slot = 0; slot < out.color_targets.size(); ++slot) {
        out.color_targets[slot] = {
            rs.color_targets[slot].base,
            rs.color_targets[slot].width,
            rs.color_targets[slot].height,
            rs.color_targets[slot].format,
            ps.color_targets[slot].format,
            ps.color_targets[slot].write_mask,
        };
    }
    out.has_depth_extent = rs.has_db_depth_size_xy;
    out.raw_depth_size_xy = rs.db_depth_size_xy;
    if (out.has_depth_extent) {
        out.depth_width = PM4_FIELD(rs.db_depth_size_xy, DB_DEPTH_SIZE_XY, X_MAX) + 1u;
        out.depth_height = PM4_FIELD(rs.db_depth_size_xy, DB_DEPTH_SIZE_XY, Y_MAX) + 1u;
    }
    out.depth_read_base = rs.depth_read_base;
    out.depth_write_base = rs.depth_write_base;
    out.stencil_read_base = rs.stencil_read_base;
    out.stencil_write_base = rs.stencil_write_base;
    out.htile_data_base = rs.htile_data_base;
    return out;
}

bool color_state_trace_matches_dimension(const ColorStateTraceSnapshot& snapshot,
                                         uint32_t width, uint32_t height) {
    return std::any_of(snapshot.color_targets.begin(), snapshot.color_targets.end(),
                       [&](const auto& target) {
                           return target.width == width && target.height == height;
                       });
}

ResolvedPipelineState resolve_pipeline_state(const RenderState& rs) {
    ResolvedPipelineState ps;
    ps.topology      = static_cast<uint32_t>(vk_topology(rs.prim_type));
    ps.color0_format = static_cast<uint32_t>(
        vk_color_format(rs.color0_format, rs.color0_number_type, rs.color0_comp_swap));
    ps.color1_format = rs.color1_format ? static_cast<uint32_t>(
        vk_color_format(rs.color1_format, rs.color1_number_type, rs.color1_comp_swap)) : 0u;
    ps.spi_shader_col_format = rs.spi_shader_col_format;
    ps.sx_ps_downconvert = rs.sx_ps_downconvert;
    // Carry the raw color-state triple through resolution unchanged (#1459). These are pure
    // provenance for the masks computed below: nothing here reads them back, so a resolved pipeline
    // keeps recording WHY its color write mask ended up where it did, not merely what it became.
    ps.has_cb_color_control = rs.has_cb_color_control;
    ps.has_cb_target_mask = rs.has_cb_target_mask;
    ps.has_cb_shader_mask = rs.has_cb_shader_mask;
    ps.cb_color_control = rs.cb_color_control;
    ps.cb_target_mask = rs.cb_target_mask;
    ps.cb_shader_mask = rs.cb_shader_mask;

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

    ps.blend_enable = rs.blend_enable;
    vk_blend_factors(rs.color_src_blend, rs.color_dst_blend,
                     ps.src_color_blend_factor, ps.dst_color_blend_factor);
    ps.color_blend_op = vk_blend_op(rs.color_comb_fcn);
    // Alpha blend factors/op (#381): use the separate ALPHA_* fields when SEPARATE_ALPHA_BLEND is set,
    // else mirror the color factors (RDNA2 behavior — Kyty GraphicsRender). The backend reads these
    // directly, so it no longer has to reuse the color factors for the alpha channel (which corrupted
    // stored dst-alpha on any target later read back — RTT composites, alpha-test/text layers).
    if (rs.separate_alpha_blend) {
        vk_blend_factors(rs.alpha_src_blend, rs.alpha_dst_blend,
                         ps.src_alpha_blend_factor, ps.dst_alpha_blend_factor);
        ps.alpha_blend_op         = vk_blend_op(rs.alpha_comb_fcn);
    } else {
        ps.src_alpha_blend_factor = ps.src_color_blend_factor;
        ps.dst_alpha_blend_factor = ps.dst_color_blend_factor;
        ps.alpha_blend_op         = ps.color_blend_op;
    }

    ps.blend1_enable = rs.blend1_enable;
    vk_blend_factors(rs.color1_src_blend, rs.color1_dst_blend,
                     ps.src_color_blend_factor1, ps.dst_color_blend_factor1);
    ps.color_blend_op1 = vk_blend_op(rs.color1_comb_fcn);
    if (rs.separate_alpha_blend1) {
        vk_blend_factors(rs.alpha1_src_blend, rs.alpha1_dst_blend,
                         ps.src_alpha_blend_factor1, ps.dst_alpha_blend_factor1);
        ps.alpha_blend_op1 = vk_blend_op(rs.alpha1_comb_fcn);
    } else {
        ps.src_alpha_blend_factor1 = ps.src_color_blend_factor1;
        ps.dst_alpha_blend_factor1 = ps.dst_color_blend_factor1;
        ps.alpha_blend_op1 = ps.color_blend_op1;
    }

    // CB_TARGET_MASK and CB_SHADER_MASK each hold a 4-bit mask per MRT. Hardware writes only their
    // intersection; RDNA2's R/G/B/A bit order matches VkColorComponentFlags 1:1.
    const uint32_t effective_color_mask = rs.cb_target_mask & rs.cb_shader_mask;
    ps.color_write_mask = effective_color_mask & 0xFu;
    ps.color1_write_mask = (effective_color_mask >> 4) & 0xFu;

    // Resolve the complete MRT array. Slots 0/1 mirror the named compatibility fields; slots 2..7
    // come directly from the generic register decode above.
    for (uint32_t slot = 0; slot < ps.color_targets.size(); ++slot) {
        auto& out = ps.color_targets[slot];
        const ColorTargetState& in = rs.color_targets[slot];
        out.format = in.format ? static_cast<uint32_t>(
            vk_color_format(in.format, in.number_type, in.comp_swap)) : 0u;
        out.has_clear = decode_clear_color(
            in.format, in.number_type, in.comp_swap, in.has_clear, in.clear_word0, out.clear);
        out.blend_enable = in.blend_enable;
        vk_blend_factors(in.color_src_blend, in.color_dst_blend,
                         out.src_color_blend_factor, out.dst_color_blend_factor);
        out.color_blend_op = vk_blend_op(in.color_comb_fcn);
        if (in.separate_alpha_blend) {
            vk_blend_factors(in.alpha_src_blend, in.alpha_dst_blend,
                             out.src_alpha_blend_factor, out.dst_alpha_blend_factor);
            out.alpha_blend_op = vk_blend_op(in.alpha_comb_fcn);
        } else {
            out.src_alpha_blend_factor = out.src_color_blend_factor;
            out.dst_alpha_blend_factor = out.dst_color_blend_factor;
            out.alpha_blend_op = out.color_blend_op;
        }
        out.write_mask = (effective_color_mask >> (slot * 4u)) & 0xFu;
        out.disable_rop3 = in.disable_rop3;
    }
    // Direct/synthetic RenderState callers predate color_targets and still populate the named fields.
    auto& out0 = ps.color_targets[0];
    out0.format = ps.color0_format; out0.has_clear = ps.has_clear_color;
    std::copy(std::begin(ps.clear_color), std::end(ps.clear_color), out0.clear);
    out0.blend_enable = ps.blend_enable;
    out0.src_color_blend_factor = ps.src_color_blend_factor;
    out0.dst_color_blend_factor = ps.dst_color_blend_factor;
    out0.color_blend_op = ps.color_blend_op;
    out0.src_alpha_blend_factor = ps.src_alpha_blend_factor;
    out0.dst_alpha_blend_factor = ps.dst_alpha_blend_factor;
    out0.alpha_blend_op = ps.alpha_blend_op;
    out0.write_mask = ps.color_write_mask; out0.disable_rop3 = rs.disable_rop3;
    auto& out1 = ps.color_targets[1];
    out1.format = ps.color1_format; out1.has_clear = ps.has_clear_color1;
    std::copy(std::begin(ps.clear_color1), std::end(ps.clear_color1), out1.clear);
    out1.blend_enable = ps.blend1_enable;
    out1.src_color_blend_factor = ps.src_color_blend_factor1;
    out1.dst_color_blend_factor = ps.dst_color_blend_factor1;
    out1.color_blend_op = ps.color_blend_op1;
    out1.src_alpha_blend_factor = ps.src_alpha_blend_factor1;
    out1.dst_alpha_blend_factor = ps.dst_alpha_blend_factor1;
    out1.alpha_blend_op = ps.alpha_blend_op1;
    out1.write_mask = ps.color1_write_mask; out1.disable_rop3 = rs.disable_rop3_1;

    // MODE=DISABLE suppresses color-buffer writes, not the whole draw: depth/stencil tests and
    // updates still execute. An absent CB_COLOR_CONTROL retains the legacy normal-draw behavior;
    // otherwise a zero-initialized synthetic RenderState would unexpectedly lose every color draw.
    // DCC_DECOMPRESS is handled by the AGC helper-program path in gpu_execute.hpp. Other hardware
    // metadata modes remain ordinary draws for compatibility, but are now visible instead of silent.
    const uint32_t cb_mode = PM4_FIELD(rs.cb_color_control, CB_COLOR_CONTROL, MODE);
    if (rs.has_cb_color_control) {
        if (cb_mode == P::CB_COLOR_CONTROL_MODE_DISABLE) {
            ps.color_write_mask = 0;
            ps.color1_write_mask = 0;
            for (auto& target : ps.color_targets) target.write_mask = 0;
        } else if (cb_mode == P::CB_COLOR_CONTROL_MODE_RESOLVE) {
            // MSAA resolve: color0 (MSAA source) -> color1 (single-sample dest). The live backend copies
            // the already-rendered color0 surface into color1 instead of running these as ordinary draws.
            ps.cb_resolve = true;
        } else if (cb_mode_is_unmodeled_metadata_operation(cb_mode)) {
            report_unmodeled_cb_color_mode(cb_mode);
        }
    }

    // CB_COLOR_CONTROL.ROP3 is AMD's global raster operation. Vulkan exposes the same 16
    // source/destination Boolean functions as VkLogicOp. COPY stays disabled here: enabling Vulkan
    // COPY would suppress ordinary blending, while disabled COPY preserves the hardware's normal
    // blend-then-copy path. CB_BLENDn_CONTROL.DISABLE_ROP3 can opt an attachment out; Vulkan's logic
    // op is global, so a mixed two-target enable cannot be represented and falls back visibly to COPY.
    const uint32_t rop3 = PM4_FIELD(rs.cb_color_control, CB_COLOR_CONTROL, ROP3);
    uint32_t logic_op = 3;
    if (cb_mode == P::CB_COLOR_CONTROL_MODE_NORMAL && vk_logic_op(rop3, logic_op) && logic_op != 3u) {
        bool any_target_uses_rop3 = false;
        bool every_target_uses_rop3 = true;
        for (uint32_t slot = 0; slot < ps.color_targets.size(); ++slot) {
            const auto& target = ps.color_targets[slot];
            // Preserve the historical MRT0 contract for synthetic/direct callers: a programmed
            // target mask is enough to exercise ROP3 even when no CB format was supplied.
            const bool active = target.write_mask != 0 && (slot == 0 || target.format != 0);
            any_target_uses_rop3 |= active && !target.disable_rop3;
            every_target_uses_rop3 &= !active || !target.disable_rop3;
        }
        if (any_target_uses_rop3 && every_target_uses_rop3) {
            ps.logic_op_enable = true;
            ps.logic_op = logic_op;
        } else if (any_target_uses_rop3) {
            static std::once_flag logged;
            std::call_once(logged, [] {
                fprintf(stderr, "[gpu] resolve_pipeline_state: per-target DISABLE_ROP3 cannot be "
                                "represented by Vulkan -> COPY fallback\n");
            });
        }
    }

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
    // Depth bias (#1349): PA_SU_POLY_OFFSET_* hold float BIT PATTERNS; Vulkan's mapping is the
    // radv inverse (radv programs FRONT_OFFSET = fui(constantFactor), FRONT_SCALE =
    // fui(slopeFactor * 16), CLAMP = fui(clamp)). Blue Prince programs the D32F configuration
    // (DB_FMT_CNTL NEG_NUM_DB_BITS=-23 + FLOAT_FMT) during its shadow passes; without the bias
    // those maps self-shadow into broad acne bands. Vulkan has a single bias for both faces —
    // prefer the FRONT block when FRONT_ENABLE is set, else the BACK block. CONFIDENCE: MED for
    // titles that program asymmetric front/back offsets (none observed).
    const bool bias_front = PM4_FIELD(su, PA_SU_SC_MODE_CNTL, POLY_OFFSET_FRONT_ENABLE) != 0u;
    const bool bias_back  = PM4_FIELD(su, PA_SU_SC_MODE_CNTL, POLY_OFFSET_BACK_ENABLE) != 0u;
    ps.depth_bias_enable = (bias_front || bias_back) ? 1u : 0u;
    if (ps.depth_bias_enable) {
        const auto as_float = [](uint32_t bits) {
            float value; std::memcpy(&value, &bits, sizeof value); return value;
        };
        const uint32_t scale  = bias_front ? rs.pa_su_poly_offset_front_scale
                                           : rs.pa_su_poly_offset_back_scale;
        const uint32_t offset = bias_front ? rs.pa_su_poly_offset_front_offset
                                           : rs.pa_su_poly_offset_back_offset;
        ps.depth_bias_constant = as_float(offset);
        ps.depth_bias_slope    = as_float(scale) / 16.0f;
        ps.depth_bias_clamp    = as_float(rs.pa_su_poly_offset_clamp);
        // The unscaled-constant inverse above is exact only for the float depth configuration:
        // prosper hosts depth exclusively as D32F, so the host driver always programs the float
        // DB_FMT_CNTL, and the guest must agree (NEG_NUM_DB_BITS=-23 + FLOAT_FMT = 0x1E9 — the
        // value Blue Prince programs). A UNORM-style guest config (D16/D24 units; constants
        // conventionally in the hundreds) would silently misscale by orders of magnitude — warn
        // once, fail-visibly, rather than guess a conversion (#1351).
        if ((rs.pa_su_poly_offset_db_fmt_cntl & 0x1FFu) != 0x1E9u) {
            static std::atomic<bool> warned_fmt{false};
            if (!warned_fmt.exchange(true))
                std::fprintf(stderr,
                             "[render_state] depth bias enabled with non-float "
                             "PA_SU_POLY_OFFSET_DB_FMT_CNTL=0x%08x (expected 0x1E9 config); "
                             "constant may be misscaled\n",
                             rs.pa_su_poly_offset_db_fmt_cntl);
        }
        // A non-finite register value would poison the pipeline; treat it as no bias. Huge-but-
        // finite values are equally implausible as float-config bias factors and DO occur as
        // float-decoded stale cx-arena garbage on this title family (#1264/#1335) — bound them
        // out too. 2^24 passes every plausible legitimate value including unorm-convention
        // constants while rejecting essentially all garbage bit patterns (#1351).
        const auto bias_bad = [](float v) {
            return !std::isfinite(v) || std::fabs(v) > 16777216.0f;
        };
        if (bias_bad(ps.depth_bias_constant) || bias_bad(ps.depth_bias_slope) ||
            bias_bad(ps.depth_bias_clamp)) {
            static std::atomic<bool> warned_mag{false};
            if (!warned_mag.exchange(true))
                std::fprintf(stderr,
                             "[render_state] implausible depth-bias values "
                             "(constant=%g slope=%g clamp=%g) — disabling bias for this draw\n",
                             ps.depth_bias_constant, ps.depth_bias_slope, ps.depth_bias_clamp);
            ps.depth_bias_enable = 0u;
            ps.depth_bias_constant = ps.depth_bias_slope = ps.depth_bias_clamp = 0.0f;
        }
    }
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
