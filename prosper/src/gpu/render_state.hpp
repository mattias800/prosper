// render_state.hpp — interpret a GpuState's register files into a semantic render-state.
//
// Third CommandProcessor stage (after decode + apply): read the well-defined RDNA2 hardware registers
// out of the folded register files and expose them as a small, typed RenderState that a Vulkan
// backend (M4) can turn into a pipeline + draw. Register offsets/masks come from pm4_registers.hpp
// (vendored from Kyty). Address fields use the confirmed RDNA2 convention `addr = (LO<<8) |
// ((HI&0xff)<<40)` (verified against Kyty GraphicsRun.cpp). Fields whose in-word bit layout is not
// yet confirmed are exposed as faithful RAW register values rather than guessed-apart, so nothing
// here is fabricated.
#pragma once
#include "command_processor.hpp"
#include <array>
#include <cstdint>

namespace prosper::gpu {

struct RenderState {
    // Shader program GPU addresses per stage (byte address; 0 if that stage's PGM regs were unset).
    uint64_t ps_addr = 0;   // pixel  (SPI_SHADER_PGM_{LO,HI}_PS)
    uint64_t gs_addr = 0;   // geometry (…_GS)
    uint64_t es_addr = 0;   // export/vertex (…_ES)
    uint64_t hs_addr = 0;   // hull/tess (…_HS)

    // Pixel-input wiring (SPI_PS_INPUT_CNTL_0..31). Each programmed word maps a PS interpolant
    // number to an ES/VS PARAM export slot, or selects one of the hardware constant defaults when
    // OFFSET=0x20. Vulkan has no equivalent fixed-function remap/default stage, so the vertex
    // recompiler materializes this contract in its output interface.
    std::array<uint32_t, 32> ps_input_cntl{};
    uint32_t ps_input_cntl_valid_mask = 0;

    // Hardware-provided PS VGPR inputs (barycentrics, floating-point position, face/sample data).
    // INPUT_ADDR controls slot reservation even for terms disabled in INPUT_ENA, so both raw words
    // are retained for the fragment recompiler.
    uint32_t ps_input_ena = 0;
    uint32_t ps_input_addr = 0;

    // Color MRT 0. format/number_type/comp_swap together select the VkFormat (see vk_translate).
    uint64_t color0_base        = 0;   // byte address (CB_COLOR0_BASE + BASE_EXT)
    uint32_t color0_format      = 0;   // CB_COLOR0_INFO.FORMAT       (surface format)
    uint32_t color0_number_type = 0;   // CB_COLOR0_INFO.NUMBER_TYPE  (UNORM/SRGB/…)
    uint32_t color0_comp_swap   = 0;   // CB_COLOR0_INFO.COMP_SWAP    (channel order: RGBA/BGRA/…)

    // Gen5 target extent from CB_COLOR0_ATTRIB2. Fields store dimension-1. This is independent
    // of VideoOut and of the viewport: Unity's grading LUT is a 1024x32 render target (#526).
    bool     color0_has_extent = false;
    uint32_t color0_width = 0, color0_height = 0;

    // CB fast-clear for MRT 0 (CB_COLOR0_CLEAR_WORD0/1 = 0x323/0x324). `has_clear` is true when the
    // game PROGRAMMED the clear words (register present); the words hold the clear value in the
    // target's pixel format, decoded to an RGBA float in resolve_pipeline_state (#309).
    bool     color0_has_clear   = false;
    uint32_t color0_clear_word0 = 0, color0_clear_word1 = 0;

    // Color MRT 1. The first two targets are enough for UE4's DOLL scene dependency; later MRTs stay
    // fail-visible until their attachment semantics are implemented.
    uint64_t color1_base        = 0;
    uint32_t color1_format      = 0;
    uint32_t color1_number_type = 0;
    uint32_t color1_comp_swap   = 0;
    bool     color1_has_extent  = false;
    uint32_t color1_width = 0, color1_height = 0;
    bool     color1_has_clear   = false;
    uint32_t color1_clear_word0 = 0, color1_clear_word1 = 0;

    // Primitive topology (VGT_PRIMITIVE_TYPE.PRIM_TYPE).
    uint32_t prim_type = 0;

    // Depth/stencil test state, decoded from DB_DEPTH_CONTROL (field shifts/masks from pm4_registers.hpp,
    // matching Kyty hw_ctx_set_depth_control).
    bool     z_enable       = false;
    bool     z_write_enable = false;
    bool     stencil_enable = false;
    uint32_t zfunc          = 0;      // compare op, 0..7 (RDNA2 order == VkCompareOp order)

    // Depth/stencil CLEAR values (DB_DEPTH_CLEAR = float depth, DB_STENCIL_CLEAR = 8-bit stencil).
    // has_depth_clear == the game PROGRAMMED DB_DEPTH_CLEAR (register present); otherwise resolve picks
    // a compare-op-appropriate default (1.0 for LESS, 0.0 for GREATER) instead of a fixed 0.5 (#371).
    bool     has_depth_clear     = false;
    bool     has_stencil_clear   = false;
    float    depth_clear_value   = 1.0f;
    uint32_t stencil_clear_value = 0;

    // Guest depth/stencil attachment identity and explicit clear intent. These are distinct from
    // enabling the tests: DB_RENDER_CONTROL requests a hardware clear, while the *_BASE registers
    // identify contents that must survive color-target changes and later submits (#518).
    uint32_t db_render_control = 0;
    bool depth_clear_enable = false, stencil_clear_enable = false;
    uint64_t depth_read_base = 0, depth_write_base = 0;
    uint64_t stencil_read_base = 0, stencil_write_base = 0;
    // Raw guest depth-surface programming retained for cache-lifetime diagnostics (#611). The four
    // plane bases alone do not distinguish mip/slice views, HTILE metadata, or a re-described backing.
    uint32_t db_depth_view = 0, db_render_override = 0, db_render_override2 = 0;
    uint64_t htile_data_base = 0;
    uint32_t db_depth_size_xy = 0, db_dfsm_control = 0, db_depth_info = 0;
    uint32_t db_z_info = 0, db_stencil_info = 0;
    uint32_t db_depth_size = 0, db_depth_slice = 0;
    uint32_t db_htile_surface = 0, db_rmi_l2_cache_control = 0;
    uint32_t db_shader_control = 0;
    bool stencil_test_val_export_enable = false, stencil_op_val_export_enable = false;

    // Color blend state for MRT 0, decoded from CB_BLEND0_CONTROL (RDNA2 factor/op enum values;
    // map to Vulkan with vk_blend_factor / vk_blend_op).
    bool     blend_enable    = false;
    uint32_t color_src_blend = 0;     // COLOR_SRCBLEND
    uint32_t color_dst_blend = 0;     // COLOR_DESTBLEND
    uint32_t color_comb_fcn  = 0;     // COLOR_COMB_FCN (blend op)
    // Separate alpha-channel blend (CB_BLEND0_CONTROL.SEPARATE_ALPHA_BLEND @29): when set, the alpha
    // channel blends with its OWN factors/op; when clear, alpha mirrors the color factors (#381).
    bool     separate_alpha_blend = false;
    uint32_t alpha_src_blend  = 0;    // ALPHA_SRCBLEND  (@16)
    uint32_t alpha_dst_blend  = 0;    // ALPHA_DESTBLEND (@24)
    uint32_t alpha_comb_fcn   = 0;    // ALPHA_COMB_FCN  (@21)

    // MRT1 blend state has the same register layout as CB_BLEND0_CONTROL.
    bool     blend1_enable = false;
    uint32_t color1_src_blend = 0, color1_dst_blend = 0, color1_comb_fcn = 0;
    bool     separate_alpha_blend1 = false;
    uint32_t alpha1_src_blend = 0, alpha1_dst_blend = 0, alpha1_comb_fcn = 0;

    // Raw state registers — remaining bit layouts decoded by the Vulkan backend later (kept faithful).
    uint32_t db_depth_control  = 0;   // DB_DEPTH_CONTROL
    uint32_t db_stencil_control   = 0;   // DB_STENCIL_CONTROL   (front/back stencil-fail / z-pass / z-fail ops)
    uint32_t db_stencilrefmask    = 0;   // DB_STENCILREFMASK    (front ref / compare-mask / write-mask)
    uint32_t db_stencilrefmask_bf = 0;   // DB_STENCILREFMASK_BF (back-face ref / compare-mask / write-mask)
    uint32_t cb_color_control  = 0;   // CB_COLOR_CONTROL
    uint32_t cb_blend0_control = 0;   // CB_BLEND0_CONTROL
    uint32_t cb_target_mask    = 0;   // CB_TARGET_MASK (per-MRT write mask)
    // Synthetic/direct RenderState users historically supplied only CB_TARGET_MASK. Match the
    // hardware/command-stream absent-register fallback so adding CB_SHADER_MASK cannot silently
    // turn those otherwise-valid color writes off.
    uint32_t cb_shader_mask    = 0xFFFFFFFFu; // CB_SHADER_MASK (components exported by the pixel shader)
    // Rasterizer cull/front-face/polygon mode (PA_SU_SC_MODE_CNTL). An ABSENT register reads 0, which
    // decodes to CULL_NONE + CCW-front + FILL — exactly the prior hardcoded default, so nothing changes
    // for a guest that never programs it. Decoded in resolve to Vk enums (#456).
    uint32_t pa_su_sc_mode_cntl = 0;

    // Viewport 0 transform (PA_CL_VPORT_{X,Y,Z}{SCALE,OFFSET} — IEEE-754 floats stored in the context
    // registers; 0 when the guest never set them). Hardware applies screen = offset + scale * ndc, with
    // screen-space Y increasing downward — a NEGATIVE yscale is how GNM-style +Y-up NDC reaches the
    // top-left-origin render target.
    float vport_xscale = 0, vport_xoffset = 0;
    float vport_yscale = 0, vport_yoffset = 0;
    float vport_zscale = 0, vport_zoffset = 0;

    // Effective guest scissor after intersecting screen, window, generic, and (when enabled)
    // viewport scissors. Coordinates follow the hardware convention: left/top inclusive and
    // right/bottom exclusive. `has_scissor` is false only when none of the relevant registers was
    // programmed, preserving the backend's full-target default for synthetic/legacy state.
    bool has_scissor = false;
    int32_t scissor_left = 0, scissor_top = 0, scissor_right = 0, scissor_bottom = 0;
};

// Extract the render-state from a folded GpuState (pure; reads register files only).
RenderState extract_render_state(const GpuState& st);

// A pipeline's fixed-function state with every RDNA2 enum already translated to its Vulkan value
// (via vk_translate). All fields are plain integers equal to the corresponding Vk* enumerators, so
// the Vulkan backend can drop them straight into a VkGraphicsPipelineCreateInfo with no further
// mapping — and this resolution is unit-testable with no Vulkan dependency (it runs on any host).
struct ResolvedPipelineState {
    uint32_t topology         = 0;   // == VkPrimitiveTopology
    uint32_t color0_format    = 0;   // == VkFormat (0 = VK_FORMAT_UNDEFINED)

    // Color-target clear value, decoded from CB_COLOR0_CLEAR_WORD0/1 per the surface format (#309).
    // has_clear_color == the game programmed a fast-clear that we decoded; clear_color is RGBA in
    // Vulkan order (float32[0]=R). When has_clear_color is false the backend clears to opaque black —
    // NOT the old diagnostic blue, which now lives behind the PROSPER_CLEAR_DEBUG env in render_runner.
    bool     has_clear_color  = false;
    float    clear_color[4]   = {0.0f, 0.0f, 0.0f, 1.0f};
    bool     depth_test_enable  = false;
    bool     depth_write_enable = false;
    uint32_t depth_compare_op  = 0;  // == VkCompareOp
    // Depth/stencil CLEAR value for VK_ATTACHMENT_LOAD_OP_CLEAR. Resolve sets depth_clear_value from the
    // guest's DB_DEPTH_CLEAR when programmed, else a compare-op-appropriate default (1.0 for LESS/LEQUAL,
    // 0.0 for GREATER/GEQUAL) — never a fixed 0.5, which wrongly rejects far-half geometry under a
    // standard depth test (#371). stencil_clear_value from DB_STENCIL_CLEAR.
    float    depth_clear_value   = 1.0f;
    uint32_t stencil_clear_value = 0;
    bool     has_depth_clear = false, has_stencil_clear = false;
    uint32_t db_render_control = 0;
    bool     depth_clear_enable = false, stencil_clear_enable = false;
    uint64_t depth_read_base = 0, depth_write_base = 0;
    uint64_t stencil_read_base = 0, stencil_write_base = 0;
    uint32_t db_depth_view = 0, db_render_override = 0, db_render_override2 = 0;
    uint64_t htile_data_base = 0;
    uint32_t db_depth_size_xy = 0, db_dfsm_control = 0, db_depth_info = 0;
    uint32_t db_z_info = 0, db_stencil_info = 0;
    uint32_t db_depth_size = 0, db_depth_slice = 0;
    uint32_t db_htile_surface = 0, db_rmi_l2_cache_control = 0;
    uint32_t db_shader_control = 0;
    bool stencil_test_val_export_enable = false, stencil_op_val_export_enable = false;

    // Stencil test state, resolved from DB_DEPTH_CONTROL (enable + STENCILFUNC) + DB_STENCIL_CONTROL
    // (ops) + DB_STENCILREFMASK[_BF] (ref / compare-mask / write-mask). Index [0]=front, [1]=back.
    // A UI mask (e.g. The Messenger's title shimmer) writes the stencil in one draw and tests it in the
    // next — with these left off the mask draw is unmasked (#264). Values are the Vk* enumerators.
    bool     stencil_enable          = false;
    uint32_t stencil_compare_op[2]   = {7, 7};       // == VkCompareOp (7 = ALWAYS)
    uint32_t stencil_fail_op[2]      = {0, 0};        // == VkStencilOp (0 = KEEP)
    uint32_t stencil_pass_op[2]      = {0, 0};        // depth-pass op
    uint32_t stencil_depth_fail_op[2]= {0, 0};
    uint32_t stencil_ref[2]          = {0, 0};        // STENCILTESTVAL — the COMPARE reference
    uint32_t stencil_op_val[2]       = {0, 0};        // STENCILOPVAL — the value written by a REPLACE op
    uint32_t stencil_compare_mask[2] = {0xFF, 0xFF};
    uint32_t stencil_write_mask[2]   = {0xFF, 0xFF};
    uint32_t raw_stencil_op[2][3]    = {{0, 0, 0}, {0, 0, 0}}; // fail, z-pass, z-fail

    bool     blend_enable        = false;
    uint32_t src_color_blend_factor = 0;   // == VkBlendFactor
    uint32_t dst_color_blend_factor = 0;   // == VkBlendFactor
    uint32_t color_blend_op         = 0;   // == VkBlendOp
    // Alpha-channel blend factors/op (== VkBlendFactor/VkBlendOp). Resolve sets these from the separate
    // ALPHA_* fields when SEPARATE_ALPHA_BLEND is set, else mirrors the color factors — so the backend
    // can always read ps->*_alpha_* directly instead of guessing (#381).
    uint32_t src_alpha_blend_factor = 0;   // == VkBlendFactor
    uint32_t dst_alpha_blend_factor = 0;   // == VkBlendFactor
    uint32_t alpha_blend_op         = 0;   // == VkBlendOp
    uint32_t color_write_mask       = 0xF; // == VkColorComponentFlags (RGBA); MRT0 nibble of CB_TARGET_MASK

    // Guest viewport as a Vulkan viewport rect. `has_viewport` is false when the guest never programmed
    // PA_CL_VPORT (then the backend keeps its full-target default). A guest with a negative yscale
    // resolves to a NEGATIVE viewport_h — Vulkan's core-1.1 (maintenance1) flipped viewport — which
    // reproduces the hardware's Y orientation exactly.
    bool  has_viewport = false;
    float viewport_x = 0, viewport_y = 0, viewport_w = 0, viewport_h = 0;
    float min_depth = 0.0f, max_depth = 1.0f;

    // Rasterizer state resolved from PA_SU_SC_MODE_CNTL (#456). Values == the Vulkan enumerators, so the
    // backend drops them straight in. Defaults reproduce the prior hardcode (no cull, CCW front, fill),
    // so a draw that never programs the register — and every test building a ResolvedPipelineState
    // directly — is byte-identical. cull_mode == VkCullModeFlags (0=NONE,1=FRONT,2=BACK,3=FRONT_AND_BACK);
    // front_face == VkFrontFace (0=CCW,1=CW); polygon_mode == VkPolygonMode (0=FILL,1=LINE,2=POINT).
    uint32_t cull_mode    = 0;   // VK_CULL_MODE_NONE
    uint32_t front_face   = 0;   // VK_FRONT_FACE_COUNTER_CLOCKWISE / guest FACE=0
    uint32_t polygon_mode = 0;   // VK_POLYGON_MODE_FILL

    // Optional second color attachment. These fields are appended so capture versions through v9
    // retain their exact serialized prefix and materialize with MRT1 disabled.
    uint32_t color1_format = 0;  // == VkFormat (0 = VK_FORMAT_UNDEFINED)
    bool     has_clear_color1 = false;
    float    clear_color1[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    bool     blend1_enable = false;
    uint32_t src_color_blend_factor1 = 0, dst_color_blend_factor1 = 0, color_blend_op1 = 0;
    uint32_t src_alpha_blend_factor1 = 0, dst_alpha_blend_factor1 = 0, alpha_blend_op1 = 0;
    uint32_t color1_write_mask = 0; // MRT1 nibble of CB_TARGET_MASK

    // Effective guest scissor in framebuffer coordinates. Appended so legacy capture prefixes and
    // aggregate initializers retain their existing layout/default behavior.
    bool has_scissor = false;
    int32_t scissor_left = 0, scissor_top = 0, scissor_right = 0, scissor_bottom = 0;
};

// A color-disabled draw must still execute when it can change the depth/stencil attachment consumed
// by later draws. KEEP-only stencil tests without a depth write have no attachment side effect.
inline bool has_depth_stencil_side_effect(const ResolvedPipelineState& ps) {
    if (ps.depth_write_enable || ps.depth_clear_enable || ps.stencil_clear_enable) return true;
    if (!ps.stencil_enable) return false;
    for (int fb = 0; fb < 2; ++fb)
        if (ps.stencil_write_mask[fb] != 0 &&
            (ps.stencil_fail_op[fb] != 0 || ps.stencil_pass_op[fb] != 0 || ps.stencil_depth_fail_op[fb] != 0))
            return true;
    return false;
}

// Translate a RenderState's RDNA2 register semantics into Vulkan-ready pipeline state (pure).
ResolvedPipelineState resolve_pipeline_state(const RenderState& rs);

// Apply the renderer's reduced-resolution scale to both viewport and scissor state. Scissor outer
// bounds round outward so resolution scaling never discards a guest-covered sample.
void scale_resolved_render_area(ResolvedPipelineState& ps, float scale_x, float scale_y);

} // namespace prosper::gpu
