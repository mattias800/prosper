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
#include <cstdint>

namespace prosper::gpu {

struct RenderState {
    // Shader program GPU addresses per stage (byte address; 0 if that stage's PGM regs were unset).
    uint64_t ps_addr = 0;   // pixel  (SPI_SHADER_PGM_{LO,HI}_PS)
    uint64_t gs_addr = 0;   // geometry (…_GS)
    uint64_t es_addr = 0;   // export/vertex (…_ES)
    uint64_t hs_addr = 0;   // hull/tess (…_HS)

    // Color MRT 0. format/number_type/comp_swap together select the VkFormat (see vk_translate).
    uint64_t color0_base        = 0;   // byte address (CB_COLOR0_BASE + BASE_EXT)
    uint32_t color0_format      = 0;   // CB_COLOR0_INFO.FORMAT       (surface format)
    uint32_t color0_number_type = 0;   // CB_COLOR0_INFO.NUMBER_TYPE  (UNORM/SRGB/…)
    uint32_t color0_comp_swap   = 0;   // CB_COLOR0_INFO.COMP_SWAP    (channel order: RGBA/BGRA/…)

    // Primitive topology (VGT_PRIMITIVE_TYPE.PRIM_TYPE).
    uint32_t prim_type = 0;

    // Depth/stencil test state, decoded from DB_DEPTH_CONTROL (field shifts/masks from pm4_registers.hpp,
    // matching Kyty hw_ctx_set_depth_control).
    bool     z_enable       = false;
    bool     z_write_enable = false;
    bool     stencil_enable = false;
    uint32_t zfunc          = 0;      // compare op, 0..7 (RDNA2 order == VkCompareOp order)

    // Color blend state for MRT 0, decoded from CB_BLEND0_CONTROL (RDNA2 factor/op enum values;
    // map to Vulkan with vk_blend_factor / vk_blend_op).
    bool     blend_enable    = false;
    uint32_t color_src_blend = 0;     // COLOR_SRCBLEND
    uint32_t color_dst_blend = 0;     // COLOR_DESTBLEND
    uint32_t color_comb_fcn  = 0;     // COLOR_COMB_FCN (blend op)

    // Raw state registers — remaining bit layouts decoded by the Vulkan backend later (kept faithful).
    uint32_t db_depth_control  = 0;   // DB_DEPTH_CONTROL
    uint32_t cb_color_control  = 0;   // CB_COLOR_CONTROL
    uint32_t cb_blend0_control = 0;   // CB_BLEND0_CONTROL
    uint32_t cb_target_mask    = 0;   // CB_TARGET_MASK (per-MRT write mask)
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
    bool     depth_test_enable  = false;
    bool     depth_write_enable = false;
    uint32_t depth_compare_op  = 0;  // == VkCompareOp
    bool     blend_enable        = false;
    uint32_t src_color_blend_factor = 0;   // == VkBlendFactor
    uint32_t dst_color_blend_factor = 0;   // == VkBlendFactor
    uint32_t color_blend_op         = 0;   // == VkBlendOp
    uint32_t color_write_mask       = 0xF; // == VkColorComponentFlags (RGBA); MRT0 nibble of CB_TARGET_MASK
};

// Translate a RenderState's RDNA2 register semantics into Vulkan-ready pipeline state (pure).
ResolvedPipelineState resolve_pipeline_state(const RenderState& rs);

} // namespace prosper::gpu
