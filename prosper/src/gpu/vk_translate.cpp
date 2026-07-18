// vk_translate.cpp — see vk_translate.hpp.
#include "vk_translate.hpp"
#include <cstdio>
#include <mutex>
#include <set>

namespace prosper::gpu {

VkTopology vk_topology(uint32_t prim_type) {
    // RDNA2 VGT_DI_PRIMITIVE_TYPE (kPrimitiveType*): 1=point, 2=line list, 3=line strip,
    // 4=triangle list, 5=triangle fan, 6=triangle strip. PS5 AGC also emits 7 for RectList:
    // Blasphemous 2 uses it for transparent fullscreen clears. The executor expands the observed
    // three-vertex procedural form to four vertices; treating 7 as the standard AMD reserved value
    // leaves stale opaque alpha over the scene. RectList(17)/QuadList(19) are the
    // screen-space primitives drivers emit for full-screen passes / fast clears / resolves;
    // Kyty (GraphicsRender.cpp) approximates them as TriangleStrip / TriangleFan — far closer
    // than the old PointList fallback, which rasterized a full-screen effect as 3 stray points.
    switch (prim_type) {
        case 1:  return VkTopology::PointList;
        case 2:  return VkTopology::LineList;
        case 3:  return VkTopology::LineStrip;
        case 4:  return VkTopology::TriangleList;
        case 5:  return VkTopology::TriangleFan;
        case 6:  return VkTopology::TriangleStrip;
        case 7:  return VkTopology::TriangleStrip;  // PS5 RectList (executor supplies the fourth corner)
        case 17: return VkTopology::TriangleStrip;  // kPrimitiveTypeRectList (3-vertex screen rect ~ strip)
        case 19: return VkTopology::TriangleFan;    // kPrimitiveTypeQuadList (~ fan)
        default: break;
    }
    // Everything else (Patch=9, adjacency LineListAdj=10..TriStripAdj=13, and any unknown value)
    // falls back to PointList. NOTE: adjacency/patch topologies map 1:1 to Vulkan values but
    // require a geometry- or tessellation-shader pipeline path prosper does not build yet — emitting
    // them here would make vkCreateGraphicsPipelines fail (returns {} = nothing) rather than render.
    // PointList at least produces visible output. Log the specific type ONCE so a silent
    // points-fallback (which used to hide RectList) is diagnosable — matching vk_color_format.
    static std::set<uint32_t> logged;
    static std::mutex logged_mu;
    {
        std::lock_guard<std::mutex> lk(logged_mu);
        if (logged.insert(prim_type).second)
            fprintf(stderr, "[gpu] vk_topology: unmapped VGT_PRIMITIVE_TYPE=%u -> PointList "
                            "(adjacency/patch need a GS/tessellation path)\n", prim_type);
    }
    return VkTopology::PointList;
}

VkFormat vk_color_format(uint32_t format, uint32_t number_type, uint32_t comp_swap) {
    // CB_COLOR0_INFO enums (GCN/RDNA CB format table; Kyty exercises the 0xA/0x1 rows):
    //   FORMAT: 1=COLOR_8, 2=COLOR_16, 3=COLOR_8_8, 4=COLOR_32, 5=COLOR_16_16, 7=COLOR_11_11_10,
    //           9=COLOR_2_10_10_10, 0xA=COLOR_8_8_8_8, 0xB=COLOR_32_32, 0xC=COLOR_16_16_16_16,
    //           0xE=COLOR_32_32_32_32, 0x10=COLOR_5_6_5
    //   NUMBER_TYPE: 0=UNORM, 1=SNORM, 4=UINT, 5=SINT, 6=SRGB, 7=FLOAT
    //   COMP_SWAP: 0=STD (RGBA order), 1=ALT (BGRA); 2/3 (REV forms) unmapped.
    // CONFIDENCE: HIGH for 0xA (live-verified); MED for the ported rows (standard table, no live
    // capture yet — an incorrect row renders wrong but cannot crash: the backend sees a real format).
    const uint32_t nt = number_type, cs = comp_swap;
    const bool std_swap = (cs == 0u), alt_swap = (cs == 1u);
    switch (format) {
        case 0x1u:   // COLOR_8
            if (!std_swap) break;
            switch (nt) { case 0: return VkFormat::R8_UNORM; case 1: return VkFormat::R8_SNORM;
                          case 4: return VkFormat::R8_UINT;  case 5: return VkFormat::R8_SINT; }
            break;
        case 0x2u:   // COLOR_16
            if (!std_swap) break;
            switch (nt) { case 0: return VkFormat::R16_UNORM; case 1: return VkFormat::R16_SNORM;
                          case 4: return VkFormat::R16_UINT;  case 5: return VkFormat::R16_SINT;
                          case 7: return VkFormat::R16_SFLOAT; }
            break;
        case 0x3u:   // COLOR_8_8
            if (!std_swap) break;
            switch (nt) { case 0: return VkFormat::R8G8_UNORM; case 1: return VkFormat::R8G8_SNORM;
                          case 4: return VkFormat::R8G8_UINT;  case 5: return VkFormat::R8G8_SINT; }
            break;
        case 0x4u:   // COLOR_32
            if (!std_swap) break;
            switch (nt) { case 4: return VkFormat::R32_UINT; case 5: return VkFormat::R32_SINT;
                          case 7: return VkFormat::R32_SFLOAT; }
            break;
        case 0x5u:   // COLOR_16_16
            if (!std_swap) break;
            switch (nt) { case 0: return VkFormat::R16G16_UNORM; case 1: return VkFormat::R16G16_SNORM;
                          case 4: return VkFormat::R16G16_UINT;  case 5: return VkFormat::R16G16_SINT;
                          case 7: return VkFormat::R16G16_SFLOAT; }
            break;
        case 0x6u:   // COLOR_10_11_11 — the ALIAS of COLOR_11_11_10 (0x7). shadPS4 canonicalizes 0x7->0x6
        case 0x7u:   // COLOR_11_11_10 (float-only packed HDR target). Both codes resolve to R11G11B10F;
                     // only 0x7 was handled, so a guest HDR/bloom target programming FORMAT=6 (the
                     // canonical code) fell through to Undefined and its whole pass was dropped (#465).
            if (std_swap && nt == 7u) return VkFormat::B10G11R11_UFLOAT_PACK32;
            break;
        case 0x8u:   // COLOR_10_10_10_2 — the ALIAS of COLOR_2_10_10_10 (0x9). shadPS4 canonicalizes 0x8->0x9.
        case 0x9u:   // COLOR_2_10_10_10
            if (nt == 0u) return std_swap ? VkFormat::A2B10G10R10_UNORM_PACK32
                        : alt_swap        ? VkFormat::A2R10G10B10_UNORM_PACK32 : VkFormat::Undefined;
            if (nt == 4u) return std_swap ? VkFormat::A2B10G10R10_UINT_PACK32
                        : alt_swap        ? VkFormat::A2R10G10B10_UINT_PACK32  : VkFormat::Undefined;
            break;
        case 0xAu:   // COLOR_8_8_8_8 (the live-verified row: RGBA/BGRA x UNORM/SRGB, + int variants)
            if (std_swap) switch (nt) {
                case 0: return VkFormat::R8G8B8A8_UNORM; case 1: return VkFormat::R8G8B8A8_SNORM;
                case 4: return VkFormat::R8G8B8A8_UINT;  case 5: return VkFormat::R8G8B8A8_SINT;
                case 6: return VkFormat::R8G8B8A8_SRGB;
            }
            if (alt_swap) switch (nt) {
                case 0: return VkFormat::B8G8R8A8_UNORM; case 1: return VkFormat::B8G8R8A8_SNORM;
                case 4: return VkFormat::B8G8R8A8_UINT;  case 5: return VkFormat::B8G8R8A8_SINT;
                case 6: return VkFormat::B8G8R8A8_SRGB;
            }
            break;
        case 0xBu:   // COLOR_32_32
            if (!std_swap) break;
            switch (nt) { case 4: return VkFormat::R32G32_UINT; case 5: return VkFormat::R32G32_SINT;
                          case 7: return VkFormat::R32G32_SFLOAT; }
            break;
        case 0xCu:   // COLOR_16_16_16_16
            if (!std_swap) break;
            switch (nt) { case 0: return VkFormat::R16G16B16A16_UNORM; case 1: return VkFormat::R16G16B16A16_SNORM;
                          case 4: return VkFormat::R16G16B16A16_UINT;  case 5: return VkFormat::R16G16B16A16_SINT;
                          case 7: return VkFormat::R16G16B16A16_SFLOAT; }
            break;
        case 0xEu:   // COLOR_32_32_32_32
            if (!std_swap) break;
            switch (nt) { case 4: return VkFormat::R32G32B32A32_UINT; case 5: return VkFormat::R32G32B32A32_SINT;
                          case 7: return VkFormat::R32G32B32A32_SFLOAT; }
            break;
        case 0x10u:  // COLOR_5_6_5
            // SWAP_STD places R at the LOW end (the convention proven by the live-verified 0xA row —
            // STD -> R8G8B8A8, R at byte 0 — and the 0x9 row — STD -> A2B10G10R10, R in bits [9:0]).
            // For a 565 packed word R-at-low is Vulkan B5G6R5_UNORM_PACK16 (R in bits [4:0]); the STD
            // and ALT arms were previously reversed, swapping red/blue on any 565 surface (#913).
            if (nt == 0u) return std_swap ? VkFormat::B5G6R5_UNORM_PACK16
                        : alt_swap        ? VkFormat::R5G6B5_UNORM_PACK16 : VkFormat::Undefined;
            break;
        default: break;
    }
    // Unmapped triple: log ONCE per triple so a broken pipeline is diagnosable (it used to resolve
    // to Undefined silently), without flooding a per-draw path.
    static std::set<uint64_t> logged;
    static std::mutex logged_mu;
    uint64_t key = ((uint64_t)format << 32) | (nt << 8) | cs;
    {
        std::lock_guard<std::mutex> lk(logged_mu);
        if (logged.insert(key).second)
            fprintf(stderr, "[gpu] vk_color_format: unmapped CB surface format=0x%x number_type=%u "
                            "comp_swap=%u -> Undefined\n", format, nt, cs);
    }
    return VkFormat::Undefined;
}

uint32_t vk_blend_factor(uint32_t f) {
    // RDNA2 blend factor -> VkBlendFactor value (Kyty GraphicsRender.cpp).
    // CONFIDENCE: HIGH — cross-checked against SharpEmu @ 85cc2b9
    // (src/SharpEmu.Libs/VideoOut/VulkanVideoPresenter.cs ToVkBlendFactor + AgcExports.cs
    // CB_BLEND0_CONTROL 0x1E0 decode): same register/bit-fields (COLOR_SRCBLEND[4:0],
    // COLOR_DESTBLEND[12:8], COLOR_COMB_FCN[7:5], ENABLE[30]) and same enum (4->SrcAlpha,
    // 5->OneMinusSrcAlpha). The two emulators decode blend identically.
    switch (f) {
        case 0x00: return 0;   // Zero              -> ZERO
        case 0x01: return 1;   // One               -> ONE
        case 0x02: return 2;   // SrcColor          -> SRC_COLOR
        case 0x03: return 3;   // OneMinusSrcColor  -> ONE_MINUS_SRC_COLOR
        case 0x04: return 6;   // SrcAlpha          -> SRC_ALPHA
        case 0x05: return 7;   // OneMinusSrcAlpha  -> ONE_MINUS_SRC_ALPHA
        case 0x06: return 8;   // DstAlpha          -> DST_ALPHA
        case 0x07: return 9;   // OneMinusDstAlpha  -> ONE_MINUS_DST_ALPHA
        case 0x08: return 4;   // DstColor          -> DST_COLOR
        case 0x09: return 5;   // OneMinusDstColor  -> ONE_MINUS_DST_COLOR
        case 0x0a: return 14;  // SrcAlphaSaturate  -> SRC_ALPHA_SATURATE
        case 0x0d: return 10;  // ConstantColor     -> CONSTANT_COLOR
        case 0x0e: return 11;  // OneMinusConstColor-> ONE_MINUS_CONSTANT_COLOR
        case 0x0f: return 15;  // Src1Color         -> SRC1_COLOR
        case 0x10: return 16;  // InvSrc1Color      -> ONE_MINUS_SRC1_COLOR
        case 0x11: return 17;  // Src1Alpha         -> SRC1_ALPHA
        case 0x12: return 18;  // InvSrc1Alpha      -> ONE_MINUS_SRC1_ALPHA
        case 0x13: return 12;  // ConstantAlpha     -> CONSTANT_ALPHA
        case 0x14: return 13;  // OneMinusConstAlpha-> ONE_MINUS_CONSTANT_ALPHA
        default:   return 0;   // ZERO
    }
}

uint32_t vk_blend_op(uint32_t comb_fcn) {
    switch (comb_fcn) {
        case 0: return 0;   // Add             -> VK_BLEND_OP_ADD
        case 1: return 1;   // Subtract        -> VK_BLEND_OP_SUBTRACT
        case 2: return 3;   // Min             -> VK_BLEND_OP_MIN
        case 3: return 4;   // Max             -> VK_BLEND_OP_MAX
        case 4: return 2;   // ReverseSubtract -> VK_BLEND_OP_REVERSE_SUBTRACT
        default: return 0;  // ADD
    }
}

} // namespace prosper::gpu
