// vk_translate.cpp — see vk_translate.hpp.
#include "vk_translate.hpp"

namespace prosper::gpu {

VkTopology vk_topology(uint32_t prim_type) {
    // RDNA2 VGT_DI_PRIMITIVE_TYPE (kPrimitiveType*): 1=point, 2=line list, 3=line strip,
    // 4=triangle list, 5=triangle fan, 6=triangle strip.
    switch (prim_type) {
        case 1:  return VkTopology::PointList;
        case 2:  return VkTopology::LineList;
        case 3:  return VkTopology::LineStrip;
        case 4:  return VkTopology::TriangleList;
        case 5:  return VkTopology::TriangleFan;
        case 6:  return VkTopology::TriangleStrip;
        default: return VkTopology::PointList;
    }
}

VkFormat vk_color_format(uint32_t format, uint32_t number_type, uint32_t comp_swap) {
    // format 0xA = COLOR_8_8_8_8; number_type 0 = UNORM, 6 = SRGB; comp_swap 0 = standard (RGBA),
    // 1 = alt (BGRA). (Kyty GraphicsRender.cpp RenderTextureFormat decode.)
    if (format == 0xAu) {
        const bool srgb = (number_type == 6u);
        const bool bgra = (comp_swap == 1u);
        if (!bgra) return srgb ? VkFormat::R8G8B8A8_SRGB : VkFormat::R8G8B8A8_UNORM;
        else       return srgb ? VkFormat::B8G8R8A8_SRGB : VkFormat::B8G8R8A8_UNORM;
    }
    return VkFormat::Undefined;   // surface not yet mapped
}

uint32_t vk_blend_factor(uint32_t f) {
    // RDNA2 blend factor -> VkBlendFactor value (Kyty GraphicsRender.cpp).
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
