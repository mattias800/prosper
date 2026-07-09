// vk_translate.hpp — map decoded RDNA2 render-state enums to their Vulkan equivalents.
//
// Kept free of <vulkan.h> so the core library needn't link Vulkan: the enum values below are the
// exact numeric values of the corresponding VkPrimitiveTopology enumerators, so the Vulkan backend
// (M4) can `static_cast` directly. Mappings follow the RDNA2 VGT_DI_PRIMITIVE_TYPE enum and Kyty's
// GraphicsRender.cpp topology switch.
#pragma once
#include <cstdint>

namespace prosper::gpu {

// Values == VkPrimitiveTopology enumerators.
enum class VkTopology : uint32_t {
    PointList     = 0,   // VK_PRIMITIVE_TOPOLOGY_POINT_LIST
    LineList      = 1,
    LineStrip     = 2,
    TriangleList  = 3,
    TriangleStrip = 4,
    TriangleFan   = 5,
};

// Map a VGT_PRIMITIVE_TYPE.PRIM_TYPE value to a Vulkan topology. Unknown/unsupported types fall back
// to PointList (a safe, visible default) rather than asserting, so an unexpected stream still runs.
VkTopology vk_topology(uint32_t prim_type);

// Selected values == VkFormat enumerators (VK_FORMAT_UNDEFINED == 0).
enum class VkFormat : uint32_t {
    Undefined     = 0,
    R5G6B5_UNORM_PACK16 = 4,
    B5G6R5_UNORM_PACK16 = 5,
    R8_UNORM  = 9,  R8_SNORM  = 10, R8_UINT  = 13, R8_SINT  = 14,
    R8G8_UNORM = 16, R8G8_SNORM = 17, R8G8_UINT = 20, R8G8_SINT = 21,
    R8G8B8A8_UNORM = 37, R8G8B8A8_SNORM = 38, R8G8B8A8_UINT = 41, R8G8B8A8_SINT = 42,
    R8G8B8A8_SRGB  = 43,
    B8G8R8A8_UNORM = 44, B8G8R8A8_SNORM = 45, B8G8R8A8_UINT = 48, B8G8R8A8_SINT = 49,
    B8G8R8A8_SRGB  = 50,
    A2R10G10B10_UNORM_PACK32 = 58, A2R10G10B10_UINT_PACK32 = 62,
    A2B10G10R10_UNORM_PACK32 = 64, A2B10G10R10_UINT_PACK32 = 68,
    R16_UNORM = 70, R16_SNORM = 71, R16_UINT = 74, R16_SINT = 75, R16_SFLOAT = 76,
    R16G16_UNORM = 77, R16G16_SNORM = 78, R16G16_UINT = 81, R16G16_SINT = 82, R16G16_SFLOAT = 83,
    R16G16B16A16_UNORM = 91, R16G16B16A16_SNORM = 92, R16G16B16A16_UINT = 95,
    R16G16B16A16_SINT = 96, R16G16B16A16_SFLOAT = 97,
    R32_UINT = 98, R32_SINT = 99, R32_SFLOAT = 100,
    R32G32_UINT = 101, R32G32_SINT = 102, R32G32_SFLOAT = 103,
    R32G32B32A32_UINT = 107, R32G32B32A32_SINT = 108, R32G32B32A32_SFLOAT = 109,
    B10G11R11_UFLOAT_PACK32 = 122,
};

// Map a CB_COLOR surface (FORMAT, NUMBER_TYPE, COMP_SWAP) triple to a VkFormat, per the RDNA2
// CB_COLOR0_INFO field enums (COLOR_8..COLOR_32_32_32_32 × UNORM/SNORM/UINT/SINT/SRGB/FLOAT ×
// STD/ALT component swap). Kyty's RenderTextureFormat decode covers the 0xA/0x1 rows it runs;
// the rest follow the standard GCN/RDNA CB format table. Returns Undefined for an unmapped
// triple and logs it ONCE (previously every non-8888 target silently resolved to Undefined ->
// undefined pipeline format, #133).
VkFormat vk_color_format(uint32_t format, uint32_t number_type, uint32_t comp_swap);

// RDNA2 DB_DEPTH_CONTROL.ZFUNC enumerates compare ops in the SAME order as VkCompareOp
// (0=NEVER,1=LESS,2=EQUAL,3=LEQUAL,4=GREATER,5=NOTEQUAL,6=GEQUAL,7=ALWAYS), so the value maps 1:1.
inline uint32_t vk_compare_op(uint32_t zfunc) { return zfunc & 0x7u; }

// Map an RDNA2 DB_STENCIL_CONTROL stencil op (STENCIL_OP enum) to a VkStencilOp value. VkStencilOp:
// KEEP=0,ZERO=1,REPLACE=2,INC_CLAMP=3,DEC_CLAMP=4,INVERT=5,INC_WRAP=6,DEC_WRAP=7. The RDNA2 enum has
// three "replace-ish" codes (ONES/REPLACE_TEST/REPLACE_OP) that all realize as REPLACE here (a UI mask
// writes the ref value). AND/OR/XOR/etc. (10..14) have no Vk equivalent -> KEEP.
inline uint32_t vk_stencil_op(uint32_t op) {
    switch (op & 0xFu) {
        case 0:  return 0;   // KEEP
        case 1:  return 1;   // ZERO
        case 2:  return 2;   // ONES         -> REPLACE
        case 3:  return 2;   // REPLACE_TEST -> REPLACE
        case 4:  return 2;   // REPLACE_OP   -> REPLACE
        case 5:  return 3;   // ADD_CLAMP    -> INCREMENT_AND_CLAMP
        case 6:  return 4;   // SUB_CLAMP    -> DECREMENT_AND_CLAMP
        case 7:  return 5;   // INVERT
        case 8:  return 6;   // ADD_WRAP     -> INCREMENT_AND_WRAP
        case 9:  return 7;   // SUB_WRAP     -> DECREMENT_AND_WRAP
        default: return 0;   // AND/OR/XOR/NAND/NOR (no Vk equivalent) -> KEEP
    }
}

// Map an RDNA2 CB_BLEND_CONTROL blend factor to a VkBlendFactor value (NOT identity — e.g. RDNA2
// DstColor=8 -> VK DST_COLOR=4). Per Kyty GraphicsRender.cpp. Unknown -> ZERO.
uint32_t vk_blend_factor(uint32_t rdna2_factor);

// Map an RDNA2 COLOR_COMB_FCN to a VkBlendOp value (0=ADD,1=SUB,2=MIN,3=MAX,4=REV_SUB ->
// VK ADD=0,SUB=1,MIN=3,MAX=4,REV_SUB=2). Unknown -> ADD.
uint32_t vk_blend_op(uint32_t comb_fcn);

} // namespace prosper::gpu
