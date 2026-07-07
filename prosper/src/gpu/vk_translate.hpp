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
    R8G8B8A8_UNORM = 37,
    R8G8B8A8_SRGB  = 43,
    B8G8R8A8_UNORM = 44,
    B8G8R8A8_SRGB  = 50,
};

// Map a CB_COLOR surface (FORMAT, NUMBER_TYPE, COMP_SWAP) triple to a VkFormat. Follows Kyty's
// GraphicsRender.cpp RenderTextureFormat decode; currently covers the 8_8_8_8 (format 0xA) surfaces
// the target uses (RGBA/BGRA × UNORM/SRGB). Returns Undefined for not-yet-mapped surfaces.
VkFormat vk_color_format(uint32_t format, uint32_t number_type, uint32_t comp_swap);

// RDNA2 DB_DEPTH_CONTROL.ZFUNC enumerates compare ops in the SAME order as VkCompareOp
// (0=NEVER,1=LESS,2=EQUAL,3=LEQUAL,4=GREATER,5=NOTEQUAL,6=GEQUAL,7=ALWAYS), so the value maps 1:1.
inline uint32_t vk_compare_op(uint32_t zfunc) { return zfunc & 0x7u; }

// Map an RDNA2 CB_BLEND_CONTROL blend factor to a VkBlendFactor value (NOT identity — e.g. RDNA2
// DstColor=8 -> VK DST_COLOR=4). Per Kyty GraphicsRender.cpp. Unknown -> ZERO.
uint32_t vk_blend_factor(uint32_t rdna2_factor);

// Map an RDNA2 COLOR_COMB_FCN to a VkBlendOp value (0=ADD,1=SUB,2=MIN,3=MAX,4=REV_SUB ->
// VK ADD=0,SUB=1,MIN=3,MAX=4,REV_SUB=2). Unknown -> ADD.
uint32_t vk_blend_op(uint32_t comb_fcn);

} // namespace prosper::gpu
