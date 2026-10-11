// Synthetic 3D unsigned storage instructions shared by compiler, validation and execution tests.
// No guest contents: exact format capabilities are supplied by each caller's own device contract.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::native_uint_3d {
using namespace prosper::gpu;

struct Format {
    DataFormat format;
    uint32_t components;
    uint32_t bytes;
    uint32_t spirv_format;
    uint32_t capability;
    const char* name;
};
inline constexpr std::array<Format, 4> formats{{
    {DataFormat::Uint32, 1, 4, kSpirvImageFormatR32ui, 1u << 25, "r32ui"},
    {DataFormat::Uint8, 1, 1, kSpirvImageFormatR8ui, 1u << 26, "r8ui"},
    {DataFormat::Uint16, 1, 2, kSpirvImageFormatR16ui, 1u << 27, "r16ui"},
    {DataFormat::Uint8, 4, 4, kSpirvImageFormatRgba8ui, 1u << 28, "rgba8ui"},
}};

inline std::vector<uint32_t> copy_row(uint32_t y = 0, uint32_t z = 0) {
    // v4=x, v5=y, v6=z; LOAD s[0:7] -> v[0:3]; STORE s[8:15] <- v[0:3].
    return {0x7e080300u, 0x7e0a02ffu, y,           0x7e0c02ffu, z,          0xf0000f10u,
            0x00000004u, 0xbf8c3f70u, 0xf0200f10u, 0x00020004u, 0xbf810000u};
}

inline ShaderResource image(const Format& format, uint32_t binding, uint32_t sgpr, uint32_t width,
                            uint32_t height, uint32_t depth, uint64_t address, uint32_t size,
                            uint32_t tile_mode = 0) {
    ShaderResource resource;
    resource.cls = ResourceClass::StorageImage;
    resource.format = format.format;
    resource.num_components = format.components;
    resource.binding = binding;
    resource.sgpr_base = sgpr;
    resource.img_dim = 2;
    resource.width = width;
    resource.height = height;
    resource.depth = depth;
    resource.gpu_addr = address;
    resource.size = size;
    resource.tile_mode = tile_mode;
    return resource;
}

inline ComputeShaderConfig config(uint32_t capability) {
    ComputeShaderConfig result;
    result.user_sgprs.resize(16);
    result.local_x = 64;
    result.local_y = result.local_z = 1;
    result.tidig_comp_cnt = 0;
    result.native_storage_format_support = capability;
    return result;
}
} // namespace prosper::test::native_uint_3d
