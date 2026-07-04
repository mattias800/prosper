// shader_resources.cpp — see shader_resources.hpp. Pure lookups + format sizing; no Vulkan, no state.
#include "shader_resources.hpp"

namespace prosper::gpu {

uint32_t data_format_bytes(DataFormat f) {
    switch (f) {
        case DataFormat::Float32: case DataFormat::Uint32: case DataFormat::Sint32: return 4;
        case DataFormat::Float16: case DataFormat::Unorm16: case DataFormat::Snorm16:
        case DataFormat::Uint16:  case DataFormat::Sint16:  return 2;
        case DataFormat::Unorm8:  case DataFormat::Snorm8:
        case DataFormat::Uint8:   case DataFormat::Sint8:   return 1;
        default: return 0;
    }
}

const ShaderResource* ShaderResourceTable::by_srt_offset(uint32_t srt_offset) const {
    if (srt_offset == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.srt_offset == srt_offset) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_sgpr_base(uint32_t sgpr) const {
    if (sgpr == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.sgpr_base == sgpr) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_binding(uint32_t binding) const {
    for (const auto& r : resources) if (r.binding == binding) return &r;
    return nullptr;
}

} // namespace prosper::gpu
