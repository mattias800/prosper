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

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t man  = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;                          // +/- zero
        else {                                                    // subnormal: normalize into f32
            uint32_t e = 0, m = man;                              // value = man * 2^-24; after k shifts
            while (!(m & 0x400u)) { m <<= 1; e++; }               // it's 1.frac * 2^(-14-k), so the f32
            m &= 0x3FFu;                                          // exponent field is 127-14-k = 113-k
            bits = (sign << 31) | ((113u - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);          // inf / NaN (payload preserved)
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (man << 13); // normal: rebias 15 -> 127
    }
    float f;
    static_assert(sizeof(f) == sizeof(bits), "float is 32-bit");
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

float f11_to_float(uint16_t v) {
    // 5-bit exp (bias 15) + 6-bit mantissa, unsigned. Widen the mantissa into a half's 10-bit
    // field: subnormal m/64*2^-14 == (m<<4)/1024*2^-14, normal/inf/NaN carry the exponent as-is.
    return half_to_float((uint16_t)(((v & 0x7FFu) >> 6 << 10) | ((v & 0x3Fu) << 4)));
}

float f10_to_float(uint16_t v) {
    // 5-bit exp (bias 15) + 5-bit mantissa, unsigned.
    return half_to_float((uint16_t)(((v & 0x3FFu) >> 5 << 10) | ((v & 0x1Fu) << 5)));
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

const ShaderResource* ShaderResourceTable::by_sgpr_base_cls(uint32_t sgpr, ResourceClass cls) const {
    if (sgpr == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.sgpr_base == sgpr && r.cls == cls) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_fetch_pc(uint32_t pc) const {
    if (pc == 0xFFFFFFFFu) return nullptr;
    for (const auto& r : resources) if (r.fetch_pc == pc) return &r;
    return nullptr;
}

const ShaderResource* ShaderResourceTable::by_binding(uint32_t binding) const {
    for (const auto& r : resources) if (r.binding == binding) return &r;
    return nullptr;
}

} // namespace prosper::gpu
