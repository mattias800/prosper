// agc_shader_layout.cpp — see agc_shader_layout.hpp. V# decode + the front-half resource-table build.
#include "agc_shader_layout.hpp"

namespace prosper::gpu {

// GCN/Gen5 buffer DATA_FORMAT (dfmt) -> component count. 0 = invalid/unhandled.
static uint32_t dfmt_components(uint32_t dfmt) {
    switch (dfmt) {
        case 1: case 2: case 4:            return 1;   // 8 / 16 / 32
        case 3: case 5: case 11:           return 2;   // 8_8 / 16_16 / 32_32
        case 13:                           return 3;   // 32_32_32
        case 10: case 12: case 14:         return 4;   // 8_8_8_8 / 16_16_16_16 / 32_32_32_32
        default:                           return 0;
    }
}
// dfmt -> per-component byte width (1/2/4).
static uint32_t dfmt_comp_bytes(uint32_t dfmt) {
    switch (dfmt) {
        case 1: case 3: case 10:           return 1;   // 8-bit components
        case 2: case 5: case 12:           return 2;   // 16-bit components
        case 4: case 11: case 13: case 14: return 4;   // 32-bit components
        default:                           return 0;
    }
}

void buffer_format(uint32_t dfmt, uint32_t nfmt, DataFormat* out_fmt, uint32_t* out_components) {
    uint32_t comp_bytes = dfmt_comp_bytes(dfmt);
    if (out_components) *out_components = dfmt_components(dfmt);
    // NFMT: 0=unorm, 1=snorm, 4=uint, 5=sint, 7=float.
    DataFormat f = DataFormat::Unknown;
    if (comp_bytes == 4) {
        f = (nfmt == 7) ? DataFormat::Float32 : (nfmt == 5) ? DataFormat::Sint32 : DataFormat::Uint32;
    } else if (comp_bytes == 2) {
        f = (nfmt == 7) ? DataFormat::Float16 : (nfmt == 0) ? DataFormat::Unorm16 : (nfmt == 1) ? DataFormat::Snorm16
          : (nfmt == 5) ? DataFormat::Sint16  : DataFormat::Uint16;
    } else if (comp_bytes == 1) {
        f = (nfmt == 0) ? DataFormat::Unorm8 : (nfmt == 1) ? DataFormat::Snorm8
          : (nfmt == 5) ? DataFormat::Sint8  : DataFormat::Uint8;
    }
    if (out_fmt) *out_fmt = f;
}

DecodedBufferDescriptor decode_buffer_descriptor(const uint32_t v[4]) {
    DecodedBufferDescriptor d;
    d.base        = ((uint64_t)v[0] | ((uint64_t)v[1] << 32)) & 0xFFFFFFFFFFFFull;  // Base48
    d.stride      = (v[1] >> 16) & 0x3FFFu;                                          // 14-bit stride
    d.num_records = v[2];
    uint32_t nfmt = (v[3] >> 12) & 0x7u;
    uint32_t dfmt = (v[3] >> 15) & 0xFu;
    buffer_format(dfmt, nfmt, &d.format, &d.num_components);
    // num_records is in units of `stride` when strided, else raw bytes.
    d.size_bytes = d.stride ? d.num_records * d.stride : d.num_records;
    return d;
}

ShaderResourceTable build_shader_resources(const AgcShaderHeader& shdr,
                                           const uint32_t* user_sgprs, uint32_t num_user_sgprs) {
    ShaderResourceTable table;
    const AgcShaderUserData* ud = shdr.user_data;
    if (!ud || !user_sgprs) return table;

    uint32_t binding = 0;

    // Constant buffers: sharp_resource_offset[3] (storage-as-constant). Each slot's offset_dw points at
    // a 4-dword V# in the user-data SGPR block. srt_offset = the descriptor's byte offset within
    // user_data (offset_dw * 4) — the recompiler's provenance key.
    const AgcShaderSharp* cbufs = ud->sharp_resource_offset[3];
    if (cbufs) {
        for (uint16_t slot = 0; slot < ud->sharp_resource_count[3]; slot++) {
            const AgcShaderSharp& s = cbufs[slot];
            if (s.empty()) continue;
            uint32_t off = s.offset_dw();
            if ((uint64_t)off + 4 > num_user_sgprs) continue;   // descriptor must fit in the SGPR block
            DecodedBufferDescriptor d = decode_buffer_descriptor(&user_sgprs[off]);
            ShaderResource r;
            r.cls            = ResourceClass::ConstantBuffer;
            r.format         = d.format;
            r.num_components  = d.num_components;
            r.binding        = binding++;
            r.gpu_addr       = d.base;
            r.size           = d.size_bytes;
            r.stride         = d.stride;
            r.srt_offset     = off * 4;   // byte offset within user_data
            table.resources.push_back(r);
        }
    }

    // (Next stage) vertex buffers via direct_resource_offset type 8/10 — the V# is reached through a
    // pointer in the SGPR block; deferred until stage 2 is wired with the recompiler.
    return table;
}

} // namespace prosper::gpu
