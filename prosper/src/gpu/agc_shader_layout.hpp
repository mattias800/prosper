// agc_shader_layout.hpp — the guest AGC/Gen5 shader-header + V#/T#/S# descriptor layout, and the
// front-half builder that turns a shader's resource usage into a ShaderResourceTable (the contract in
// shader_resources.hpp). The bit layouts here are the FRONT-HALF's to own (see RESOURCE_BINDING.md);
// the recompiler only consumes the resulting DataFormat/binding.
//
// Layouts cross-checked against Kyty (PS5/RDNA2 gfx1030, same target): the SDK `Shader`/`ShaderUserData`
// (Shader.h:911/974) and the buffer resource descriptor V# (ShaderBufferResource getters: Base48,
// Stride[16:29], NumRecords=word2, Nfmt[12:14], Dfmt[15:18]).
#pragma once
#include "shader_resources.hpp"
#include <cstdint>

namespace prosper::gpu {

// One entry of a ShaderUserData sharp array: a 16-bit {offset_dw:15, size:1}. offset_dw is the dword
// index of the descriptor within the shader's user-data SGPR block; 0x7fff = empty slot.
struct AgcShaderSharp {
    uint16_t bits = 0;
    uint32_t offset_dw() const { return bits & 0x7fffu; }
    uint32_t size()      const { return (bits >> 15) & 1u; }
    bool     empty()     const { return offset_dw() == 0x7fffu; }
};

// Shader resource-usage table (Kyty ShaderUserData, Shader.h:911). Categories (from ShaderParseUsage2):
//   direct_resource_offset[type]: type 8 = vertex buffer, type 10 = vertex attrib (value = SGPR index)
//   sharp_resource_offset[0]: textures2D   [2]: samplers   [3]: constant/storage buffers   [1]: unused
struct AgcShaderUserData {
    uint16_t*       direct_resource_offset;      // +0x00
    AgcShaderSharp* sharp_resource_offset[4];    // +0x08
    uint16_t        eud_size_dw;                 // +0x28
    uint16_t        srt_size_dw;                 // +0x2a
    uint16_t        direct_resource_count;       // +0x2c
    uint16_t        sharp_resource_count[4];     // +0x2e
};

// The SDK shader header (Kyty Shader.h:974) — only the fields the resource builder needs.
struct AgcShaderHeader {
    uint32_t           file_header;   // +0x00 '1234'
    uint32_t           version;       // +0x04
    AgcShaderUserData* user_data;     // +0x08
    const void*        code;          // +0x10
    uint8_t            pad[0x5a - 0x18];
    uint8_t            type;          // +0x5a (2=VS/ES, 1=PS, 0=CS)
};

// A decoded buffer resource descriptor (V#, 4 dwords). base/stride/num_records per Kyty's getters;
// format+num_components from the (dfmt,nfmt) split. `size_bytes` is the backing region size.
struct DecodedBufferDescriptor {
    uint64_t   base = 0;
    uint32_t   stride = 0;
    uint32_t   num_records = 0;
    uint32_t   size_bytes = 0;
    DataFormat format = DataFormat::Float32;
    uint32_t   num_components = 1;
};

// Decode a 4-dword V# (RDNA2/Gen5 buffer resource). Pure; exposed for reuse + testing.
DecodedBufferDescriptor decode_buffer_descriptor(const uint32_t v[4]);

// Map a GCN/Gen5 buffer (DFMT, NFMT) pair to a DataFormat + component count. Pure.
void buffer_format(uint32_t dfmt, uint32_t nfmt, DataFormat* out_fmt, uint32_t* out_components);

// FRONT-HALF DELIVERABLE: build the resource table a shader uses. `user_sgprs` is the shader's bound
// user-data SGPR block (num_user_sgprs dwords) — the V# descriptors live there at each sharp's
// offset_dw (Kyty ShaderParseUsage2 reads them from the user_sgpr block, not the header). Descriptor
// gpu_addr is a 1:1-mapped guest pointer the pipeline binds directly.
//
// Signature takes the user-data SGPR block because the V# descriptor bytes live there (confirmed with
// agent 1). Fills constant buffers (sharp[3], INDIRECT srt_offset provenance) + vertex buffers
// (direct usage types 8/10, DIRECT sgpr_base provenance). Textures/samplers follow.
ShaderResourceTable build_shader_resources(const AgcShaderHeader& shdr,
                                           const uint32_t* user_sgprs, uint32_t num_user_sgprs);

} // namespace prosper::gpu
