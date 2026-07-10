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

// A decoded image resource descriptor (T#, 8 dwords / 256-bit). Layout = Kyty ShaderTextureResource
// Gen5 getters (Base40/Width5/Height5/Format/TileMode). `base` is the byte address of the texel data
// in unified guest memory. `tile_mode` 0 = linear (no detiling needed); non-zero = GPU-tiled.
struct DecodedImageDescriptor {
    uint64_t base = 0;
    uint32_t width = 0, height = 0;
    uint32_t format = 0;      // Gen5 surface-format enum (fields[1] bits 20..28)
    uint32_t tile_mode = 0;   // 0 = linear
    uint8_t  type = 0;        // SQ_RSRC_IMG dim (8 = 2D, 9 = 2D_ARRAY, ...)
    // DST_SEL_X/Y/Z/W channel swizzle (WORD3 [2:0]/[5:3]/[8:6]/[11:9]); SQ_SEL enum:
    // 0=0, 1=1, 4=X(R), 5=Y(G), 6=Z(B), 7=W(A). Default = identity (R,G,B,A).
    uint8_t  dst_sel[4] = {4, 5, 6, 7};
};
// Decode an 8-dword T# (RDNA2/Gen5 image resource). Pure; exposed for reuse + testing.
DecodedImageDescriptor decode_image_descriptor(const uint32_t t[8]);

// A Gen5/GFX10 T# IMG_FMT (the 9-bit combined format field) decoded to sizing + conversion info.
// bytes_per_block is the byte size of one block_width x block_height texel block — for uncompressed
// formats block dims are 1x1 and it equals bytes-per-texel; for BCn blocks are 4x4.
struct Gen5ImageFormatInfo {
    DataFormat format         = DataFormat::Unknown;
    uint32_t   num_components = 0;
    uint32_t   bytes_per_block = 0;
    uint32_t   block_width = 1, block_height = 1;
    bool       srgb = false;   // gamma-encoded variant (sampling it as UNORM is a gamma error only,
                               // not garbage — the numbering distinguishes e.g. 56 vs 130)
};
// Map a T#'s 9-bit Gen5 IMG_FMT value to format info. Returns false (out left Unknown/zero) for
// values not in the table — callers must not assume RGBA8 for those (#65). Pure; exposed for testing.
bool gen5_image_format(uint32_t fmt, Gen5ImageFormatInfo* out);

// Map a GCN/Gen5 buffer (DFMT, NFMT) pair to a DataFormat + component count. Pure.
// Decode an RDNA2 (GFX10/PS5) buffer V# combined 7-bit FORMAT field (dword3 bits[18:12]) into a
// DataFormat + component count. (Replaces the GCN dfmt/nfmt split, which is wrong for PS5 V#s.)
void rdna2_buffer_format(uint32_t fmt, DataFormat* out_fmt, uint32_t* out_components);

// FRONT-HALF DELIVERABLE: build the resource table a shader uses. `user_sgprs` is the shader's bound
// user-data SGPR block (num_user_sgprs dwords) — the V# descriptors live there at each sharp's
// offset_dw (Kyty ShaderParseUsage2 reads them from the user_sgpr block, not the header). Descriptor
// gpu_addr is a 1:1-mapped guest pointer the pipeline binds directly.
//
// Signature takes the user-data SGPR block because the V# descriptor bytes live there (confirmed with
// agent 1). Fills constant buffers (sharp[3], INDIRECT srt_offset provenance) + vertex buffers
// (direct usage types 8/10, DIRECT sgpr_base provenance). Textures/samplers follow.
// `user_sgpr_base` = the shader SGPR the SPI loads the user-data block into (0 for PS; 8 for NGG VS/GS,
// whose s0..s7 are system SGPRs). Each resource's sgpr_base is set to that shader SGPR so the recompiler
// resolves an s_buffer_load / image_sample by its SBASE/SRSRC register (by_sgpr_base).
ShaderResourceTable build_shader_resources(const AgcShaderHeader& shdr,
                                           const uint32_t* user_sgprs, uint32_t num_user_sgprs,
                                           uint32_t user_sgpr_base = 0);

} // namespace prosper::gpu
