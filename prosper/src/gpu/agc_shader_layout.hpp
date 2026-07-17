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
#include <array>
#include <cstddef>
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
//   sharp_resource_offset[0]: read-only T#/V# resources (4-dword V# entries use size=1; some T#s do too)
//                         [1]: writable T#/V# resources (4-dword V# entries use size=1)
//                         [2]: samplers   [3]: constant/storage buffers
struct AgcShaderUserData {
    uint16_t*       direct_resource_offset;      // +0x00
    AgcShaderSharp* sharp_resource_offset[4];    // +0x08
    uint16_t        eud_size_dw;                 // +0x28
    uint16_t        srt_size_dw;                 // +0x2a
    uint16_t        direct_resource_count;       // +0x2c
    uint16_t        sharp_resource_count[4];     // +0x2e
};

// The shader "specials" block (Kyty Shader.h:947; pointed to by header +0x28) — only the user-data
// range. user_data_range_start/end give the DWORD range within the stage's SPI_SHADER_USER_DATA_*
// register block that this shader's SGPR-visible user data occupies (e.g. DOLL's UE4 Slate VS
// declares [0,8) for its {V#, ptr, ptr} block). Every shader observed live declares start=0; the
// executor honors a non-zero start as latent support (guarded).
struct AgcShaderSpecials {
    uint8_t  pad[0x14];
    uint16_t user_data_range_start;   // +0x14 (dwords, relative to USER_DATA_<stage>_0)
    uint16_t user_data_range_end;     // +0x16
};

// One packed shader input/output semantic (Kyty Shader.h:958). Keep the storage as a raw dword:
// C++ bitfield allocation is implementation-defined, while these accessors describe the SDK ABI.
struct AgcShaderSemantic {
    uint32_t bits = 0;
    uint32_t semantic()         const { return  bits        & 0xffu; }
    uint32_t hardware_mapping() const { return (bits >>  8) & 0xffu; }
    bool     is_flat_shaded()   const { return ((bits >> 22) & 1u) != 0; }
    uint32_t default_value()    const { return (bits >> 28) & 0x3u; }
    uint32_t default_value_hi() const { return (bits >> 30) & 0x3u; }
};
static_assert(sizeof(AgcShaderSemantic) == 4, "AGC semantic is one dword");

// The SDK shader header (Kyty Shader.h:974) — only the fields the resource builder needs.
struct AgcShaderHeader {
    uint32_t           file_header;   // +0x00 '1234'
    uint32_t           version;       // +0x04
    AgcShaderUserData* user_data;     // +0x08
    const void*        code;          // +0x10
    const void*        cx_registers;  // +0x18
    const void*        sh_registers;  // +0x20
    const AgcShaderSpecials* specials;// +0x28
    const AgcShaderSemantic* input_semantics;  // +0x30
    const AgcShaderSemantic* output_semantics; // +0x38
    uint32_t           header_size;             // +0x40
    uint32_t           shader_size;             // +0x44
    uint32_t           embedded_constant_buffer_size_dqw; // +0x48
    uint32_t           target;                  // +0x4c
    uint32_t           num_input_semantics;     // +0x50
    uint16_t           scratch_size_dw_per_thread; // +0x54
    uint16_t           num_output_semantics;    // +0x56
    uint16_t           special_sizes_bytes;     // +0x58
    uint8_t            type;          // +0x5a (2=VS/ES, 1=PS, 0=CS)
    uint8_t            num_cx_registers; // +0x5b
    uint8_t            num_sh_registers; // +0x5c
};
static_assert(offsetof(AgcShaderHeader, input_semantics) == 0x30 &&
              offsetof(AgcShaderHeader, num_input_semantics) == 0x50 &&
              offsetof(AgcShaderHeader, type) == 0x5a,
              "AgcShaderHeader must match the SDK blob layout");

// Hardware SPI_PS_INPUT_CNTL values derived from an ES/GS producer and a PS consumer's semantic
// metadata. Each valid control maps one logical PS input either to the matching producer PARAM slot
// or to OFFSET=0x20 + DEFAULT_VAL when the producer does not export that semantic.
struct AgcPixelInputControls {
    std::array<uint32_t, 32> controls{};
    uint32_t valid_mask = 0;
};
AgcPixelInputControls derive_agc_pixel_input_controls(const AgcShaderHeader* producer,
                                                      const AgcShaderHeader* pixel);

// A decoded buffer resource descriptor (V#, 4 dwords). base/stride/num_records per Kyty's getters;
// format+num_components from the (dfmt,nfmt) split. `size_bytes` is the backing region size.
struct DecodedBufferDescriptor {
    uint64_t   base = 0;
    uint32_t   stride = 0;
    uint32_t   num_records = 0;
    uint32_t   size_bytes = 0;
    DataFormat format = DataFormat::Float32;
    uint32_t   num_components = 1;
    // True when the descriptor names a packed format whose exact semantics are known to be
    // unsupported (currently non-identity DST_SEL or scaled 2_10_10_10). Dynamic fetch recovery
    // must not reinterpret this deliberate rejection as the legacy unresolved-format Float32 fallback.
    bool       forbid_unknown_fallback = false;
};

// Decode a 4-dword V# (RDNA2/Gen5 buffer resource). Pure; exposed for reuse + testing.
DecodedBufferDescriptor decode_buffer_descriptor(const uint32_t v[4]);

// A decoded image resource descriptor (T#, 8 dwords / 256-bit). Layout = Kyty ShaderTextureResource
// Gen5 getters (Base40/Width5/Height5/Format/TileMode). `base` is the byte address of the texel data
// in unified guest memory. `tile_mode` 0 = linear (no detiling needed); non-zero = GPU-tiled. GFX10
// DCC state lives in WORD6/7 and is kept separately from the base allocation: compressed texels are
// not meaningful without the metadata surface.
struct DecodedImageDescriptor {
    uint64_t base = 0;
    uint32_t width = 0, height = 0, depth = 1;
    uint32_t format = 0;      // Gen5 surface-format enum (fields[1] bits 20..28)
    uint32_t tile_mode = 0;   // 0 = linear
    uint8_t  base_level = 0;  // WORD3[15:12]
    uint8_t  last_level = 0;  // WORD3[19:16]
    uint8_t  max_mip = 0;     // WORD5[7:4], total allocation's final mip level
    uint8_t  type = 0;        // SQ_RSRC_IMG dim (GFX10: 8=1D, 9=2D, 10=3D, 11=CUBE, 12=1D_ARRAY, 13=2D_ARRAY).
                              // NB: 9 = plain 2D (already handled); 2D_ARRAY is 13 — the decode code in
                              // agc_shader_layout.cpp / gpu_executor.cpp uses these correct values (#378).
    // DST_SEL_X/Y/Z/W channel swizzle (WORD3 [2:0]/[5:3]/[8:6]/[11:9]); SQ_SEL enum:
    // 0=0, 1=1, 4=X(R), 5=Y(G), 6=Z(B), 7=W(A). Default = identity (R,G,B,A).
    uint8_t  dst_sel[4] = {4, 5, 6, 7};
    uint32_t max_uncompressed_block_size = 0; // WORD6[16:15]
    uint32_t max_compressed_block_size = 0;   // WORD6[18:17]
    bool     meta_pipe_aligned = false;       // WORD6[19]
    bool     write_compress_enabled = false;  // WORD6[20]
    bool     compression_enabled = false;     // WORD6[21]
    bool     alpha_is_on_msb = false;          // WORD6[22]
    bool     color_transform = false;          // WORD6[23]
    uint64_t metadata_addr = 0;                // 40-bit META_DATA_ADDRESS, in 256-byte units
};
// Decode an 8-dword T# (RDNA2/Gen5 image resource). Pure; exposed for reuse + testing.
DecodedImageDescriptor decode_image_descriptor(const uint32_t t[8]);

// Convert SQ_RSRC_IMG TYPE (8..15) to the MIMG dim convention (0..7). Unknown values retain the
// long-standing 2D fallback.
uint32_t image_type_to_dim(uint8_t type);

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
    bool       snorm = false;  // signed-normalized BCn variant (BC4/BC5_SNORM, fmt 176/178): the
                               // UNORM8 upload path can't carry signed samples, so callers keep
                               // skipping these (the UNORM decode would be numerically wrong).
};

// The single mip level Prosper uploads for a T# view. GFX10 descriptor BASE_ADDRESS names the
// entire allocation, whose mip-tail-first layout can put BASE_LEVEL at a non-zero byte offset.
struct DecodedImageView {
    uint64_t base = 0;
    uint32_t width = 0, height = 0;
    size_t mip_offset = 0;
    // False when BASE_LEVEL selects a mip whose layout this helper cannot prove. Callers must reject
    // the binding instead of silently sampling the allocation's base level with the wrong dimensions.
    bool supported = true;
};
DecodedImageView image_base_level_view(const DecodedImageDescriptor& descriptor,
                                       const Gen5ImageFormatInfo& format);
// Emit one diagnostic per unsupported layout signature. Callers use this immediately before
// rejecting the binding; deduplication keeps per-draw resource rebuilds from flooding stderr.
void warn_unsupported_image_view(const DecodedImageDescriptor& descriptor);
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
