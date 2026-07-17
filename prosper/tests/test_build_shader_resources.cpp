// test_build_shader_resources — guards the front-half resource-table builder + V# decode
// (agc_shader_layout.cpp). Constructs a synthetic shader whose user-data describes constant buffers,
// places real V# descriptors in the user-data SGPR block, and asserts build_shader_resources decodes
// base/stride/size/format and assigns provenance (srt_offset) + bindings — the contract the recompiler
// and pipeline consume. Pure/headless; validates the decode against hand-built descriptors.
#include "../src/gpu/agc_shader_layout.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Build a 4-dword V# (buffer resource): Base48, 14-bit stride @[16:29] of word1, num_records=word2,
// RDNA2 combined 7-bit FORMAT @[18:12] of word3 and identity DST_SEL X/Y/Z/W in [11:0].
static void make_vsharp(uint32_t v[4], uint64_t base, uint32_t stride, uint32_t records,
                        uint32_t fmt) {
    v[0] = (uint32_t)(base & 0xffffffffu);
    v[1] = (uint32_t)((base >> 32) & 0xffffu) | ((stride & 0x3fffu) << 16);
    v[2] = records;
    v[3] = ((fmt & 0x7Fu) << 12) | 0xFACu;
}

// Build an 8-dword T# (Gen5 image resource): Base40 (stored >>8), 9-bit IMG_FMT @[28:20] of word1,
// Width5-1 split over word1[31:30] (lo) + word2[11:0] (hi), Height5-1 @word2[27:14], TileMode
// @word3[24:20], Type @word3[31:28]. Mirrors decode_image_descriptor / Kyty's Gen5 getters.
static void make_tsharp(uint32_t t[8], uint64_t base, uint32_t w, uint32_t h, uint32_t fmt,
                        uint32_t tile_mode, uint32_t type, uint32_t depth = 1) {
    memset(t, 0, 8 * sizeof(uint32_t));
    uint64_t b = base >> 8;                       // 256-byte-aligned base, stored >>8
    t[0] = (uint32_t)(b & 0xffffffffu);
    t[1] = (uint32_t)((b >> 32) & 0xffu) | ((fmt & 0x1ffu) << 20) | (((w - 1) & 0x3u) << 30);
    t[2] = (((w - 1) >> 2) & 0xfffu) | (((h - 1) & 0x3fffu) << 14);
    t[3] = ((tile_mode & 0x1fu) << 20) | ((type & 0xfu) << 28);
    if (type == 10) t[4] = (depth - 1) & 0x1fffu;
}

int main() {
    printf("== test_build_shader_resources ==\n");

    // --- Gen5 IMG_FMT mapper (pure table, #65) --------------------------------------------------
    {
        Gen5ImageFormatInfo fi;
        // The two title-live anchors: fmt=56 (composite RT, Kyty-confirmed) and fmt=1 (2048x1024 atlas).
        CHECK(gen5_image_format(56, &fi) && fi.format == DataFormat::Unorm8 && fi.num_components == 4 &&
              fi.bytes_per_block == 4 && fi.block_width == 1 && !fi.srgb, "IMG_FMT 56 -> Unorm8 x4, 4 B/texel");
        CHECK(gen5_image_format(1, &fi) && fi.format == DataFormat::Unorm8 && fi.num_components == 1 &&
              fi.bytes_per_block == 1, "IMG_FMT 1 -> Unorm8 x1 (R8), 1 B/texel");
        CHECK(gen5_image_format(14, &fi) && fi.format == DataFormat::Unorm8 && fi.num_components == 2 &&
              fi.bytes_per_block == 2, "IMG_FMT 14 -> Unorm8 x2 (RG8)");
        CHECK(gen5_image_format(71, &fi) && fi.format == DataFormat::Float16 && fi.num_components == 4 &&
              fi.bytes_per_block == 8, "IMG_FMT 71 -> Float16 x4 (RGBA16F), 8 B/texel");
        CHECK(gen5_image_format(130, &fi) && fi.format == DataFormat::Unorm8 && fi.num_components == 4 &&
              fi.bytes_per_block == 4 && fi.srgb, "IMG_FMT 130 -> Unorm8 x4 SRGB");
        CHECK(gen5_image_format(169, &fi) && fi.format == DataFormat::Bc1 && fi.bytes_per_block == 8 &&
              fi.block_width == 4 && fi.block_height == 4, "IMG_FMT 169 -> BC1, 8 B per 4x4 block");
        CHECK(gen5_image_format(173, &fi) && fi.format == DataFormat::Bc3 && fi.bytes_per_block == 16,
              "IMG_FMT 173 -> BC3, 16 B per 4x4 block");
        CHECK(gen5_image_format(182, &fi) && fi.format == DataFormat::Bc7 && fi.srgb,
              "IMG_FMT 182 -> BC7 SRGB");
        // #290: BC4/BC5 UNORM vs SNORM split — SNORM carries the snorm flag (upload keeps skipping it).
        CHECK(gen5_image_format(175, &fi) && fi.format == DataFormat::Bc4 && fi.bytes_per_block == 8 &&
              !fi.snorm, "IMG_FMT 175 -> BC4 UNORM (decodable)");
        CHECK(gen5_image_format(176, &fi) && fi.format == DataFormat::Bc4 && fi.snorm,
              "IMG_FMT 176 -> BC4 SNORM (flagged snorm, skip-only)");
        CHECK(gen5_image_format(177, &fi) && fi.format == DataFormat::Bc5 && fi.bytes_per_block == 16 &&
              !fi.snorm, "IMG_FMT 177 -> BC5 UNORM (decodable)");
        CHECK(gen5_image_format(178, &fi) && fi.format == DataFormat::Bc5 && fi.snorm,
              "IMG_FMT 178 -> BC5 SNORM (flagged snorm, skip-only)");
        // #294: fmt 36 = 10_11_11_FLOAT (UE4 R11G11B10F scene color) — packed 4-byte texel, 3 comps.
        CHECK(gen5_image_format(36, &fi) && fi.format == DataFormat::Float10_11_11 &&
              fi.num_components == 3 && fi.bytes_per_block == 4 && fi.block_width == 1 && !fi.srgb,
              "IMG_FMT 36 -> Float10_11_11 packed, 4 B/texel");
        CHECK(gen5_image_format(50, &fi) && fi.format == DataFormat::Unorm2_10_10_10 &&
              fi.num_components == 4 && fi.bytes_per_block == 4 && fi.block_width == 1 && !fi.srgb,
              "IMG_FMT 50 -> R10G10B10A2 UNORM packed, 4 B/texel");
        CHECK(!gen5_image_format(0, &fi) && fi.format == DataFormat::Unknown, "IMG_FMT 0 (INVALID) unmapped");
        CHECK(!gen5_image_format(9, &fi), "IMG_FMT 9 (16_USCALED) unmapped -> false");
        CHECK(!gen5_image_format(44, &fi), "IMG_FMT 44 (10_10_10_2) unmapped -> false");
        CHECK(!gen5_image_format(51, &fi) && !gen5_image_format(54, &fi) && !gen5_image_format(55, &fi),
              "packed SNORM/UINT/SINT buffer formats remain fail-closed for image upload");
        CHECK(!gen5_image_format(157, &fi), "IMG_FMT 157 (FMASK) unmapped -> false");
        CHECK(!gen5_image_format(511, &fi), "IMG_FMT 511 out-of-table -> false");
        CHECK(image_type_to_dim(8) == 0 && image_type_to_dim(9) == 1 &&
              image_type_to_dim(10) == 2 && image_type_to_dim(11) == 3,
              "T# TYPE 1D/2D/3D/CUBE maps to MIMG dims 0..3");
        CHECK(image_type_to_dim(12) == 4 && image_type_to_dim(13) == 5 &&
              image_type_to_dim(14) == 6 && image_type_to_dim(15) == 7,
              "T# TYPE array/MSAA variants map to MIMG dims 4..7");
        CHECK(image_type_to_dim(0) == 1, "unknown T# TYPE keeps the conservative 2D fallback");
        uint32_t t3d[8];
        make_tsharp(t3d, 0x12000000ull, 32, 16, /*fmt*/56, /*tile*/0, /*type 3D*/10, /*depth*/8);
        t3d[6] = (2u << 15) | (1u << 17) | (1u << 19) | (1u << 20) |
                 (1u << 21) | (1u << 22) | (1u << 23) | (0xabu << 24);
        t3d[7] = 0x00206e33u;
        const DecodedImageDescriptor d3d = decode_image_descriptor(t3d);
        CHECK(d3d.width == 32 && d3d.height == 16 && d3d.depth == 8,
              "3D T# word4 DEPTH decodes as depth-minus-one");
        CHECK(d3d.compression_enabled && d3d.write_compress_enabled && d3d.meta_pipe_aligned &&
              d3d.alpha_is_on_msb && d3d.color_transform &&
              d3d.max_uncompressed_block_size == 2 && d3d.max_compressed_block_size == 1 &&
              d3d.metadata_addr == 0x206e33ab00ull,
              "GFX10 T# WORD6/7 DCC flags and 40-bit metadata address decode exactly");

        uint32_t tmip[8];
        make_tsharp(tmip, 0x18000000ull, 2048, 1152, /*fmt*/56, /*SW_4KB_S*/5, /*2D*/9);
        tmip[3] |= 1u << 16;       // LAST_LEVEL
        tmip[5] |= 11u << 4;       // MAX_MIP: twelve levels in the allocation
        const DecodedImageDescriptor dmip0 = decode_image_descriptor(tmip);
        Gen5ImageFormatInfo mip_format;
        CHECK(gen5_image_format(56, &mip_format), "mip-view fixture format is mapped");
        const DecodedImageView vmip0 = image_base_level_view(dmip0, mip_format);
        CHECK(dmip0.base_level == 0 && dmip0.last_level == 1 && dmip0.max_mip == 11,
              "GFX10 T# base/last/MAX_MIP fields decode exactly");
        CHECK(vmip0.base == 0x18000000ull + 3186688 && vmip0.width == 2048 &&
                  vmip0.height == 1152 && vmip0.mip_offset == 3186688,
              "tiled level-zero view advances past the tail-first mip chain");
        tmip[3] |= 1u << 12;       // BASE_LEVEL 1, still outside the packed tail
        const DecodedImageDescriptor dmip1 = decode_image_descriptor(tmip);
        const DecodedImageView vmip1 = image_base_level_view(dmip1, mip_format);
        CHECK(vmip1.base == 0x18000000ull + 827392 && vmip1.width == 1024 &&
                  vmip1.height == 576 && vmip1.mip_offset == 827392,
              "non-tail base-level view applies its proven offset and dimensions");
        uint32_t tmip_cube[8]; memcpy(tmip_cube, tmip, sizeof tmip_cube);
        tmip_cube[3] = (tmip_cube[3] & ~(0xfu << 28)) | (11u << 28); // CUBE
        const DecodedImageDescriptor dmip_cube = decode_image_descriptor(tmip_cube);
        const DecodedImageView vmip_cube = image_base_level_view(dmip_cube, mip_format);
        CHECK(!vmip_cube.supported && vmip_cube.base == dmip_cube.base &&
                  vmip_cube.mip_offset == 0,
              "nonzero cube mip view is rejected until its slice/tail layout is modeled");
        tmip[3] &= ~(0x1fu << 20); // linear layout: offset is not modeled by the tiled helper
        const DecodedImageDescriptor dlinear = decode_image_descriptor(tmip);
        const DecodedImageView vlinear = image_base_level_view(dlinear, mip_format);
        CHECK(!vlinear.supported && vlinear.base == dlinear.base && vlinear.mip_offset == 0,
              "unmodeled nonzero linear mip view is rejected instead of sampling level zero");
    }

    // A shifted thin-2D mip emits the selected extent and a matching backing span. Allocation-level
    // DCC metadata cannot be reused without its mip offset, so the shifted view disables compression.
    {
        uint32_t sg[8];
        make_tsharp(sg, 0x18000000ull, 2048, 1152, /*fmt*/56, /*SW_4KB_S*/5, /*2D*/9);
        sg[3] |= 1u << 12; // BASE_LEVEL 1
        sg[5] |= 11u << 4;
        sg[6] = (1u << 19) | (1u << 20) | (1u << 21);
        sg[7] = 0x00123456u;
        AgcShaderSharp sharp[1]; sharp[0].bits = 0;
        AgcShaderUserData ud{};
        ud.sharp_resource_offset[0] = sharp;
        ud.sharp_resource_count[0] = 1;
        AgcShaderHeader sh{};
        sh.file_header = 0x34333231u; sh.version = 0x18; sh.type = 1; sh.user_data = &ud;
        const ShaderResourceTable resources = build_shader_resources(sh, sg, 8);
        const ShaderResource* mip = resources.by_sgpr_base(0);
        CHECK(mip && mip->gpu_addr == 0x18000000ull + 827392 && mip->width == 1024 &&
                  mip->height == 576 && mip->size == 1024u * 576u * 4u,
              "shifted mip resource uses selected address, extent, and backing span");
        CHECK(mip && !mip->compression_enabled && !mip->write_compress_enabled &&
                  !mip->meta_pipe_aligned && mip->metadata_addr == 0,
              "shifted mip resource does not pair texels with unshifted DCC metadata");

        sg[3] &= ~(0x1fu << 20); // same nonzero BASE_LEVEL, now an unsupported linear chain
        const ShaderResourceTable unsupported = build_shader_resources(sh, sg, 8);
        CHECK(!unsupported.by_sgpr_base(0),
              "resource builder rejects an unmodeled nonzero mip view instead of binding level zero");
    }

    // --- AGC semantic metadata -> SPI_PS_INPUT_CNTL wiring ------------------------------------
    // Live DOLL shape: the producer exports semantics 15..18 in PARAM0..3, while the PS consumes
    // semantic 15 and semantic 18. Identity wiring would feed PS input 1 from PARAM1; semantic
    // matching must instead select PARAM3. A missing producer semantic uses DEFAULT_VAL.
    {
        AgcShaderSemantic producer_out[] = {
            {0x0000000fu}, {0x00000110u}, {0x00000211u}, {0x00000312u},
        };
        AgcShaderSemantic pixel_in[] = {{0x0000000fu}, {0x00000012u}};
        AgcShaderHeader producer{};
        producer.output_semantics = producer_out;
        producer.num_output_semantics = 4;
        AgcShaderHeader pixel{};
        pixel.input_semantics = pixel_in;
        pixel.num_input_semantics = 2;
        AgcPixelInputControls mapping = derive_agc_pixel_input_controls(&producer, &pixel);
        CHECK(mapping.valid_mask == 0x3u && mapping.controls[0] == 0u && mapping.controls[1] == 3u,
              "DOLL semantics map PS inputs {15,18} to producer PARAM slots {0,3}");

        producer.output_semantics = nullptr;
        producer.num_output_semantics = 0;
        mapping = derive_agc_pixel_input_controls(&producer, &pixel);
        CHECK(mapping.valid_mask == 0x3u && mapping.controls[0] == 0x20u && mapping.controls[1] == 0x20u,
              "procedural producer with no PARAM exports materializes PS input defaults");

        AgcShaderSemantic default_flat[] = {{0x30400055u}}; // DEFAULT_VAL=3 + flat, no matching output
        pixel.input_semantics = default_flat;
        pixel.num_input_semantics = 1;
        mapping = derive_agc_pixel_input_controls(&producer, &pixel);
        CHECK(mapping.valid_mask == 1u && mapping.controls[0] == 0x720u,
              "unmatched flat semantic preserves DEFAULT_VAL and FLAT_SHADE control bits");
    }

    // --- V# decode in isolation ---------------------------------------------------------------
    {
        uint32_t v[4];
        make_vsharp(v, 0x123456780ull, 16, 64, /*fmt*/77);   // 32_32_32_32 FLOAT, stride 16
        DecodedBufferDescriptor d = decode_buffer_descriptor(v);
        CHECK(d.base == 0x123456780ull, "V# base48 decoded");
        CHECK(d.stride == 16, "V# stride decoded");
        CHECK(d.num_records == 64, "V# num_records decoded");
        CHECK(d.size_bytes == 64 * 16, "V# size = records*stride");
        CHECK(d.format == DataFormat::Float32 && d.num_components == 4, "fmt77 -> Float32 x4");

        make_vsharp(v, 0x123456780ull, 4, 64, /*fmt*/50);
        d = decode_buffer_descriptor(v);
        CHECK(d.format == DataFormat::Unorm2_10_10_10 && d.num_components == 4,
              "packed V# with identity DST_SEL decodes normally");
        v[3] = (50u << 12) | 0x977u; // A/B/G/R permutation: SQ_SEL_W/Z/Y/X
        d = decode_buffer_descriptor(v);
        CHECK(d.format == DataFormat::Unknown && d.num_components == 0,
              "packed V# with a component permutation stays fail-closed");
        v[3] = (36u << 12);          // all four selectors synthesize constant zero
        d = decode_buffer_descriptor(v);
        CHECK(d.format == DataFormat::Unknown && d.num_components == 0,
              "packed V# with constant selectors stays fail-closed");
        v[3] = (77u << 12) | 0x977u;
        d = decode_buffer_descriptor(v);
        CHECK(d.format == DataFormat::Float32 && d.num_components == 4,
              "non-packed V# selector behavior is unchanged");
    }
    {   // RDNA2 combined-format decode coverage (the four game-observed anchors + a real V# regression).
        DataFormat f; uint32_t n;
        rdna2_buffer_format(74, &f, &n); CHECK(f == DataFormat::Float32 && n == 3, "fmt74 -> Float32 x3 (positions)");
        rdna2_buffer_format(64, &f, &n); CHECK(f == DataFormat::Float32 && n == 2, "fmt64 -> Float32 x2 (uvs)");
        rdna2_buffer_format(56, &f, &n); CHECK(f == DataFormat::Unorm8 && n == 4, "fmt56 -> Unorm8 x4 (colors)");
        rdna2_buffer_format(22, &f, &n); CHECK(f == DataFormat::Float32 && n == 1, "fmt22 -> Float32 x1");
        rdna2_buffer_format(36, &f, &n); CHECK(f == DataFormat::Float10_11_11 && n == 3,
                                               "fmt36 -> packed Float10_11_11 x3");
        rdna2_buffer_format(50, &f, &n); CHECK(f == DataFormat::Unorm2_10_10_10 && n == 4,
                                               "fmt50 -> packed Unorm2_10_10_10 x4");
        rdna2_buffer_format(51, &f, &n); CHECK(f == DataFormat::Snorm2_10_10_10 && n == 4,
                                               "fmt51 -> packed Snorm2_10_10_10 x4");
        rdna2_buffer_format(54, &f, &n); CHECK(f == DataFormat::Uint2_10_10_10 && n == 4,
                                               "fmt54 -> packed Uint2_10_10_10 x4");
        rdna2_buffer_format(55, &f, &n); CHECK(f == DataFormat::Sint2_10_10_10 && n == 4,
                                               "fmt55 -> packed Sint2_10_10_10 x4");
        rdna2_buffer_format(52, &f, &n); CHECK(f == DataFormat::Unknown && n == 0,
                                               "fmt52 2_10_10_10_USCALED stays fail-closed");
        rdna2_buffer_format(53, &f, &n); CHECK(f == DataFormat::Unknown && n == 0,
                                               "fmt53 2_10_10_10_SSCALED stays fail-closed");
        // The game's real color V# dword3 == 0x38fac: FORMAT [18:12] == 56 and identity DST_SEL == 0xFAC.
        uint32_t real_v3 = 0x38facu;
        rdna2_buffer_format((real_v3 >> 12) & 0x7Fu, &f, &n);
        CHECK(f == DataFormat::Unorm8 && n == 4, "real color V# 0x38fac -> Unorm8 x4");
    }

    // --- build_shader_resources: two constant buffers via sharp[3] --------------------------------
    // user-data SGPR block: V#0 at dword 4, V#1 at dword 12.
    uint32_t sgprs[32]; memset(sgprs, 0, sizeof sgprs);
    make_vsharp(&sgprs[4],  0xA0000000ull, 0, 256,  22);   // cbuf0: 256 bytes (stride 0), Float32 x1
    make_vsharp(&sgprs[12], 0xB0000000ull, 0, 1024, 22);   // cbuf1: 1024 bytes

    AgcShaderSharp cbuf_sharps[2];
    cbuf_sharps[0].bits = (uint16_t)(4  & 0x7fff);            // offset_dw=4,  size bit 0
    cbuf_sharps[1].bits = (uint16_t)(12 & 0x7fff);            // offset_dw=12
    AgcShaderUserData ud; memset(&ud, 0, sizeof ud);
    ud.sharp_resource_offset[3] = cbuf_sharps;
    ud.sharp_resource_count[3]  = 2;
    AgcShaderHeader shdr; memset(&shdr, 0, sizeof shdr);
    shdr.file_header = 0x34333231u; shdr.version = 0x18; shdr.type = 2;
    shdr.user_data = &ud;

    ShaderResourceTable t = build_shader_resources(shdr, sgprs, 32);
    CHECK(t.resources.size() == 2, "built 2 constant-buffer resources");

    // In-SGPR cbufs resolve by DIRECT provenance (the s_buffer_load SBASE SGPR), NOT by srt_offset:
    // their V# is never s_loaded, so the recompiler matches sgpr_base via by_sgpr_base_cls (#453).
    const ShaderResource* r0 = t.by_sgpr_base(4);    // cbuf0 V# at user-SGPR dword 4 (user_sgpr_base 0)
    const ShaderResource* r1 = t.by_sgpr_base(12);   // cbuf1 V# at dword 12
    CHECK(r0 && r0->cls == ResourceClass::ConstantBuffer, "cbuf0 resolvable by sgpr_base (in-SGPR = DIRECT)");
    CHECK(r0 && r0->gpu_addr == 0xA0000000ull && r0->size == 256, "cbuf0 base+size decoded");
    CHECK(r1 && r1->gpu_addr == 0xB0000000ull && r1->size == 1024, "cbuf1 base+size decoded");
    CHECK(r0 && r1 && r0->binding != r1->binding, "distinct bindings assigned");
    CHECK(t.by_binding(r0 ? r0->binding : 999) == r0, "resolvable by binding");
    CHECK(r0 && r0->srt_offset == 0xFFFFFFFFu,
          "#453: in-SGPR cbuf leaves srt_offset UNSET (can't false-match an EUD cbuf at the same spill offset)");
    CHECK(t.by_sgpr_base(99) == nullptr, "unknown sgpr_base -> null");

    // An empty slot (0x7fff) is skipped.
    cbuf_sharps[1].bits = 0x7fff;
    ShaderResourceTable t2 = build_shader_resources(shdr, sgprs, 32);
    CHECK(t2.resources.size() == 1, "empty sharp slot (0x7fff) skipped");

    // --- #375: EUD-resident cbuf reads are bounds-checked (no SIGSEGV on an out-of-range offset_dw) --
    // An offset_dw >= num_user_sgprs lands in the guest EUD spill (base from direct_resource_offset[5]).
    // The read must be rejected when it runs past the declared eud_size_dw, instead of dereferencing
    // unmapped memory. Here: one in-bounds slot (eoff 0) resolves, one out-of-range slot is dropped.
    {
        uint32_t sg[32]; memset(sg, 0, sizeof sg);
        uint32_t eud[4]; make_vsharp(eud, 0xD0000000ull, 0, 512, 22);   // a real readable 4-dword spill
        uint64_t eud_ptr = (uint64_t)(uintptr_t)eud;
        uint16_t dro5[16]; for (auto& x : dro5) x = 0xffff;
        dro5[5] = 24; sg[24] = (uint32_t)eud_ptr; sg[25] = (uint32_t)(eud_ptr >> 32);   // EUD base pointer
        AgcShaderSharp eud_cbufs[2];
        eud_cbufs[0].bits = (uint16_t)(32 & 0x7fff);   // offset_dw 32 == num_user_sgprs -> EUD eoff 0 (in bounds)
        eud_cbufs[1].bits = (uint16_t)(40 & 0x7fff);   // offset_dw 40 -> eoff 8, +4 = 12 > eud_size_dw 4 -> rejected
        AgcShaderUserData eud_ud; memset(&eud_ud, 0, sizeof eud_ud);
        eud_ud.direct_resource_offset = dro5; eud_ud.direct_resource_count = 16;
        eud_ud.eud_size_dw = 4;                        // declared EUD size: 4 dwords
        eud_ud.sharp_resource_offset[3] = eud_cbufs; eud_ud.sharp_resource_count[3] = 2;
        AgcShaderHeader eud_sh; memset(&eud_sh, 0, sizeof eud_sh);
        eud_sh.file_header = 0x34333231u; eud_sh.version = 0x18; eud_sh.type = 2; eud_sh.user_data = &eud_ud;
        ShaderResourceTable te = build_shader_resources(eud_sh, sg, 32);
        CHECK(te.resources.size() == 1, "#375: in-bounds EUD cbuf resolves; slot past eud_size_dw skipped (no SIGSEGV)");
        const ShaderResource* re = te.by_srt_offset(0);   // EUD provenance key = eoff*4 = 0
        CHECK(re && re->gpu_addr == 0xD0000000ull, "#375: in-bounds EUD cbuf decoded from the spill buffer");
    }

    // --- #719: UE4 mixed sharp[0] buffers + sharp[1] writable buffer. -------------------------------
    // Live source 258 declares sharp_counts={8,1,1,2}. sharp[0] contains both size=0 T#s and size=1
    // V#s: a direct read-only V# at offset 8 and an EUD V# at offset 0x48. sharp[1] has the writable
    // V# at offset 0x4c. With 32 user SGPRs, those EUD entries have provenance keys 0xa0 and 0xb0.
    {
        uint32_t sg[32]; memset(sg, 0, sizeof sg);
        uint32_t eud[60]; memset(eud, 0, sizeof eud);
        make_vsharp(&sg[8], 0x10C109AC60ull, 16, 7, 5);       // live-shaped raw V#, unknown format
        make_vsharp(&eud[40], 0x207D300000ull, 16, 18, 77);   // sharp[0] EUD read-only V#
        make_vsharp(&eud[44], 0x207D400000ull, 16, 64, 77);
        const uint64_t eud_ptr = (uint64_t)(uintptr_t)eud;
        uint16_t dro5[11]; for (auto& x : dro5) x = 0xffff;
        dro5[5] = 12;
        sg[12] = (uint32_t)eud_ptr; sg[13] = (uint32_t)(eud_ptr >> 32);
        AgcShaderSharp readonly_buffers[2];
        readonly_buffers[0].bits = (uint16_t)(0x8000u | 0x08u);
        readonly_buffers[1].bits = (uint16_t)(0x8000u | 0x48u);
        AgcShaderSharp writable;
        writable.bits = (uint16_t)(0x8000u | 0x4cu);
        AgcShaderUserData wud; memset(&wud, 0, sizeof wud);
        wud.direct_resource_offset = dro5; wud.direct_resource_count = 11;
        wud.eud_size_dw = 60;
        wud.sharp_resource_offset[0] = readonly_buffers; wud.sharp_resource_count[0] = 2;
        wud.sharp_resource_offset[1] = &writable; wud.sharp_resource_count[1] = 1;
        AgcShaderHeader wsh; memset(&wsh, 0, sizeof wsh);
        wsh.file_header = 0x34333231u; wsh.version = 0x18; wsh.type = 0; wsh.user_data = &wud;

        ShaderResourceTable wt = build_shader_resources(wsh, sg, 32);
        const ShaderResource* direct = wt.by_sgpr_base(8);
        CHECK(direct && direct->cls == ResourceClass::ConstantBuffer &&
              direct->gpu_addr == 0x10C109AC60ull && direct->size == 112 && direct->stride == 16,
              "#719: sharp[0] size=1 direct entry decodes as a read-only V#, not a texture");
        const ShaderResource* indirect = wt.by_srt_offset(0xa0);
        CHECK(indirect && indirect->gpu_addr == 0x207D300000ull && indirect->size == 288,
              "#719: sharp[0] size=1 EUD entry resolves by its 0xa0 s_load provenance key");
        const ShaderResource* wr = wt.by_srt_offset(0xb0);
        CHECK(wr && wr->cls == ResourceClass::ConstantBuffer,
              "#719: sharp[1] writable V# becomes a storage-buffer-backed resource");
        CHECK(wr && wr->gpu_addr == 0x207D400000ull && wr->size == 1024 && wr->stride == 16,
              "#719: sharp[1] writable buffer descriptor decodes base/size/stride");
        CHECK(wr && wr->sgpr_base == 0xFFFFFFFFu,
              "#719: EUD-resident sharp[1] resolves only by its 0xb0 s_load provenance key");
    }

    // sharp[1] also contains writable images. Live volume kernels use size=0 8-dword T#/U# entries;
    // decoding their first four dwords as a V# loses the image descriptor's <<8 address scale and can
    // fabricate a >1 GiB storage-buffer range, which makes full capture fail preflight. Leave these for
    // the compute backend's dynamic storage-image fold instead of emitting a false buffer resource.
    {
        uint32_t sg[32]; memset(sg, 0, sizeof sg);
        make_tsharp(&sg[0], 0x207C670000ull, 120, 68, /*fmt*/56, /*tile*/27, /*type 2D*/9);
        AgcShaderSharp writable_image;
        writable_image.bits = 0;   // sharp[1] offset_dw=0, size=0: live writable image shape
        AgcShaderUserData iud; memset(&iud, 0, sizeof iud);
        iud.sharp_resource_offset[1] = &writable_image; iud.sharp_resource_count[1] = 1;
        AgcShaderHeader ish; memset(&ish, 0, sizeof ish);
        ish.file_header = 0x34333231u; ish.version = 0x18; ish.type = 0; ish.user_data = &iud;

        ShaderResourceTable it = build_shader_resources(ish, sg, 32);
        CHECK(it.resources.empty(),
              "#719: sharp[1] size=0 writable image is not misclassified as a giant buffer");
    }

    // --- vertex buffers: direct resource usage type 8 (V# inline in the user-data SGPRs) ------------
    // A 16-entry direct_resource_offset table; type 8 (vertex buffer) points at a V# at SGPR dword 20.
    uint16_t dro[16]; for (auto& x : dro) x = 0xffff;
    make_vsharp(&sgprs[16], 0xD0000000ull, 0, 0x0fe9c0c0u, /*fmt*/0); // user data: near-cap false V#
    make_vsharp(&sgprs[20], 0xC0000000ull, 12, 90, /*fmt*/74);  // 32_32_32 FLOAT, stride 12
    dro[8] = 20;                                                            // vertex buffer V# at sgpr 20
    dro[10] = 16;                                                           // metadata can name non-V# data
    ud.direct_resource_offset = dro;
    ud.direct_resource_count  = 16;
    cbuf_sharps[1].bits = (uint16_t)(12 & 0x7fff);   // restore cbuf1 so we test cbuf + vbuf together

    ShaderResourceTable t3 = build_shader_resources(shdr, sgprs, 32);
    const ShaderResource* vb = t3.by_sgpr_base(20);
    CHECK(vb != nullptr, "vertex buffer resolvable by sgpr_base (DIRECT provenance)");
    CHECK(vb && vb->cls == ResourceClass::VertexBuffer, "type-8 direct resource -> VertexBuffer");
    CHECK(vb && vb->format == DataFormat::Float32 && vb->num_components == 3, "vbuf format Float32 x3");
    CHECK(vb && vb->gpu_addr == 0xC0000000ull && vb->stride == 12 && vb->size == 90 * 12, "vbuf base/stride/size");
    CHECK(vb && vb->srt_offset == 0xFFFFFFFFu, "DIRECT vbuf leaves srt_offset unset");
    CHECK(t3.by_sgpr_base(16) == nullptr,
          "unknown-format user data just below the size cap is not guessed as a direct V#");
    CHECK(t3.by_sgpr_base(4) != nullptr, "constant buffers still present alongside vertex buffers");
    CHECK(t3.by_sgpr_base(99) == nullptr, "unknown sgpr_base -> null");

    make_vsharp(&sgprs[20], 0xC0000000ull, 4, 90, /*fmt*/50);
    sgprs[23] = (50u << 12) | 0x977u; // non-identity packed selector must be rejected before recompilation
    ShaderResourceTable packed_permuted = build_shader_resources(shdr, sgprs, 32);
    CHECK(packed_permuted.by_sgpr_base(20) == nullptr,
          "direct packed V# permutation is omitted from the runtime resource table");
    sgprs[23] = 50u << 12;             // constant-zero selectors are likewise unsupported
    ShaderResourceTable packed_constant = build_shader_resources(shdr, sgprs, 32);
    CHECK(packed_constant.by_sgpr_base(20) == nullptr,
          "direct packed V# constants are omitted from the runtime resource table");
    const uint32_t packed_fetch[] = {
        0x7e000f00u, 0xe0002000u, 0x80050100u, 0xbf810000u, // format_x v1, v0, s[20:23], 0 idxen
    };
    CHECK(recompile_valu(packed_fetch, sizeof(packed_fetch) / 4, 1, 1, &packed_permuted).empty() &&
          recompile_valu(packed_fetch, sizeof(packed_fetch) / 4, 1, 1, &packed_constant).empty(),
          "non-identity packed descriptors cannot reach emitted SPIR-V");
    make_vsharp(&sgprs[20], 0xC0000000ull, 4, 90, /*fmt*/50);
    ShaderResourceTable packed_identity = build_shader_resources(shdr, sgprs, 32);
    CHECK(packed_identity.by_sgpr_base(20) != nullptr &&
          !recompile_valu(packed_fetch, sizeof(packed_fetch) / 4, 1, 1, &packed_identity).empty(),
          "identity packed descriptor reaches the tested unpacking path");

    // --- Dead Cells compute direct resource: type 1 points at an inline V# in user SGPRs (#574). --
    {
        uint32_t csgprs[32]; memset(csgprs, 0, sizeof csgprs);
        make_vsharp(&csgprs[0], 0x74B4A8240000ull, 16, 0x1fe000, /*fmt*/75);
        uint16_t cdro[11]; for (auto& x : cdro) x = 0xffff;
        cdro[1] = 0;
        AgcShaderUserData cud; memset(&cud, 0, sizeof cud);
        cud.direct_resource_offset = cdro; cud.direct_resource_count = 11; cud.srt_size_dw = 8;
        AgcShaderHeader csh; memset(&csh, 0, sizeof csh);
        csh.file_header = 0x34333231u; csh.version = 0x18; csh.type = 0; csh.user_data = &cud;

        ShaderResourceTable ct = build_shader_resources(csh, csgprs, 32);
        const ShaderResource* cb = ct.by_sgpr_base(0);
        CHECK(cb && cb->cls == ResourceClass::ConstantBuffer,
              "compute type-1 direct V# emitted as a storage-buffer-backed resource");
        CHECK(cb && cb->gpu_addr == 0x74B4A8240000ull && cb->stride == 16 &&
              cb->size == 0x1fe0000, "compute direct V# base/stride/size decoded from live-shaped data");

        csh.type = 1;
        CHECK(build_shader_resources(csh, csgprs, 32).resources.empty(),
              "type-1 direct resource is not guessed for unobserved non-compute stages");
    }

    // --- #566: additional directly-placed compute buffers (store-destination V# not in metadata). ---
    // A Gen5 format-copy compute (Dead Cells) places TWO buffer V#s straight in the user-SGPR block —
    // load source at reg0, store destination at reg4 — but its resource-usage metadata declares only the
    // FIRST (direct_resource_offset type 1 -> reg 0). The store's SRSRC (s4) therefore had no resource and
    // the whole dispatch was rejected/dropped, losing a scene-composition pass. build_shader_resources
    // now also exposes plausible buffer V#s found directly in the user-SGPR block (keyed by sgpr_base),
    // while the same plausibility guard filters leftover non-descriptor SGPRs, and the scan is compute-only.
    {
        uint32_t csgprs[32]; memset(csgprs, 0, sizeof csgprs);
        make_vsharp(&csgprs[0], 0x74BA456E0000ull, 4, 393216, /*fmt*/2);   // src (declared via type 1)
        make_vsharp(&csgprs[4], 0x74BA2F3E0000ull, 4, 393216, /*fmt*/2);   // dst (NOT declared anywhere)
        make_vsharp(&csgprs[12], 0x50000000ull,    4, 0x8000000, /*fmt*/2);// 512 MB -> implausible, filtered
        // reg8 + reg16..28 stay zero -> base 0 -> filtered.
        uint16_t cdro[11]; for (auto& x : cdro) x = 0xffff;
        cdro[1] = 0;                                                       // declare ONLY reg0
        AgcShaderUserData cud; memset(&cud, 0, sizeof cud);
        cud.direct_resource_offset = cdro; cud.direct_resource_count = 11;
        AgcShaderHeader csh; memset(&csh, 0, sizeof csh);
        csh.file_header = 0x34333231u; csh.version = 0x18; csh.type = 0; csh.user_data = &cud;

        ShaderResourceTable ct = build_shader_resources(csh, csgprs, 32);
        const ShaderResource* src = ct.by_sgpr_base(0);
        const ShaderResource* dst = ct.by_sgpr_base(4);
        CHECK(src && src->gpu_addr == 0x74BA456E0000ull, "#566: declared compute source buffer at sgpr 0");
        CHECK(dst && dst->cls == ResourceClass::ConstantBuffer && dst->gpu_addr == 0x74BA2F3E0000ull,
              "#566: undeclared compute store-dest V# at sgpr 4 exposed (store SRSRC now resolves)");
        CHECK(dst && dst->size == 393216u * 4u && dst->stride == 4, "#566: dest V# base/size/stride decoded");
        CHECK(ct.by_sgpr_base(8) == nullptr, "#566: leftover base-0 SGPR slot not emitted");
        CHECK(ct.by_sgpr_base(12) == nullptr, "#566: >256 MB implausible V# filtered (no stale-SGPR binding)");
        int at0 = 0; for (const auto& r : ct.resources) if (r.sgpr_base == 0) at0++;
        CHECK(at0 == 1, "#566: declared source buffer emitted exactly once (scan dedups the type-1 slot)");

        csh.type = 1;   // PS: the additional-buffer scan is compute-only.
        CHECK(build_shader_resources(csh, csgprs, 32).by_sgpr_base(4) == nullptr,
              "#566: additional directly-placed-buffer scan does not run for non-compute stages");
    }

    // --- textures: sharp[0] T#s carry their real format + byte size; BC1/2/3 are bound (decoded to RGBA8
    // on upload, #121); BC4-7 + unmapped formats are still skipped (#65) ---
    {
        uint32_t tsgprs[32]; memset(tsgprs, 0, sizeof tsgprs);
        make_tsharp(&tsgprs[0],  0xD0000000ull, 1920, 1080, /*fmt*/56,  /*tile*/5, /*type 2D*/9);
        tsgprs[6] = (2u << 15) | (1u << 17) | (1u << 19) | (1u << 21) |
                    (1u << 22) | (0xabu << 24);
        tsgprs[7] = 0x00206e33u;
        make_tsharp(&tsgprs[8],  0xE0000000ull, 2048, 1024, /*fmt*/1,   /*tile*/5, /*type*/9);
        make_tsharp(&tsgprs[16], 0x207E870000ull, 256, 256, /*fmt*/169, /*tile*/5, /*type*/9);  // BC1
        make_tsharp(&tsgprs[24], 0xF0010000ull,  128,  128, /*fmt*/44,  /*tile*/5, /*type*/9);  // unmapped

        {   // T# decode round-trip of the first descriptor
            DecodedImageDescriptor d = decode_image_descriptor(&tsgprs[0]);
            CHECK(d.base == 0xD0000000ull && d.width == 1920 && d.height == 1080 &&
                  d.format == 56 && d.tile_mode == 5 && d.type == 9, "T# decode round-trip (base/w/h/fmt/tile/type)");
        }

        AgcShaderSharp tex_sharps[4];
        for (int i = 0; i < 4; i++) tex_sharps[i].bits = (uint16_t)(i * 8);
        tex_sharps[2].bits |= 0x8000u;  // live UE4 T#s may carry size=1: must remain textures, not fake V#s
        AgcShaderUserData tud; memset(&tud, 0, sizeof tud);
        tud.sharp_resource_offset[0] = tex_sharps;
        tud.sharp_resource_count[0]  = 4;
        AgcShaderHeader tshdr; memset(&tshdr, 0, sizeof tshdr);
        tshdr.file_header = 0x34333231u; tshdr.version = 0x18; tshdr.type = 1;   // PS
        tshdr.user_data = &tud;

        ShaderResourceTable tt = build_shader_resources(tshdr, tsgprs, 32);
        CHECK(tt.resources.size() == 3, "fmt56 + fmt1 + BC1 emitted; unmapped fmt44 skipped (3 textures)");

        const ShaderResource* t0 = tt.by_sgpr_base(0);
        CHECK(t0 && t0->cls == ResourceClass::Texture, "fmt=56 texture emitted (title composite path)");
        CHECK(t0 && t0->format == DataFormat::Unorm8 && t0->num_components == 4,
              "fmt=56 -> Unorm8 x4 (current behavior preserved)");
        CHECK(t0 && t0->size == 1920u * 1080u * 4u, "fmt=56 size = w*h*4 (unchanged)");
        CHECK(t0 && t0->compression_enabled && t0->meta_pipe_aligned && t0->alpha_is_on_msb &&
              !t0->write_compress_enabled && !t0->color_transform &&
              t0->max_uncompressed_block_size == 2 && t0->max_compressed_block_size == 1 &&
              t0->metadata_addr == 0x206e33ab00ull,
              "texture resource preserves all decoded DCC state for renderer/capture consumers");

        const ShaderResource* t1 = tt.by_sgpr_base(8);
        CHECK(t1 && t1->format == DataFormat::Unorm8 && t1->num_components == 1,
              "fmt=1 -> Unorm8 x1 (R8, no longer assumed RGBA)");
        CHECK(t1 && t1->size == 2048u * 1024u, "fmt=1 size = w*h (was w*h*4, an 8 MB over-read of 2 MB)");

        const ShaderResource* t2 = tt.by_sgpr_base(16);
        CHECK(t2 && t2->format == DataFormat::Bc1, "BC1 T# now bound (decoded to RGBA8 on upload)");
        CHECK(t2 && t2->gpu_addr == 0x207E870000ull,
              "#719: size=1 T# that fails V# shape checks remains a texture");
        CHECK(t2 && t2->size == ((256u + 3) / 4) * ((256u + 3) / 4) * 8u,
              "BC1 size = ceil(w/4)*ceil(h/4)*8 (compressed block bytes, not w*h*4)");
        CHECK(tt.by_sgpr_base(24) == nullptr, "unmapped IMG_FMT T# skipped (no silent RGBA8)");
    }

    // #382: an EUD-resident texture (T# spilled beyond the user-SGPR block) must be emitted, mirroring
    // the cbuf EUD path — the old hard `continue` dropped it, so the sampling draw rendered untextured.
    // Provenance is INDIRECT (srt_offset = the s_load immediate); sgpr_base is invalid so a stray
    // by_sgpr_base for an unrelated SGPR can't spuriously match its out-of-file index.
    {
        uint32_t sg[32]; memset(sg, 0, sizeof sg);
        // EUD spill: an 8-dword T# at eoff 0 + a paired 4-dword S# at eoff 8 (12 dwords total).
        uint32_t eud[12]; memset(eud, 0, sizeof eud);
        make_tsharp(&eud[0], 0xAB000000ull, 128, 128, /*fmt*/56, /*tile*/0, /*type 2D*/9);
        // #451: the paired S# also spills to the EUD. Program POINT filter + addr (wrap,mirror,clamp) so
        // it is distinguishable from the struct defaults (LINEAR + clamp). SQ_IMG_SAMP WORD0: CLAMP_X[2:0],
        // CLAMP_Y[5:3], CLAMP_Z[8:6]; WORD2 filter bits are 0 -> point.
        eud[8] = (0u << 0) | (1u << 3) | (2u << 6);   // addr_uvw = {0 wrap, 1 mirror, 2 clamp}
        uint64_t ep = (uint64_t)(uintptr_t)eud;
        uint16_t dro5[16]; for (auto& x : dro5) x = 0xffff;
        dro5[5] = 24; sg[24] = (uint32_t)ep; sg[25] = (uint32_t)(ep >> 32);   // EUD base pointer
        AgcShaderSharp et_sharp[1]; et_sharp[0].bits = (uint16_t)(32 & 0x7fff);   // T# offset_dw 32 -> eoff 0
        AgcShaderSharp es_sharp[1]; es_sharp[0].bits = (uint16_t)(40 & 0x7fff);   // S# offset_dw 40 -> eoff 8
        AgcShaderUserData etud; memset(&etud, 0, sizeof etud);
        etud.direct_resource_offset = dro5; etud.direct_resource_count = 16;
        etud.eud_size_dw = 12;
        etud.sharp_resource_offset[0] = et_sharp; etud.sharp_resource_count[0] = 1;
        etud.sharp_resource_offset[2] = es_sharp; etud.sharp_resource_count[2] = 1;   // paired sampler
        AgcShaderHeader etsh; memset(&etsh, 0, sizeof etsh);
        etsh.file_header = 0x34333231u; etsh.version = 0x18; etsh.type = 1; etsh.user_data = &etud;
        ShaderResourceTable et = build_shader_resources(etsh, sg, 32);
        CHECK(et.resources.size() == 1, "#382: EUD-resident texture is emitted (not dropped)");
        const ShaderResource* etr = et.by_srt_offset(0);   // EUD provenance key = eoff*4 = 0
        CHECK(etr && etr->cls == ResourceClass::Texture && etr->gpu_addr == 0xAB000000ull,
              "#382: EUD texture decoded from the spill buffer, resolvable by srt_offset");
        CHECK(etr && etr->sgpr_base == 0xFFFFFFFFu, "#382: EUD texture leaves sgpr_base invalid (INDIRECT provenance)");
        // #451: the EUD-resident paired S# is decoded (not left at the LINEAR/clamp defaults).
        CHECK(etr && etr->mag_filter == 0u && etr->min_filter == 0u,
              "#451: EUD-resident sampler decoded -> POINT filter (not the default LINEAR)");
        CHECK(etr && etr->addr_uvw[0] == 0u && etr->addr_uvw[1] == 1u && etr->addr_uvw[2] == 2u,
              "#451: EUD-resident sampler addr modes decoded from the spill (not default clamp)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
