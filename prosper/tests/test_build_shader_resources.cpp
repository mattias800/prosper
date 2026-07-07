// test_build_shader_resources — guards the front-half resource-table builder + V# decode
// (agc_shader_layout.cpp). Constructs a synthetic shader whose user-data describes constant buffers,
// places real V# descriptors in the user-data SGPR block, and asserts build_shader_resources decodes
// base/stride/size/format and assigns provenance (srt_offset) + bindings — the contract the recompiler
// and pipeline consume. Pure/headless; validates the decode against hand-built descriptors.
#include "../src/gpu/agc_shader_layout.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Build a 4-dword V# (buffer resource): Base48, 14-bit stride @[16:29] of word1, num_records=word2,
// RDNA2 combined 7-bit FORMAT @[18:12] of word3 (GFX10/PS5 layout).
static void make_vsharp(uint32_t v[4], uint64_t base, uint32_t stride, uint32_t records,
                        uint32_t fmt) {
    v[0] = (uint32_t)(base & 0xffffffffu);
    v[1] = (uint32_t)((base >> 32) & 0xffffu) | ((stride & 0x3fffu) << 16);
    v[2] = records;
    v[3] = (fmt & 0x7Fu) << 12;
}

int main() {
    printf("== test_build_shader_resources ==\n");

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
    }
    {   // RDNA2 combined-format decode coverage (the four game-observed anchors + a real V# regression).
        DataFormat f; uint32_t n;
        rdna2_buffer_format(74, &f, &n); CHECK(f == DataFormat::Float32 && n == 3, "fmt74 -> Float32 x3 (positions)");
        rdna2_buffer_format(64, &f, &n); CHECK(f == DataFormat::Float32 && n == 2, "fmt64 -> Float32 x2 (uvs)");
        rdna2_buffer_format(56, &f, &n); CHECK(f == DataFormat::Unorm8 && n == 4, "fmt56 -> Unorm8 x4 (colors)");
        rdna2_buffer_format(22, &f, &n); CHECK(f == DataFormat::Float32 && n == 1, "fmt22 -> Float32 x1");
        // The game's real color V# dword3 == 0x38fac: FORMAT field [18:12] == 56 (dst_sel [11:0] ignored).
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

    const ShaderResource* r0 = t.by_srt_offset(4 * 4);   // provenance key = offset_dw*4 bytes
    const ShaderResource* r1 = t.by_srt_offset(12 * 4);
    CHECK(r0 && r0->cls == ResourceClass::ConstantBuffer, "cbuf0 resolvable by srt_offset");
    CHECK(r0 && r0->gpu_addr == 0xA0000000ull && r0->size == 256, "cbuf0 base+size decoded");
    CHECK(r1 && r1->gpu_addr == 0xB0000000ull && r1->size == 1024, "cbuf1 base+size decoded");
    CHECK(r0 && r1 && r0->binding != r1->binding, "distinct bindings assigned");
    CHECK(t.by_binding(r0 ? r0->binding : 999) == r0, "resolvable by binding");
    CHECK(t.by_srt_offset(0x999) == nullptr, "unknown srt_offset -> null");

    // An empty slot (0x7fff) is skipped.
    cbuf_sharps[1].bits = 0x7fff;
    ShaderResourceTable t2 = build_shader_resources(shdr, sgprs, 32);
    CHECK(t2.resources.size() == 1, "empty sharp slot (0x7fff) skipped");

    // --- vertex buffers: direct resource usage type 8 (V# inline in the user-data SGPRs) ------------
    // A 16-entry direct_resource_offset table; type 8 (vertex buffer) points at a V# at SGPR dword 20.
    uint16_t dro[16]; for (auto& x : dro) x = 0xffff;
    make_vsharp(&sgprs[20], 0xC0000000ull, 12, 90, /*fmt*/74);  // 32_32_32 FLOAT, stride 12
    dro[8] = 20;                                                            // vertex buffer V# at sgpr 20
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
    CHECK(t3.by_srt_offset(4 * 4) != nullptr, "constant buffers still present alongside vertex buffers");
    CHECK(t3.by_sgpr_base(99) == nullptr, "unknown sgpr_base -> null");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
