// test_shader_resources — fixes the resource-binding contract (shader_resources.hpp): format sizing
// and the recompiler/pipeline lookups both halves rely on. Pure (no Vulkan), runs in CI.
#include "../src/gpu/shader_resources.hpp"
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_shader_resources ==\n");

    CHECK(data_format_bytes(DataFormat::Float32) == 4 && data_format_bytes(DataFormat::Uint32) == 4,
          "32-bit formats are 4 bytes");
    CHECK(data_format_bytes(DataFormat::Float16) == 2 && data_format_bytes(DataFormat::Unorm16) == 2,
          "16-bit formats are 2 bytes");
    CHECK(data_format_bytes(DataFormat::Unorm8) == 1 && data_format_bytes(DataFormat::Sint8) == 1,
          "8-bit formats are 1 byte");
    CHECK(data_format_bytes(DataFormat::Unknown) == 0, "Unknown format is 0 bytes");

    // A table as the front-half would build it: a float32×4 constant buffer (descriptor at SRT 0x20)
    // and a unorm8×4 vertex buffer (descriptor at SRT 0x40).
    ShaderResourceTable t;
    t.resources.push_back({ResourceClass::ConstantBuffer, DataFormat::Float32, 4, /*binding*/2,
                           /*gpu_addr*/0xC0000000ull, /*size*/256, /*stride*/0, /*srt_offset*/0x20});
    t.resources.push_back({ResourceClass::VertexBuffer, DataFormat::Unorm8, 4, /*binding*/3,
                           /*gpu_addr*/0xD0000000ull, /*size*/4096, /*stride*/4, /*srt_offset*/0x40});

    // Recompiler's provenance lookup: descriptor at SRT 0x40 -> the vertex buffer at binding 3.
    const ShaderResource* v = t.by_srt_offset(0x40);
    CHECK(v && v->cls == ResourceClass::VertexBuffer && v->format == DataFormat::Unorm8 &&
          v->binding == 3 && v->stride == 4, "by_srt_offset resolves the descriptor to its resource");
    const ShaderResource* c = t.by_srt_offset(0x20);
    CHECK(c && c->cls == ResourceClass::ConstantBuffer && c->binding == 2 && c->gpu_addr == 0xC0000000ull,
          "by_srt_offset resolves the constant buffer");
    CHECK(t.by_srt_offset(0x99) == nullptr && t.by_srt_offset(0xFFFFFFFFu) == nullptr,
          "unknown / not-descriptor-keyed offset resolves to null");

    // Pipeline's lookup: bind by binding number.
    const ShaderResource* b3 = t.by_binding(3);
    CHECK(b3 && b3->gpu_addr == 0xD0000000ull && b3->size == 4096,
          "by_binding gives the pipeline the bytes to bind");

    // DIRECT provenance: a vertex-buffer V# placed straight in user-data SGPRs (s[8:11]) — keyed by
    // sgpr_base, not srt_offset (that's how vertex descriptors reach the shader; no in-shader s_load).
    ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
    vb.num_components = 3; vb.binding = 4; vb.stride = 12; vb.sgpr_base = 8;   // srt_offset stays 0xFFFFFFFF
    t.resources.push_back(vb);
    const ShaderResource* v2 = t.by_sgpr_base(8);
    CHECK(v2 && v2->cls == ResourceClass::VertexBuffer && v2->binding == 4 && v2->num_components == 3,
          "by_sgpr_base resolves a direct (user-data) vertex descriptor");
    CHECK(t.by_sgpr_base(0x99) == nullptr && t.by_sgpr_base(0xFFFFFFFFu) == nullptr,
          "unknown / not-SGPR-keyed resolves to null");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
