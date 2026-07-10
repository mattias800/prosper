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

    // half_to_float (#290 fp16 texture upload): exact IEEE binary16 decode incl. subnormals/inf/NaN.
    CHECK(half_to_float(0x3C00) == 1.0f && half_to_float(0xBC00) == -1.0f, "half 0x3C00/0xBC00 = +/-1.0");
    CHECK(half_to_float(0x3800) == 0.5f, "half 0x3800 = 0.5 (mid-gray magnitude, not saturated)");
    CHECK(half_to_float(0x0000) == 0.0f && half_to_float(0x8000) == 0.0f, "half +/-0 = 0.0");
    CHECK(half_to_float(0x4248) == 3.140625f, "half 0x4248 = 3.140625 (pi to half precision)");
    CHECK(half_to_float(0x7BFF) == 65504.0f, "half 0x7BFF = 65504 (max finite)");
    CHECK(half_to_float(0x0001) == 5.9604644775390625e-8f, "half 0x0001 = smallest subnormal");
    CHECK(half_to_float(0x03FF) == 6.0975551605224609375e-5f, "half 0x03FF = largest subnormal");
    { float inf = half_to_float(0x7C00); CHECK(inf > 3.4e38f && inf == inf * 2, "half 0x7C00 = +inf"); }
    { float nan = half_to_float(0x7E01); CHECK(nan != nan, "half 0x7E01 = NaN"); }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
