// test_agc_shader -- focused guards for sceAgcCreateShader's guest-visible side effects.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace prosper;

namespace {

constexpr uint64_t kAgcErrInvalidArg = 0x8a6c000aull;

struct ShaderRegister { uint32_t offset, value; };
struct ShaderUserData {
    uint16_t* direct_resource_offset;
    void* sharp_resource_offset[4];
    uint16_t eud_size_dw, srt_size_dw, direct_resource_count, sharp_resource_count[4];
};
struct Shader {
    uint32_t file_header;
    uint32_t version;
    ShaderUserData* user_data;
    const void* code;
    ShaderRegister* cx_registers;
    ShaderRegister* sh_registers;
    void* specials;
    void* input_semantics;
    void* output_semantics;
    uint32_t header_size;
    uint32_t shader_size;
    uint32_t embedded_constant_buffer_size_dqw;
    uint32_t target;
    uint32_t num_input_semantics;
    uint16_t scratch_size_dw_per_thread;
    uint16_t num_output_semantics;
    uint16_t special_sizes_bytes;
    uint8_t type;
    uint8_t num_cx_registers;
    uint8_t num_sh_registers;
};

static_assert(offsetof(Shader, user_data) == 0x08 && offsetof(Shader, code) == 0x10 &&
              offsetof(Shader, specials) == 0x28 && offsetof(Shader, type) == 0x5a &&
              offsetof(Shader, num_sh_registers) == 0x5c, "test Shader layout must mirror hle_agc");

int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

} // namespace

extern "C" size_t prosper_agc_shader_count();

int main() {
    printf("== test_agc_shader ==\n");
    register_builtin_hle();

    HleFn create_shader = Hle::lookup("f3dg2CSgRKY");
    CHECK(create_shader != nullptr, "sceAgcCreateShader registered");
    if (!create_shader) return 1;

    size_t count0 = prosper_agc_shader_count();

    Shader bad{};
    bad.file_header = 0;
    bad.version = 0;
    bad.code = reinterpret_cast<const void*>(0x11111111ull);
    bad.cx_registers = reinterpret_cast<ShaderRegister*>(0x10ull);
    bad.sh_registers = reinterpret_cast<ShaderRegister*>(0x20ull);
    bad.specials = reinterpret_cast<void*>(0x30ull);
    bad.input_semantics = reinterpret_cast<void*>(0x40ull);
    bad.output_semantics = reinterpret_cast<void*>(0x50ull);

    Shader* dst = reinterpret_cast<Shader*>(0x12345678ull);
    uint64_t rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&bad),
                                0x1000, 0, 0, 0);
    CHECK(rc == kAgcErrInvalidArg, "invalid shader header is rejected");
    CHECK(dst == reinterpret_cast<Shader*>(0x12345678ull), "invalid shader does not write *dst");
    CHECK(bad.code == reinterpret_cast<const void*>(0x11111111ull), "invalid shader does not overwrite code");
    CHECK(bad.cx_registers == reinterpret_cast<ShaderRegister*>(0x10ull) &&
          bad.sh_registers == reinterpret_cast<ShaderRegister*>(0x20ull) &&
          bad.specials == reinterpret_cast<void*>(0x30ull) &&
          bad.input_semantics == reinterpret_cast<void*>(0x40ull) &&
          bad.output_semantics == reinterpret_cast<void*>(0x50ull),
          "invalid shader does not relocate pointer fields");
    CHECK(prosper_agc_shader_count() == count0, "invalid shader does not enter registry");

    Shader good{};
    good.file_header = 0x34333231u; // '1234'
    good.version = 0x18u;
    good.shader_size = 64;
    good.type = 1;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&good),
                       0x2000, 0, 0, 0);
    CHECK(rc == 0, "valid minimal shader header succeeds");
    CHECK(dst == &good, "valid shader writes *dst");
    CHECK(good.code == reinterpret_cast<const void*>(0x2000ull), "valid shader stores code pointer");
    CHECK(prosper_agc_shader_count() == count0 + 1, "valid shader enters registry");

    // #719: UE4 pixel shaders use buffer_load_format_* for structured/material data too. The
    // dynamic V# fold is stage-agnostic; build_stage_table must retain its result for PS instead of
    // discarding it under the old "PS has no vertex fetch" assumption. Otherwise the recompiler
    // silently accesses its legacy binding 2, which the PS table does not provide.
    const uint32_t pixel_buffer_fetch[] = {
        0xBE8803FFu, 0x00020000u,   // s_mov_b32 s8, 0x20000 (V# base low)
        0xBE8903FFu, 0x00100000u,   // s_mov_b32 s9, 0x00100000 (stride 16)
        0xBE8A03C0u,                // s_mov_b32 s10, 64 records
        0xBE8B0380u,                // s_mov_b32 s11, 0 (format defaults conservatively)
        0xE0002000u, 0x80020100u,   // pc=6: buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    Shader pixel{};
    pixel.file_header = 0x34333231u;
    pixel.version = 0x18u;
    pixel.shader_size = sizeof(pixel_buffer_fetch);
    pixel.type = 1;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&pixel),
                       reinterpret_cast<uint64_t>(pixel_buffer_fetch), 0, 0, 0);
    CHECK(rc == 0 && dst == &pixel, "pixel buffer-fetch shader enters the AGC registry");
    prosper::gpu::GpuState empty_state;
    auto pixel_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(pixel_buffer_fetch), true, 4);
    const prosper::gpu::ShaderResource* pixel_buffer = pixel_table
        ? pixel_table->by_fetch_pc(6) : nullptr;
    CHECK(pixel_buffer && pixel_buffer->cls == prosper::gpu::ResourceClass::ConstantBuffer &&
          pixel_buffer->gpu_addr == 0x20000u && pixel_buffer->stride == 16u,
          "pixel-stage buffer_load_format keeps its exact structured-buffer V# resource");

    // #158: the dynamic fold must use the registered header's byte size, not a fixed 64 KiB walk.
    // The valid-looking descriptor setup and fetch deliberately sit beyond the declared one-dword
    // shader. The old 0x4000-dword call found them; the bounded walk must not inspect them.
    const uint32_t trailing_fetch[] = {
        0xBF800000u,                // declared shader: s_nop 0
        0xBE8803FFu, 0x00020000u,   // outside blob: s_mov_b32 s8, V# base low
        0xBE8903FFu, 0x00100000u,   // s_mov_b32 s9, stride 16
        0xBE8A0380u,                // s_mov_b32 s10, 0 records
        0xBE8B0380u,                // s_mov_b32 s11, unknown format
        0xE0002000u, 0x80020100u,   // pc=7: buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    Shader bounded{};
    bounded.file_header = 0x34333231u;
    bounded.version = 0x18u;
    bounded.shader_size = sizeof(uint32_t);
    bounded.type = 2;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&bounded),
                       reinterpret_cast<uint64_t>(trailing_fetch), 0, 0, 0);
    CHECK(rc == 0 && dst == &bounded, "short shader enters the AGC registry");
    auto bounded_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(trailing_fetch), false, 7);
    CHECK(!bounded_table || !bounded_table->by_fetch_pc(7),
          "dynamic fetch walk cannot discover instructions beyond header.shader_size");

    // The declared size is also untrusted work metadata. Assert the exact production span calculation
    // directly: the previous implementation would request the whole declared range, while the bounded
    // implementation truncates partial dwords and retains the old 0x4000-dword work ceiling.
    CHECK(prosper::gpu::dynamic_fold_shader_dwords(3u) == 0u &&
          prosper::gpu::dynamic_fold_shader_dwords(7u) == 1u,
          "dynamic-fold span never probes a partial trailing dword");
    CHECK(prosper::gpu::dynamic_fold_shader_dwords(0x10000u) == 0x4000u &&
          prosper::gpu::dynamic_fold_shader_dwords(0xFFFFFFFCu) == 0x4000u,
          "dynamic-fold probe and decode request is capped at 0x4000 dwords");

    // Keep an integration assertion as well: a valid fetch just beyond that cap must stay hidden.
    std::vector<uint32_t> oversized_fetch(0x4000u + 10u, 0u);
    const size_t tail = 0x4000u;
    oversized_fetch[tail + 0] = 0xBE8803FFu;
    oversized_fetch[tail + 1] = 0x00020000u;
    oversized_fetch[tail + 2] = 0xBE8903FFu;
    oversized_fetch[tail + 3] = 0x00100000u;
    oversized_fetch[tail + 4] = 0xBE8A0380u;
    oversized_fetch[tail + 5] = 0xBE8B0380u;
    oversized_fetch[tail + 6] = 0xE0002000u;
    oversized_fetch[tail + 7] = 0x80020100u;
    oversized_fetch[tail + 8] = 0xBF810000u;
    Shader oversized{};
    oversized.file_header = 0x34333231u;
    oversized.version = 0x18u;
    oversized.shader_size = 0xFFFFFFFCu;
    oversized.type = 2;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&oversized),
                       reinterpret_cast<uint64_t>(oversized_fetch.data()), 0, 0, 0);
    CHECK(rc == 0 && dst == &oversized, "oversized shader metadata enters the AGC registry");
    auto oversized_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(oversized_fetch.data()), false, 7);
    CHECK(!oversized_table || !oversized_table->by_fetch_pc(static_cast<uint32_t>(tail + 6)),
          "oversized shader metadata cannot expand dynamic-fold work beyond 64 KiB");

    // A zero-record V# has no descriptor-provided byte bound. Size it from this draw instead of the
    // title-screen-specific stride*4 fallback. Unknown format remains a compatibility fallback, but
    // is surfaced and its unstrided record width is derived from the effective Float32x4 shape.
    const uint32_t zero_record_strided[] = {
        0xBE8803FFu, 0x00020000u,   // s_mov_b32 s8, V# base low
        0xBE8903FFu, 0x00140000u,   // s_mov_b32 s9, stride 20
        0xBE8A0380u,                // s_mov_b32 s10, 0 records
        0xBE8B0380u,                // s_mov_b32 s11, unknown format
        0xE0002000u, 0x80020100u,   // pc=6: buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    Shader strided{};
    strided.file_header = 0x34333231u;
    strided.version = 0x18u;
    strided.shader_size = sizeof(zero_record_strided);
    strided.type = 2;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&strided),
                       reinterpret_cast<uint64_t>(zero_record_strided), 0, 0, 0);
    CHECK(rc == 0 && dst == &strided, "zero-record strided shader enters the AGC registry");
    auto strided_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(zero_record_strided), false, 7);
    const prosper::gpu::ShaderResource* strided_buffer = strided_table
        ? strided_table->by_fetch_pc(6) : nullptr;
    CHECK(strided_buffer && strided_buffer->format == prosper::gpu::DataFormat::Float32 &&
          strided_buffer->num_components == 4 && strided_buffer->size == 7u * 20u,
          "zero-record strided V# size follows draw vertex count, not four vertices");

    // The same descriptor in a pixel shader is a VADDR-indexed structured/material buffer, not a
    // gl_VertexIndex-backed vertex attribute. A three-vertex draw therefore cannot shrink its bound
    // to three records: index 3 still needs the fourth strided record that the old compatibility
    // allocation exposed.
    const uint32_t ps_zero_record[] = {
        0xBE8803FFu, 0x00028000u,   // s_mov_b32 s8, V# base low
        0xBE8903FFu, 0x00100000u,   // s_mov_b32 s9, stride 16
        0xBE8A0380u,                // s_mov_b32 s10, 0 records
        0xBE8B0380u,                // s_mov_b32 s11, unknown format
        0xE0002000u, 0x80020100u,   // pc=6: buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    Shader ps_zero{};
    ps_zero.file_header = 0x34333231u;
    ps_zero.version = 0x18u;
    ps_zero.shader_size = sizeof(ps_zero_record);
    ps_zero.type = 1;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&ps_zero),
                       reinterpret_cast<uint64_t>(ps_zero_record), 0, 0, 0);
    CHECK(rc == 0 && dst == &ps_zero, "zero-record pixel shader enters the AGC registry");
    auto ps_zero_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(ps_zero_record), true, 3);
    const prosper::gpu::ShaderResource* ps_zero_buffer = ps_zero_table
        ? ps_zero_table->by_fetch_pc(6) : nullptr;
    CHECK(ps_zero_buffer &&
          ps_zero_buffer->cls == prosper::gpu::ResourceClass::ConstantBuffer &&
          ps_zero_buffer->stride == 16u && ps_zero_buffer->size == 4u * 16u &&
          3u * ps_zero_buffer->stride + sizeof(uint32_t) <= ps_zero_buffer->size,
          "pixel zero-record V# keeps VADDR index 3 in-bounds on a three-vertex draw");

    const uint32_t zero_record_unstrided[] = {
        0xBE8803FFu, 0x00030000u,   // s_mov_b32 s8, V# base low
        0xBE890380u,                // s_mov_b32 s9, stride 0
        0xBE8A0380u,                // s_mov_b32 s10, 0 records
        0xBE8B0380u,                // s_mov_b32 s11, unknown format
        0xE0002000u, 0x80020100u,   // pc=5: buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    Shader unstrided{};
    unstrided.file_header = 0x34333231u;
    unstrided.version = 0x18u;
    unstrided.shader_size = sizeof(zero_record_unstrided);
    unstrided.type = 2;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&unstrided),
                       reinterpret_cast<uint64_t>(zero_record_unstrided), 0, 0, 0);
    CHECK(rc == 0 && dst == &unstrided, "zero-record unstrided shader enters the AGC registry");
    auto unstrided_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(zero_record_unstrided), false, 7);
    const prosper::gpu::ShaderResource* unstrided_buffer = unstrided_table
        ? unstrided_table->by_fetch_pc(5) : nullptr;
    CHECK(unstrided_buffer && unstrided_buffer->size == 7u * 4u * sizeof(float),
          "zero-record unstrided V# size uses draw count and effective format, not magic 128");

    // Packed V# formats report no per-component byte size because their fields share one dword.
    // The draw-derived fallback must retain that physical four-byte record instead of collapsing
    // the resource to zero bytes when both NUM_RECORDS and STRIDE are zero.
    const uint32_t zero_record_packed[] = {
        0xBE8803FFu, 0x00040000u,   // s_mov_b32 s8, V# base low
        0xBE890380u,                // s_mov_b32 s9, stride 0
        0xBE8A0380u,                // s_mov_b32 s10, 0 records
        0xBE8B03FFu, (50u << 12) | 0xFACu, // identity 2_10_10_10_UNORM
        0xE00C2000u, 0x80020100u,   // pc=6: buffer_load_format_xyzw v[1:4], v0, s[8:11]
        0xBF810000u,
    };
    Shader packed{};
    packed.file_header = 0x34333231u;
    packed.version = 0x18u;
    packed.shader_size = sizeof(zero_record_packed);
    packed.type = 2;
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&packed),
                       reinterpret_cast<uint64_t>(zero_record_packed), 0, 0, 0);
    CHECK(rc == 0 && dst == &packed, "zero-record packed shader enters the AGC registry");
    auto packed_table = prosper::gpu::build_stage_table(
        empty_state, reinterpret_cast<uint64_t>(zero_record_packed), false, 7);
    const prosper::gpu::ShaderResource* packed_buffer = packed_table
        ? packed_table->by_fetch_pc(6) : nullptr;
    CHECK(packed_buffer && packed_buffer->format == prosper::gpu::DataFormat::Unorm2_10_10_10 &&
          packed_buffer->size == 7u * sizeof(uint32_t),
          "zero-record packed V# size uses one physical dword per drawn record");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
