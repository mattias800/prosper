// test_agc_shader -- focused guards for sceAgcCreateShader's guest-visible side effects.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

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
        empty_state, reinterpret_cast<uint64_t>(pixel_buffer_fetch), true);
    const prosper::gpu::ShaderResource* pixel_buffer = pixel_table
        ? pixel_table->by_fetch_pc(6) : nullptr;
    CHECK(pixel_buffer && pixel_buffer->cls == prosper::gpu::ResourceClass::ConstantBuffer &&
          pixel_buffer->gpu_addr == 0x20000u && pixel_buffer->stride == 16u,
          "pixel-stage buffer_load_format keeps its exact structured-buffer V# resource");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
