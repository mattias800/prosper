// test_agc_shader -- focused guards for sceAgcCreateShader's guest-visible side effects.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <thread>
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

struct ShaderSpecialRegs {
    ShaderRegister ge_cntl;
    ShaderRegister vgt_shader_stages_en;
    uint32_t dispatch_modifier;
    uint16_t user_data_range_start;
    uint16_t user_data_range_end;
    uint64_t draw_modifier;
    ShaderRegister vgt_gs_out_prim_type;
    ShaderRegister ge_user_vgpr_en;
};

struct SizeAlign { uint64_t size, align; };

static_assert(offsetof(Shader, user_data) == 0x08 && offsetof(Shader, code) == 0x10 &&
              offsetof(Shader, specials) == 0x28 && offsetof(Shader, type) == 0x5a &&
              offsetof(Shader, num_sh_registers) == 0x5c, "test Shader layout must mirror hle_agc");

int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

} // namespace

extern "C" size_t prosper_agc_shader_count();
extern "C" const void* prosper_agc_fused_back_header_for_front(uint64_t front_code_addr);

int main() {
    printf("== test_agc_shader ==\n");
    register_builtin_hle();

    HleFn create_shader = Hle::lookup("f3dg2CSgRKY");
    HleFn fused_size = Hle::lookup("dolOmWH+huQ");
    HleFn fused_size_320 = Hle::lookup("nQT5kYLv0cg");
    HleFn fuse_halves = Hle::lookup("nApJjpKNBl4");
    HleFn fuse_halves_old = Hle::lookup("fd5Bp5tGTgo");
    HleFn create_interp = Hle::lookup("dbOlWdppb4o");
    HleFn create_interp_320 = Hle::lookup("pdEV7bI6COI");
    CHECK(create_shader != nullptr, "sceAgcCreateShader registered");
    CHECK(fused_size != nullptr && fused_size_320 != nullptr &&
          fuse_halves != nullptr && fuse_halves_old != nullptr,
          "fused-shader size query and SDK aliases registered");
    CHECK(create_interp != nullptr && create_interp_320 != nullptr,
          "CreateInterpolantMapping SDK aliases registered");
    if (!create_shader) return 1;

    // Pathless builds GS/HS shaders from separately compiled front/back binaries. A success-only
    // stub left its stack-local fused Shader untouched, and the later register copier interpreted
    // unrelated stack bytes as a nonzero count paired with a null register pointer.
    if (fused_size && fused_size_320 && fuse_halves && fuse_halves_old) {
        using namespace prosper::agc::Pm4;
        ShaderRegister front_regs[] = {
            {SPI_SHADER_PGM_CHKSUM_GS, 0x11111111u},
            {SPI_SHADER_PGM_CHKSUM_GS, 0x22222222u},
        };
        ShaderRegister back_regs[] = {
            {SPI_SHADER_PGM_LO_ES, 0},
            {SPI_SHADER_PGM_LO_ES + 1u, 0xabcdef00u},
            {SPI_SHADER_PGM_CHKSUM_GS, 0xaaaaaaaau},
            {SPI_SHADER_PGM_CHKSUM_GS, 0xbbbbbbbbu},
            {0x234u, 0xccccccccu},
        };
        ShaderSpecialRegs front_specials{}, back_specials{};
        Shader front{};
        front.type = 4; // GS front
        front.code = reinterpret_cast<const void*>(0x123456789a00ull);
        front.sh_registers = front_regs;
        front.num_sh_registers = (uint8_t)(sizeof(front_regs) / sizeof(front_regs[0]));
        front.specials = &front_specials;
        Shader back{};
        back.type = 6; // GS back
        back.code = reinterpret_cast<const void*>(0x20c0000000ull);
        back.sh_registers = back_regs;
        back.num_sh_registers = (uint8_t)(sizeof(back_regs) / sizeof(back_regs[0]));
        back.user_data = reinterpret_cast<ShaderUserData*>(0x12345678ull);
        back.specials = &back_specials;

        SizeAlign requirements{0xdeadbeefull, 0xdeadbeefull};
        uint64_t fused_rc = fused_size_320((uint64_t)(uintptr_t)&requirements,
                                           (uint64_t)(uintptr_t)&front,
                                           (uint64_t)(uintptr_t)&back, 0, 0, 0);
        CHECK(fused_rc == 0 && requirements.size == sizeof(back_regs) && requirements.align == 4,
              "fused-shader size query reserves one copied SH-register array");

        ShaderRegister scratch[sizeof(back_regs) / sizeof(back_regs[0])]{};
        Shader fused{};
        fused_rc = fuse_halves((uint64_t)(uintptr_t)&fused, (uint64_t)(uintptr_t)&front,
                               (uint64_t)(uintptr_t)&back, (uint64_t)(uintptr_t)scratch, 0, 0);
        CHECK(fused_rc == 0 && fused.type == 2 && fused.sh_registers == scratch,
              "GS halves produce a fused GS backed by caller scratch");
        CHECK(fused.user_data == nullptr && fused.code == back.code && fused.specials == back.specials,
              "fused shader inherits the back half but clears user data");
        CHECK(scratch[0].value == 0x3456789au && scratch[1].value == 0xabcdef12u,
              "GS fusion patches the front program address into ES registers");
        CHECK(scratch[2].value == 0x11111111u && scratch[3].value == 0x22222222u &&
              scratch[4].value == 0xccccccccu,
              "GS fusion replaces both checksums without disturbing other back registers");
        CHECK(back_regs[0].value == 0 && back_regs[2].value == 0xaaaaaaaau,
              "GS fusion leaves the source back-half register array unchanged");
        CHECK(prosper_agc_fused_back_header_for_front(
                  reinterpret_cast<uint64_t>(front.code)) == &back,
              "GS fusion publishes the back-half body for the bound front address");

        ShaderRegister hs_regs[] = {
            {SPI_SHADER_PGM_LO_LS, 0},
            {SPI_SHADER_PGM_LO_LS + 1u, 0x11223300u},
        };
        Shader hs_front{};
        hs_front.type = 5; // HS front
        hs_front.code = reinterpret_cast<const void*>(0x445566778800ull);
        Shader hs_back{};
        hs_back.type = 7; // HS back
        hs_back.sh_registers = hs_regs;
        hs_back.num_sh_registers = 2;
        Shader hs_fused{};
        fused_rc = fuse_halves_old((uint64_t)(uintptr_t)&hs_fused,
                                   (uint64_t)(uintptr_t)&hs_front,
                                   (uint64_t)(uintptr_t)&hs_back, 0, 0, 0);
        CHECK(fused_rc == 0 && hs_fused.type == 3 && hs_fused.sh_registers == hs_regs &&
              hs_regs[0].value == 0x55667788u && hs_regs[1].value == 0x11223344u,
              "older fusion alias supports HS halves and patches LS registers in place");

        Shader wrong{};
        wrong.type = 1;
        requirements = {0xdeadbeefull, 0xdeadbeefull};
        fused_rc = fused_size((uint64_t)(uintptr_t)&requirements,
                              (uint64_t)(uintptr_t)&front,
                              (uint64_t)(uintptr_t)&wrong, 0, 0, 0);
        CHECK(fused_rc == 0x8a6c0008ull && requirements.size == 0xdeadbeefull,
              "mismatched shader halves are rejected without writing requirements");
        fused_rc = fuse_halves(0, (uint64_t)(uintptr_t)&front,
                               (uint64_t)(uintptr_t)&back, 0, 0, 0);
        CHECK(fused_rc == kAgcErrInvalidArg, "null fused-shader output is rejected");
    }

    // Cobra's eboot wrapper allocates exactly 32 ShaderRegisters, calls dbOlWdppb4o with
    // (out, producer type 2, pixel type 1), then advertises all 32 to PatchAddRegisters. A success
    // stub left most of the stack array unwritten, so coincidental offsets became real Cx writes.
    if (create_interp && create_interp_320) {
        Shader producer{}; producer.type = 2;
        Shader pixel{}; pixel.type = 1;
        ShaderRegister regs[32];
        for (auto& reg : regs) reg = {0xDEADBEEFu, 0xA5A5A5A5u};
        uint64_t interp_rc = create_interp((uint64_t)(uintptr_t)regs,
                                           (uint64_t)(uintptr_t)&producer,
                                           (uint64_t)(uintptr_t)&pixel, 0, 0, 0);
        bool complete = interp_rc == 0;
        for (uint32_t i = 0; i < 32; ++i)
            complete = complete && regs[i].offset == prosper::agc::Pm4::SPI_PS_INPUT_CNTL_0 + i &&
                       regs[i].value == 0;
        CHECK(complete, "Cobra interpolant builder initializes all 32 advertised Cx records");
    }

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

    // Sonic Origins recycles shader-header allocations. A new blob at an address seen by an earlier
    // CreateShader call once kept its raw 0x70/0x78 self-relative register pointers because relocation
    // was guarded by header address alone, then crashed while patching the PGM pair. Drive two distinct
    // blob lifetimes through the same storage and require both to relocate and patch normally.
    alignas(Shader) unsigned char recycled_storage[256]{};
    auto* recycled = reinterpret_cast<Shader*>(recycled_storage);
    auto* recycled_regs = reinterpret_cast<ShaderRegister*>(recycled_storage + 0x80);
    const uintptr_t raw_sh_offset = reinterpret_cast<uintptr_t>(recycled_regs) -
                                    reinterpret_cast<uintptr_t>(&recycled->sh_registers);
    auto prepare_recycled = [&] {
        *recycled = {};
        recycled->file_header = 0x34333231u;
        recycled->version = 0x18u;
        recycled->header_size = sizeof(recycled_storage);
        recycled->shader_size = 64;
        recycled->type = 2;
        recycled->num_sh_registers = 2;
        recycled->sh_registers = reinterpret_cast<ShaderRegister*>(raw_sh_offset);
        recycled_regs[0] = {prosper::agc::Pm4::SPI_SHADER_PGM_LO_ES, 0};
        recycled_regs[1] = {prosper::agc::Pm4::SPI_SHADER_PGM_HI_ES, 0};
    };
    prepare_recycled();
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(recycled),
                       0x1234567800ull, 0, 0, 0);
    CHECK(rc == 0 && dst == recycled && recycled->sh_registers == recycled_regs,
          "first shader lifetime relocates its self-relative register pointer");
    CHECK(recycled_regs[0].value == 0x12345678u,
          "first shader lifetime patches the program base");
    prepare_recycled();
    dst = nullptr;
    rc = create_shader(reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(recycled),
                       0x2345678900ull, 0, 0, 0);
    CHECK(rc == 0 && dst == recycled && recycled->sh_registers == recycled_regs,
          "recycled shader allocation relocates the new header again");
    CHECK(recycled_regs[0].value == 0x23456789u,
          "recycled shader allocation patches the new program base");

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

    // Hundreds of draw snapshots dominate Evergate's opening transition. Prime the exact serial
    // result, then realize the same immutable snapshots through the live parallel/shared-word path.
    // The worker completion order is intentionally unconstrained; compaction must still return the
    // original PM4 order and every semantic field must match the serial oracle.
    alignas(256) static const uint32_t parallel_vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
        0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
        0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    alignas(256) static const uint32_t parallel_ps[] = {
        0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
        0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    Shader parallel_vs_header{};
    parallel_vs_header.file_header = 0x34333231u;
    parallel_vs_header.version = 0x18u;
    parallel_vs_header.shader_size = sizeof(parallel_vs);
    parallel_vs_header.type = 2;
    dst = nullptr;
    const uint64_t parallel_vs_rc = create_shader(
        reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&parallel_vs_header),
        reinterpret_cast<uint64_t>(parallel_vs), 0, 0, 0);
    Shader parallel_ps_header{};
    parallel_ps_header.file_header = 0x34333231u;
    parallel_ps_header.version = 0x18u;
    parallel_ps_header.shader_size = sizeof(parallel_ps);
    parallel_ps_header.type = 1;
    dst = nullptr;
    const uint64_t parallel_ps_rc = create_shader(
        reinterpret_cast<uint64_t>(&dst), reinterpret_cast<uint64_t>(&parallel_ps_header),
        reinterpret_cast<uint64_t>(parallel_ps), 0, 0, 0);
    CHECK(parallel_vs_rc == 0 && parallel_ps_rc == 0,
          "parallel-realization shaders enter the AGC registry");

    namespace P = prosper::agc::Pm4;
    prosper::gpu::GpuState parallel_state;
    auto set_pgm = [&](uint32_t lo, uint32_t hi, const void* code) {
        const uint64_t address = reinterpret_cast<uint64_t>(code);
        parallel_state.sh[lo] = static_cast<uint32_t>(address >> 8);
        parallel_state.sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
    };
    set_pgm(P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, parallel_vs);
    set_pgm(P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, parallel_ps);
    parallel_state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    parallel_state.cx[P::CB_TARGET_MASK] = 0xf;
    parallel_state.draws.resize(128);
    for (size_t i = 0; i < parallel_state.draws.size(); ++i) {
        parallel_state.draws[i].index_count = 3;
        parallel_state.draws[i].command_order = 1000 + i * 3;
    }
    prosper::gpu::clear_shader_recompile_cache();
    prosper::gpu::clear_shader_decode_cache();
    const auto serial_draws = prosper::gpu::realize_gpustate_draws(
        parallel_state, 0x10000, 1.0f, 1.0f, nullptr, false, false);
    const auto serial_decode = prosper::gpu::shader_decode_cache_stats();
    // Cold-start the address-local decode and analysis layers so parallel workers race their two
    // inserts while retaining warm compiled shaders. Entry bytes must remain exact under both races.
    prosper::gpu::clear_shader_decode_cache();
    prosper::gpu::clear_shader_analysis_cache();
    const auto parallel_before = prosper::gpu::parallel_draw_realization_stats();
    const auto parallel_draws = prosper::gpu::realize_gpustate_draws(
        parallel_state, 0x10000, 1.0f, 1.0f, nullptr, true, true);
    const auto parallel_after = prosper::gpu::parallel_draw_realization_stats();
    const auto parallel_decode = prosper::gpu::shader_decode_cache_stats();
    const auto parallel_analysis = prosper::gpu::shader_analysis_cache_stats();
    bool equivalent = serial_draws.size() == parallel_state.draws.size() &&
                      parallel_draws.size() == serial_draws.size();
    for (size_t i = 0; equivalent && i < serial_draws.size(); ++i) {
        const auto& serial = serial_draws[i];
        const auto& parallel = parallel_draws[i];
        equivalent = serial.draw_index == i && parallel.draw_index == i &&
            serial.command_order == parallel.command_order &&
            serial.vertex_count == parallel.vertex_count &&
            serial.instance_count == parallel.instance_count &&
            serial.indices == parallel.indices && serial.vs_words() == parallel.vs_words() &&
            serial.gs_words() == parallel.gs_words() && serial.fs_words() == parallel.fs_words() &&
            serial.vs_identity == parallel.vs_identity &&
            serial.fs_identity == parallel.fs_identity &&
            serial.ps.topology == parallel.ps.topology &&
            serial.ps.color_write_mask == parallel.ps.color_write_mask &&
            serial.ps.blend_enable == parallel.ps.blend_enable &&
            serial.color0_base == parallel.color0_base &&
            serial.color0_width == parallel.color0_width &&
            serial.color0_height == parallel.color0_height &&
            parallel.vs_shared && parallel.fs_shared &&
            !parallel.vs_shared->empty() && !parallel.fs_shared->empty() &&
            parallel.vs.empty() && parallel.fs.empty();
    }
    CHECK(equivalent,
          "parallel shared-word realization is field- and order-equivalent to serial realization");
    CHECK(parallel_after.batches == parallel_before.batches + 1 &&
              parallel_after.semantic_draws ==
                  parallel_before.semantic_draws + parallel_state.draws.size() &&
              parallel_after.worker_threads > parallel_before.worker_threads,
          "dense draw realization records one multi-threaded batch");
    CHECK(parallel_analysis.entries == 2 &&
              parallel_analysis.bytes == sizeof(parallel_vs) + sizeof(parallel_ps),
          "parallel cold analysis keeps exact entry and byte accounting");
    CHECK(serial_decode.entries == 2 && parallel_decode.entries == serial_decode.entries &&
              parallel_decode.bytes == serial_decode.bytes,
          "parallel cold decode keeps exact entry and byte accounting");

    // Public realization calls can arrive from independent submit threads. The process-lifetime
    // pool must serialize whole batches instead of replacing the active generation mid-flight.
    std::atomic<size_t> concurrent_ready{0};
    std::atomic<bool> concurrent_start{false};
    std::vector<prosper::gpu::DrawItem> concurrent_a, concurrent_b;
    auto concurrent_realize = [&](std::vector<prosper::gpu::DrawItem>& output) {
        concurrent_ready.fetch_add(1, std::memory_order_release);
        while (!concurrent_start.load(std::memory_order_acquire)) std::this_thread::yield();
        output = prosper::gpu::realize_gpustate_draws(
            parallel_state, 0x10000, 1.0f, 1.0f, nullptr, true, true);
    };
    std::thread concurrent_thread_a(concurrent_realize, std::ref(concurrent_a));
    std::thread concurrent_thread_b(concurrent_realize, std::ref(concurrent_b));
    while (concurrent_ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    concurrent_start.store(true, std::memory_order_release);
    concurrent_thread_a.join();
    concurrent_thread_b.join();
    bool concurrent_ok = concurrent_a.size() == parallel_state.draws.size() &&
                         concurrent_b.size() == parallel_state.draws.size();
    for (size_t i = 0; concurrent_ok && i < parallel_state.draws.size(); ++i)
        concurrent_ok = concurrent_a[i].draw_index == i && concurrent_b[i].draw_index == i &&
                        concurrent_a[i].command_order == parallel_state.draws[i].command_order &&
                        concurrent_b[i].command_order == parallel_state.draws[i].command_order;
    CHECK(concurrent_ok, "concurrent callers complete serialized realization batches in draw order");

    // Exercise persistent-worker generation changes repeatedly. This catches an early return or a
    // stale batch/context reference that a single batch cannot expose, while warm shader-cache reuse
    // keeps the stress loop small and deterministic.
    const auto repeated_before = prosper::gpu::parallel_draw_realization_stats();
    bool repeated_ok = true;
    constexpr size_t kRepeatedBatches = 64;
    for (size_t batch = 0; batch < kRepeatedBatches && repeated_ok; ++batch) {
        const auto repeated = prosper::gpu::realize_gpustate_draws(
            parallel_state, 0x10000, 1.0f, 1.0f, nullptr, true, true);
        repeated_ok = repeated.size() == parallel_state.draws.size();
        for (size_t i = 0; repeated_ok && i < repeated.size(); ++i)
            repeated_ok = repeated[i].draw_index == i &&
                          repeated[i].command_order == parallel_state.draws[i].command_order &&
                          repeated[i].vs_shared && repeated[i].fs_shared;
    }
    const auto repeated_after = prosper::gpu::parallel_draw_realization_stats();
    CHECK(repeated_ok &&
              repeated_after.batches == repeated_before.batches + kRepeatedBatches &&
              repeated_after.semantic_draws == repeated_before.semantic_draws +
                  kRepeatedBatches * parallel_state.draws.size(),
          "persistent realization workers survive repeated batch generations");

    // An attempted parallel batch can correctly produce no items. That must not be mistaken for
    // "parallel disabled" and replay all draws serially. Each no-effect draw reaches both already-warm
    // shader cache entries once; a serial retry would double this exact hit count.
    prosper::gpu::GpuState filtered_state = parallel_state;
    filtered_state.cx[P::CB_TARGET_MASK] = 0;
    const auto filtered_parallel_before = prosper::gpu::parallel_draw_realization_stats();
    const auto filtered_shader_before = prosper::gpu::shader_recompile_cache_stats();
    const auto filtered_draws = prosper::gpu::realize_gpustate_draws(
        filtered_state, 0x10000, 1.0f, 1.0f, nullptr, true, true);
    const auto filtered_shader_after = prosper::gpu::shader_recompile_cache_stats();
    const auto filtered_parallel_after = prosper::gpu::parallel_draw_realization_stats();
    CHECK(filtered_draws.empty() &&
              filtered_parallel_after.batches == filtered_parallel_before.batches + 1 &&
              filtered_shader_after.hits ==
                  filtered_shader_before.hits + filtered_state.draws.size() * 2,
          "all-filtered parallel batch is not retried through the serial path");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
