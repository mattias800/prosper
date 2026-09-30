// Portable scalar Wave64 reductions must cross host subgroups, but never guest waves.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "[FAIL] line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)
constexpr uint32_t kLanes = 256;
constexpr uint32_t kStride = prosper::gpu::kNggExportProbeWords;

// v0/v1 are raw per-lane comparison operands; s3 is a per-wave iteration count. EXEC
// is narrowed to the saved comparison mask, so scalar results must reach inactive lanes
// too. Four waves execute different numbers of dispatcher iterations and end separately.
std::vector<uint32_t> guest() {
    return {
        0xbe800303u,              // s_mov_b32 s0,s3
        0x7d8402f9u, 0x06068600u, // v_cmp_eq_u32_sdwa s[6:7],v0,v1
        0xbe882406u,              // s_and_saveexec_b64 s[8:9],s[6:7]
        0xbf068000u,              // s_cmp_eq_u32 s0,0: initially false
        0xbe841006u,              // s_bcnt1_i32_b64 s4,s[6:7] (loop header)
        0xbe851406u,              // s_ff1_i32_b64 s5,s[6:7]; preserves BCNT SCC
        0xbe8a03fdu,              // s_mov_b32 s10,scc
        0xbe8b107eu,              // s_bcnt1_i32_b64 s11,exec
        0xbe8c147eu,              // s_ff1_i32_b64 s12,exec
        0x80808100u,              // s_sub_u32 s0,s0,1
        0xbf078000u,              // s_cmp_lg_u32 s0,0
        0xbf85fff8u,              // s_cbranch_scc1 loop header (pc 5)
        0xbefe0408u,              // s_mov_b64 exec,s[8:9]
        0x7e040204u,              // v_mov_b32 v2,s4
        0x7e060205u,              // v_mov_b32 v3,s5
        0x7e08020au,              // v_mov_b32 v4,s10
        0x7e0a020bu,              // v_mov_b32 v5,s11
        0x7e0c020cu,              // v_mov_b32 v6,s12
        0xf80000cfu, 0x05040302u, // EXP POS0, count/first/SCC/EXEC count
        0xf8000201u, 0x00000006u, // EXP PARAM0.x, EXEC first
        0xbf810000u,
    };
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code) {
    return prosper::gpu::recompile_ngg_exports_for_test(
        code.data(), code.size(), 10, 0, nullptr, 4, 0, {}, true, true);
}

std::vector<float> inputs(uint32_t mode) {
    std::vector<float> input(kLanes * 10u, 0);
    constexpr std::array<uint32_t, 4> selected = {0, 31, 32, 63};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t wave = lane / 64u, local = lane % 64u;
        input[lane * 10u] = std::bit_cast<float>(local);
        // singleton, full, empty, and distinct prefix populations 1/32/33/64
        const uint32_t rhs = mode == 0 ? selected[wave] :
            mode == 1 ? local : mode == 2 ? 64u :
            local < selected[wave] + 1u ? local : 64u;
        input[lane * 10u + 1u] = std::bit_cast<float>(rhs);
        input[lane * 10u + 9u] = std::bit_cast<float>(wave + 1u);
    }
    return input;
}

uint32_t mismatches(const std::vector<float>& output, uint32_t mode) {
    if (output.size() != kLanes * kStride) return kLanes;
    constexpr std::array<uint32_t, 4> selected = {0, 31, 32, 63};
    uint32_t bad = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t wave = lane / 64u;
        const uint32_t count = mode == 0 ? 1u : mode == 1 ? 64u :
            mode == 2 ? 0u : selected[wave] + 1u;
        const uint32_t first = mode == 0 ? selected[wave] :
            mode == 2 ? UINT32_MAX : 0u;
        const auto word = [&](uint32_t slot) {
            return std::bit_cast<uint32_t>(output[lane * kStride + slot]);
        };
        if (word(1) != count || word(2) != first || word(3) != (count != 0) ||
            word(4) != count || word(9) != first) {
            if (bad < 4) std::fprintf(stderr,
                "mode=%u lane=%u got=%u/%u/%u/%u/%u expected=%u/%u/%u\n",
                mode, lane, word(1), word(2), word(3), word(4), word(9),
                count, first, count != 0);
            ++bad;
        }
    }
    return bad;
}
}

int main(int argc, char** argv) {
    const auto code = guest();
    const auto module = compile(code);
    CHECK(!module.empty());
    if (argc == 3 && std::strcmp(argv[1], "--dump-spv") == 0) {
        auto* file = std::fopen(argv[2], "wb");
        CHECK(file != nullptr);
        if (file) {
            CHECK(std::fwrite(module.data(), sizeof(uint32_t), module.size(), file) == module.size());
            CHECK(std::fclose(file) == 0);
        }
        return failures ? 1 : 0;
    }
    uint32_t switches = 0, barriers = 0, subgroup_ops = 0;
    for (size_t word = 5; word < module.size();) {
        const uint32_t length = module[word] >> 16, op = module[word] & 0xffffu;
        CHECK(length && word + length <= module.size());
        if (!length || word + length > module.size()) break;
        switches += op == 251u;
        barriers += op == 224u;
        subgroup_ops += op >= 333u && op <= 366u;
        word += length;
    }
    CHECK(switches > 0 && barriers > 0 && subgroup_ops == 0);
    // Exercise the production compute entry, not only the shell used for raw lane readback.
    prosper::gpu::ComputeShaderConfig config;
    config.wave_size = 64;
    config.local_x = kLanes;
    const uint32_t exec_count[] = {0xbe84107eu, 0x7e020204u, 0xbf810000u};
    CHECK(!prosper::gpu::recompile_compute(exec_count, std::size(exec_count), nullptr, config).empty());
    const uint32_t exec_first[] = {0xbe84147eu, 0x7e040204u, 0xbf810000u};
    CHECK(!prosper::gpu::recompile_compute(exec_first, std::size(exec_first), nullptr, config).empty());
    config.native_subgroup_size = 32;
    CHECK(!prosper::gpu::recompile_compute(exec_first, std::size(exec_first), nullptr, config).empty());
    config.wave_size = 32;
    CHECK(prosper::gpu::recompile_compute(exec_first, std::size(exec_first), nullptr, config).empty());
    config.wave_size = 64;
    config.native_subgroup_size = 0;
    config.local_x = 16;
    CHECK(prosper::gpu::recompile_compute(exec_count, std::size(exec_count), nullptr, config).empty());
    // A padded 64-invocation host workgroup is not a complete 64-lane guest wave. The
    // complement contains bits belonging to absent guest lanes, not just the launched prefix.
    const uint32_t partial_complement[] = {
        0xbe86087eu, // s_not_b64 s[6:7],exec
        0xbe841006u, // s_bcnt1_i32_b64 s4,s[6:7]
        0xbf8a0000u, // s_barrier
        0x7e020204u, // v_mov_b32 v1,s4
        0xbf810000u,
    };
    config.local_x = 64;
    config.exact_thread_extent = true;
    config.threads_x = 16;
    config.threads_y = config.threads_z = 1;
    CHECK(prosper::gpu::recompile_compute(partial_complement,
        std::size(partial_complement), nullptr, config).empty());
    config.threads_x = 64;
    CHECK(!prosper::gpu::recompile_compute(partial_complement,
        std::size(partial_complement), nullptr, config).empty());

    auto overwritten = code;
    overwritten[8] = 0xbe87107eu; // count overwrites saved mask's HIGH word
    CHECK(compile(overwritten).empty());
    overwritten[8] = 0xbe8d107eu; // neighbouring non-overlapping scalar destination
    CHECK(!compile(overwritten).empty());

    // An ordinary uint pair must stay numerical, rather than reading a false mask placeholder.
    const std::vector<uint32_t> numerical = {
        0xbe8603ffu, 0x80000001u, // s_mov_b32 s6, literal: two low-word bits
        0xbe870381u,             // s_mov_b32 s7,1: one high-word bit
        0xbf068080u,             // s_cmp_eq_u32 0,0 -> stale SCC=true
        0xbe841006u,             // s_bcnt1_i32_b64 s4,s[6:7] -> 3
        0x7e040204u,             // v_mov_b32 v2,s4
        0x7e0602fdu,             // v_mov_b32 v3,scc
        0xf80000c3u, 0x00000302u,// EXP POS0.xy
        0xbf810000u,
    };
    const auto numerical_module = compile(numerical);
    CHECK(!numerical_module.empty());
    if (!numerical_module.empty()) {
        const auto output = prosper::test::run_compute(numerical_module, inputs(0),
            kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
        CHECK(output.size() == kLanes * kStride);
        for (uint32_t lane = 0; lane < kLanes && output.size() == kLanes * kStride; ++lane) {
            CHECK(std::bit_cast<uint32_t>(output[lane * kStride + 1u]) == 3u);
            CHECK(std::bit_cast<uint32_t>(output[lane * kStride + 2u]) == 1u);
        }
    }
    if (!module.empty()) {
        for (uint32_t mode = 0; mode < 4; ++mode) {
            const auto output = prosper::test::run_compute(
                module, inputs(mode), kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
            CHECK(mismatches(output, mode) == 0);
            auto wrong = output;
            if (!wrong.empty()) wrong[32u * kStride + 1u] = std::bit_cast<float>(999u);
            CHECK(mismatches(wrong, mode) > 0);
        }
    }
    std::fprintf(stderr, "portable Wave64 reductions: %d failures\n", failures);
    return failures ? 1 : 0;
}
