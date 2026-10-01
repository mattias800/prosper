// Architectural read-first-lane over complete Wave64s on narrower host subgroups.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "[FAIL] line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)
constexpr uint32_t kLanes = 256, kStride = prosper::gpu::kNggExportProbeWords;
constexpr std::array<uint32_t, 4> kFirst = {0, 31, 32, 63};
uint32_t source(uint32_t wave, uint32_t lane, uint32_t vgpr) {
    return (vgpr == 2 ? 0xa5000000u : vgpr == 3 ? 0x5b000000u : 0x7d000000u) |
        (wave << 16) | lane;
}
std::vector<uint32_t> guest(bool scc, bool refresh_source = false) {
    std::vector<uint32_t> code = {
        0xbe800303u,              // s0=s3: each wave has 1..4 iterations
        0x7d8402f9u, 0x06068600u, // SDWA v_cmp_eq_u32 s[6:7],v0,v1
        0xbe882406u,              // s_and_saveexec_b64 s[8:9],s[6:7]
        scc ? 0xbf068080u : 0xbf068180u, // loop: s_cmp_eq_u32 0,0 or 0,1
        0x7e080502u,              // v_readfirstlane_b32 s4,v2
        0xbe8a03fdu,              // s10=SCC (readfirstlane must preserve it)
        0xbf068203u,              // s_cmp_eq_u32 s3,2
        0xbf840002u,              // s_cbranch_scc0 pc11
        0x7e0a0503u,              // v_readfirstlane_b32 s5,v3 (only wave 1)
        0xbf820001u,              // s_branch pc12
        0x7e0a0504u,              // v_readfirstlane_b32 s5,v4 (other waves)
        0x80808100u,              // s_sub_u32 s0,s0,1
        0xbf078000u,              // s_cmp_lg_u32 s0,0
        0xbf85fff5u,              // s_cbranch_scc1 pc4
        0xbefe0408u,              // restore full EXEC
        0x7e0a0204u,              // v5=s4
        0x7e0c0205u,              // v6=s5
        0x7e0e020au,              // v7=s10
        0xbf0d8504u,              // s_bitcmp1_b32 s4,5
        0xbe8b0380u,              // s11=0
        0xbf840001u,              // s_cbranch_scc0 skips the wave-uniform write
        0xbe8b0381u,              // s11=1
        0x7e10020bu,              // v8=s11
        0xf80000c7u, 0x00070605u,// EXP POS0.xyz
        0xf8000201u, 0x00000008u,// EXP PARAM0.x
        0xbf810000u,
    };
    if (refresh_source) {
        // Change each source on every EXEC-active iteration. Empty EXEC changes nothing.
        code.insert(code.begin() + 5, {0x4a040481u, 0x4a060681u, 0x4a080881u});
        code[17] = 0xbf85fff2u; // backedge still targets the SCC compare at pc4
    }
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool native = false) {
    return prosper::gpu::recompile_ngg_exports_for_test(
        code.data(), code.size(), 10, 0, nullptr, 4, 0,
        {prosper::gpu::RecompileDiagnosticStage::Vertex, 0x3998u}, true, true, native);
}
void check_refusal(const std::vector<uint32_t>& code, bool native, const char* reason) {
    prosper::gpu::TerminalRejectCapture capture;
    CHECK(compile(code, native).empty());
    const auto records = capture.take();
    CHECK(std::any_of(records.begin(), records.end(), [&](const auto& record) {
        return record.second.find(reason) != std::string::npos;
    }));
}
uint32_t opcode_count(const std::vector<uint32_t>& module, uint32_t opcode) {
    uint32_t count = 0;
    for (size_t i = 5; i < module.size();) {
        const uint32_t length = module[i] >> 16;
        CHECK(length && i + length <= module.size());
        if (!length || i + length > module.size()) break;
        count += (module[i] & 0xffffu) == opcode;
        i += length;
    }
    return count;
}
std::vector<float> inputs(uint32_t mode) {
    std::vector<float> input(kLanes * 10u, 0);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t wave = lane / 64u, local = lane % 64u;
        const uint32_t rhs = mode == 0 ? kFirst[wave] : mode == 1 ? local :
            mode == 2 ? 64u : local >= kFirst[wave] ? local : 64u;
        input[lane * 10u] = std::bit_cast<float>(local);
        input[lane * 10u + 1u] = std::bit_cast<float>(rhs);
        for (uint32_t reg = 2; reg <= 4; ++reg)
            input[lane * 10u + reg] = std::bit_cast<float>(source(wave, local, reg));
        input[lane * 10u + 9u] = std::bit_cast<float>(wave + 1u);
    }
    return input;
}
uint32_t mismatches(const std::vector<float>& output, uint32_t mode, bool scc,
                    bool refresh_source = false) {
    if (output.size() != kLanes * kStride) return kLanes;
    uint32_t bad = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t wave = lane / 64u;
        const uint32_t first = mode == 0 || mode == 3 ? kFirst[wave] : 0u;
        const auto word = [&](uint32_t slot) {
            return std::bit_cast<uint32_t>(output[lane * kStride + slot]);
        };
        const uint32_t delta = refresh_source && mode != 2 ? wave + 1u : 0u;
        const uint32_t expected2 = source(wave, first, 2) + delta;
        const uint32_t expected5 = source(wave, first, wave == 1 ? 3u : 4u) + delta;
        if (word(1) != expected2 || word(2) != expected5 ||
            word(3) != static_cast<uint32_t>(scc) || word(9) != ((expected2 >> 5) & 1u)) {
            if (bad < 3) std::fprintf(stderr,
                "mode=%u scc=%u lane=%u got=%08x/%08x/%u/%u first=%u\n",
                mode, scc, lane, word(1), word(2), word(3), word(9), first);
            ++bad;
        }
    }
    return bad;
}
}

int main(int argc, char** argv) {
    const auto module = compile(guest(false));
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
    CHECK(opcode_count(module, 251u) > 0 && opcode_count(module, 224u) > 0);
    for (uint32_t op = 333; op <= 366; ++op) CHECK(opcode_count(module, op) == 0);
    for (uint32_t scc = 0; scc < 2; ++scc) {
        const auto code = compile(guest(scc != 0));
        CHECK(!code.empty());
        if (code.empty()) continue;
        for (uint32_t mode = 0; mode < 4; ++mode) {
            const auto output = prosper::test::run_compute(
                code, inputs(mode), kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
            CHECK(mismatches(output, mode, scc != 0) == 0);
            auto wrong = output;
            if (!wrong.empty()) wrong[32u * kStride + 1u] = std::bit_cast<float>(999u);
            CHECK(mismatches(wrong, mode, scc != 0) > 0);
        }
    }
    // Physical high-word overwrite must end the saved-mask lifetime, not resurrect a Bool variable.
    auto overlap = guest(false);
    overlap.insert(overlap.begin() + 6, {0x7e0e0502u, 0xbe8d1006u});
    // Additional RFL s7,v2 overwrites HIGH half of s[6:7]; BCNT s13 then has no whole mask.
    overlap[16] = 0xbf85fff3u; // retarget the loop backedge after inserting two dwords
    CHECK(compile(overlap).empty());
    overlap[6] = 0x7e1c0502u; // s14: overlaps neither the BCNT source nor saved EXEC
    CHECK(!compile(overlap).empty());

    // Production compute entry, plus paired complete/partial guest-wave boundaries.
    const uint32_t rfl[] = {0x7e080500u, 0x7e020204u, 0xbf810000u};
    prosper::gpu::ComputeShaderConfig config;
    config.local_x = 64;
    auto production = prosper::gpu::recompile_compute(rfl, std::size(rfl), nullptr, config);
    CHECK(!production.empty() && opcode_count(production, 224u) > 0);
    config.native_subgroup_size = 32;
    CHECK(!prosper::gpu::recompile_compute(rfl, std::size(rfl), nullptr, config).empty());
    config.native_subgroup_size = 0;
    config.local_x = 16;
    CHECK(prosper::gpu::recompile_compute(rfl, std::size(rfl), nullptr, config).empty());
    config.local_x = 64;
    config.exact_thread_extent = true;
    config.threads_x = 16; config.threads_y = config.threads_z = 1;
    CHECK(prosper::gpu::recompile_compute(rfl, std::size(rfl), nullptr, config).empty());
    config.threads_x = 64;
    CHECK(!prosper::gpu::recompile_compute(rfl, std::size(rfl), nullptr, config).empty());

    // Architectural neighbour of the legacy T20 scalar-bit test: an explicit complete-wave
    // test-shell launch broadcasts lane zero's bit to every lane, not each lane's own bit.
    const uint32_t bit_test[] = {
        0x7e020f00u, 0x7e080501u, 0xbf0d8104u,
        0x850580ffu, 100u, 0x7e000c05u, 0xbf810000u,
    };
    const auto bit_module = prosper::gpu::recompile_valu(bit_test, std::size(bit_test), 1, 0,
        nullptr, 0, prosper::gpu::kDefaultComputePgmRsrc1, true, 64, 64);
    CHECK(!bit_module.empty() && opcode_count(bit_module, 224u) > 0);
    if (!bit_module.empty()) {
        for (uint32_t first : {2u, 4u}) {
            std::vector<float> input(64);
            for (uint32_t lane = 0; lane < 64; ++lane) input[lane] = static_cast<float>(lane + first);
            const auto output = prosper::test::run_compute(bit_module, input, 64, 64);
            CHECK(output.size() == 64);
            for (float word : output) CHECK(word == (first == 2 ? 100.0f : 0.0f));
        }
    }

    // A four-wave launch's s3 may differ by wave. A guest barrier cannot sit behind its terminal
    // guard in EITHER mode. EXP is present, so this is not merely the shell's !saw_export refusal.
    const std::vector<uint32_t> guarded = {
        0xbf068003u, 0xbf840003u, 0xbf8a0000u,
        0xf80000c1u, 0x00000002u, 0xbf810000u,
    };
    check_refusal(guarded, false, "per-wave-terminal-guard");
    check_refusal(guarded, true, "per-wave-terminal-guard");
    const std::vector<uint32_t> unguarded = {
        0xbf8a0000u, 0xf80000c1u, 0x00000002u, 0xbf810000u,
    };
    CHECK(!compile(unguarded).empty());
    CHECK(!compile(unguarded, true).empty());
    auto one_wave_guard = guarded;
    one_wave_guard[4] = 0u; // EXP v0 exists in the smaller probe's input/vertex shell
    CHECK(!prosper::gpu::recompile_ngg_exports_for_test(one_wave_guard.data(),
        one_wave_guard.size(), 2, 0, nullptr, 4, 0, {}, true).empty());

    // A workgroup-uniform terminal guard may have an RFL in its peeled prefix. Its scalar
    // result must remain architectural when consumed after the guest barrier, not per lane.
    for (bool mask_prefix : {false, true}) for (uint32_t skip_body : {0u, 1u}) {
        std::vector<uint32_t> guarded_prefix = mask_prefix
            ? std::vector<uint32_t>{0x7d8402f9u, 0x06068600u, 0xbe841006u}
            : std::vector<uint32_t>{0x7e080500u};
        // Also protect the same peeled-prefix route for the existing saved-mask count service.
        guarded_prefix.insert(guarded_prefix.end(), {
            0xbf068003u, 0xbf840004u, // independently uniform compare s3,0; skip to END
            0xbf8a0000u, 0x7e040204u, // barrier; v2=s4
            0xf80000c1u, 0x00000002u, 0xbf810000u,
        });
        const auto prefix_module = prosper::gpu::recompile_ngg_exports_for_test(
            guarded_prefix.data(), guarded_prefix.size(), 2, 0, nullptr, 4, skip_body, {}, true);
        CHECK(!prefix_module.empty());
        if (prefix_module.empty()) continue;
        std::vector<float> input(64u * 2u, 0);
        for (uint32_t lane = 0; lane < 64u; ++lane) {
            input[lane * 2u] = std::bit_cast<float>(mask_prefix ? lane : source(0, lane, 2));
            input[lane * 2u + 1u] = std::bit_cast<float>(31u);
        }
        const auto output = prosper::test::run_compute(prefix_module, input,
            64, 64u * kStride, {}, {}, nullptr, 64);
        CHECK(output.size() == 64u * kStride);
        uint32_t bad = 0;
        for (uint32_t lane = 0; lane < 64u && output.size() == 64u * kStride; ++lane)
            bad += std::bit_cast<uint32_t>(output[lane * kStride + 1u]) !=
                (skip_body ? 0u : mask_prefix ? 1u : source(0, 0, 2));
        if (bad) std::fprintf(stderr, "guarded prefix mask=%u skip=%u: %u bad lanes\n",
                              mask_prefix, skip_body, bad);
        CHECK(bad == 0);
    }
    for (uint32_t scc = 0; scc < 2; ++scc) {
        const auto refreshed = compile(guest(scc != 0, true));
        CHECK(!refreshed.empty());
        if (refreshed.empty()) continue;
        for (uint32_t mode = 0; mode < 4; ++mode) {
            const auto output = prosper::test::run_compute(refreshed, inputs(mode),
                kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
            CHECK(mismatches(output, mode, scc != 0, true) == 0);
        }
    }

    // The second phase must use the already-sized two-plane array and carry exact scalar state.
    const std::vector<uint32_t> phased = {
        0xbf8a0000u, 0x7e080502u, 0xbf8a0000u, 0x7e0a0204u,
        0xf80000c1u, 0x00000005u, 0xbf810000u,
    };
    const auto phased_module = compile(phased);
    CHECK(!phased_module.empty());
    if (!phased_module.empty()) {
        const auto output = prosper::test::run_compute(phased_module, inputs(0),
            kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
        CHECK(output.size() == kLanes * kStride);
        for (uint32_t lane = 0; lane < kLanes && output.size() == kLanes * kStride; ++lane)
            CHECK(std::bit_cast<uint32_t>(output[lane * kStride + 1u]) == source(lane / 64u, 0, 2));
    }
    // Whole-shader exclusion, not a per-RFL exemption: a plain prefix RFL and the subsequent
    // legacy waterfall both retain the original per-invocation lowering. The preceding guest
    // barrier is unconditional. No real waterfall iteration is claimed by this compatibility arm.
    const std::vector<uint32_t> waterfall = {
        0x7e040500u, 0xbf8a0000u, // plain prefix RFL s2,v0; barrier
        0x7e020f00u, 0xbe86047eu,
        0x7e0402ffu, 0x40a00000u, 0x7e0602ffu, 0x40e00000u,
        0x7e0802ffu, 0x41100000u, 0x7e080501u, 0x7da40204u,
        0xbefc0304u, 0x7e0a8702u, 0x8a867e06u, 0xbefe0406u, 0xbf85fff9u,
        0xbefe04c1u, 0x7e000305u, 0x7e100202u,
        0xf80000c1u, 0x00000000u, 0xf8000201u, 0x00000008u, 0xbf810000u,
    };
    const auto waterfall_module = compile(waterfall);
    CHECK(!waterfall_module.empty());
    if (!waterfall_module.empty()) {
        auto input = inputs(0);
        for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane * 10u] = static_cast<float>(lane % 3u);
        const auto output = prosper::test::run_compute(waterfall_module, input,
            kLanes, kLanes * kStride, {}, {}, nullptr, kLanes);
        CHECK(output.size() == kLanes * kStride);
        for (uint32_t lane = 0; lane < kLanes && output.size() == kLanes * kStride; ++lane) {
            CHECK(output[lane * kStride + 1u] == 5.0f + 2.0f * (lane % 3u));
            CHECK(output[lane * kStride + 9u] == static_cast<float>(lane % 3u));
        }
    }
    std::fprintf(stderr, "portable Wave64 readfirstlane: %d failures\n", failures);
    return failures ? 1 : 0;
}
