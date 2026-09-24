// The guest wave contract needed by merged-NGG producers: a lane exchanges an LDS value with its
// neighbour, reconverges after an EXEC-narrowed prefix count, and preserves the result. The current
// compute shell supplies the workgroup semantics; this does not admit a graphics program by itself.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include "fixtures/compute_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using prosper::gpu::recompile_valu;
using prosper::gpu::recompile_ngg_exports_for_test;

namespace {
constexpr uint32_t kWaveSize = 64;

// Input v0 is a floating-point byte address (lane * 4). All lanes write, exchange with lane^1,
// then only lanes 0..15 count their active EXEC predecessors before the full wave reconverges.
// The VOP2 operand edits below follow the checked-in gfx1030 encodings of the same operations in
// test_rdna2_to_spirv.cpp; the test executes the complete resulting guest stream on Vulkan.
constexpr std::array<uint32_t, 20> kGuest = {
    0x7e000f00u,              // v_cvt_u32_f32 v0,v0
    0x4a0200c0u,              // v_add_nc_u32 v1,64,v0
    0xd8340000u, 0x00000100u, // ds_write_b32 v0,v1
    0xbf8a0000u,              // s_barrier
    0x3a040084u,              // v_xor_b32 v2,4,v0: peer byte address
    0xd8d80000u, 0x03000002u, // ds_read_b32 v3,v2
    0x7e0802c0u,              // v_mov_b32 v4,64
    0x7e0a0280u,              // v_mov_b32 v5,0, including lanes later masked off
    0xbe80047eu,              // s_mov_b64 s[0:1],exec
    0x7da20900u,              // v_cmpx_lt_u32 v0,v4: narrow EXEC
    0xd7650005u, 0x0001007eu, // v_mbcnt_lo_u32_b32 v5,exec_lo,0
    0xd7660005u, 0x00020a7fu, // v_mbcnt_hi_u32_b32 v5,exec_hi,v5
    0xbefe0400u,              // s_mov_b64 exec,s[0:1]
    0x4a0c0b03u,              // v_add_nc_u32 v6,v3,v5
    0x7e000d06u,              // v_cvt_f32_u32 v0,v6
    0xbf810000u,              // s_endpgm
};

std::vector<float> run(const std::array<uint32_t, kGuest.size()>& code,
                       bool* portable_wave = nullptr) {
    const auto module = recompile_valu(code.data(), code.size(), 1, 0);
    if (module.empty()) return {};
    if (portable_wave) {
        uint32_t barriers = 0;
        bool native_subgroup = false;
        for (size_t word = 5; word < module.size();) {
            const uint32_t count = module[word] >> 16;
            if (!count || word + count > module.size()) return {};
            const uint32_t opcode = module[word] & 0xffffu;
            barriers += opcode == 224u; // OpControlBarrier: peer publication and exact-wave scan
            native_subgroup |= opcode >= 333u && opcode <= 366u;
            word += count;
        }
        *portable_wave = barriers >= 2 && !native_subgroup;
    }
    std::vector<float> input(kWaveSize);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        input[lane] = static_cast<float>(lane * 4);
    return prosper::test::run_compute(module, input, kWaveSize, kWaveSize);
}

uint32_t mismatches(const std::vector<float>& actual, bool report = false) {
    if (actual.size() != kWaveSize) return kWaveSize;
    uint32_t bad = 0;
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        const uint32_t peer = lane ^ 1u;
        const float expected = static_cast<float>(64u + peer * 4u +
                                                  (lane < 16u ? lane : 0u));
        if (std::fabs(actual[lane] - expected) > 0.01f) {
            if (report && bad < 4)
                std::fprintf(stderr, "lane=%u got=%g expected=%g\n",
                             lane, actual[lane], expected);
            ++bad;
        }
    }
    return bad;
}

std::array<uint32_t, 28> exported_guest() {
    std::array<uint32_t, 28> code{};
    std::copy_n(kGuest.begin(), kGuest.size() - 1, code.begin());
    constexpr std::array<uint32_t, 9> terminal = {
        0xf8000941u, 0x00000006u, // EXP PRIM, v6
        0xf80000cfu, 0x03020100u, // EXP POS0, v0..v3
        0xf80008d4u, 0x00050000u, // EXP POS1.z, v5 (layer)
        0xf800020fu, 0x00050603u, // EXP PARAM0, v3/v6/v5/v0
        0xbf810000u,              // s_endpgm
    };
    std::copy(terminal.begin(), terminal.end(), code.begin() + kGuest.size() - 1);
    return code;
}

std::vector<float> run_exports(const std::array<uint32_t, 28>& code) {
    const auto module = recompile_ngg_exports_for_test(code.data(), code.size(), 1);
    if (module.empty()) return {};
    std::vector<float> input(kWaveSize);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        input[lane] = static_cast<float>(lane * 4);
    return prosper::test::run_compute(
        module, input, kWaveSize, kWaveSize * prosper::gpu::kNggExportProbeWords);
}

uint32_t export_mismatches(const std::vector<float>& actual, bool report = false) {
    constexpr uint32_t stride = prosper::gpu::kNggExportProbeWords;
    if (actual.size() != kWaveSize * stride) return kWaveSize;
    uint32_t bad = 0;
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        const uint32_t peer = lane ^ 1u;
        const uint32_t prefix = lane < 16u ? lane : 0u;
        const uint32_t result = 64u + peer * 4u + prefix;
        const auto word = [&](uint32_t slot) {
            return std::bit_cast<uint32_t>(actual[lane * stride + slot]);
        };
        const bool right = word(0) == result &&
            actual[lane * stride + 1] == static_cast<float>(result) &&
            word(2) == 64u + lane * 4u && word(3) == peer * 4u &&
            word(4) == 64u + peer * 4u &&
            word(5) == 0 && word(6) == 0 && word(7) == prefix && word(8) == 0 &&
            word(9) == 64u + peer * 4u && word(10) == result &&
            word(11) == prefix && actual[lane * stride + 12] == static_cast<float>(result);
        if (!right) {
            if (report && bad < 4)
                std::fprintf(stderr, "export lane=%u prim=%u pos0.x=%g layer=%u "
                                     "param0.y=%u expected=%u/%u\n",
                             lane, word(0), actual[lane * stride + 1], word(7),
                             word(10), result, prefix);
            ++bad;
        }
    }
    return bad;
}
} // namespace

int main() {
    bool portable_wave = false;
    const uint64_t host_barriers_before =
        prosper::gpu::backend_host_read_barrier_count().load();
    const auto actual = run(kGuest, &portable_wave);
    if (prosper::gpu::backend_host_read_barrier_count().load() - host_barriers_before != 1) {
        std::fprintf(stderr, "compute output did not record its host-read barrier\n");
        return 1;
    }
    const uint32_t positive_bad = mismatches(actual, true);
    if (positive_bad || !portable_wave) {
        std::fprintf(stderr, "guest wave peer-LDS/EXEC-prefix mismatches=%u, size=%zu "
                             "portable=%d\n", positive_bad, actual.size(), portable_wave);
        return 1;
    }

    // Four 64-lane guest waves can belong to one merged-NGG threadgroup. Exercise the shell's
    // shared LDS across wave boundaries before trusting any Kena launch model that needs it.
    // A separate 64-invocation dispatch for each wave cannot pass this arm.
    std::vector<uint32_t> cross_wave_guest = {
        0x7e000f00u,              // v_cvt_u32_f32 v0,v0: byte address = lane * 4
        0x4a0200c0u,              // v_add_nc_u32 v1,64,v0
        0xd8340000u, 0x00000100u, // ds_write_b32 v0,v1
        0xbf8a0000u,              // s_barrier across the full 256-invocation workgroup
        0x3a0400ffu, 0x00000100u, // v_xor_b32 v2,256,v0: read another wave
        0xd8d80000u, 0x03000002u, // ds_read_b32 v3,v2
        0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
        0x7e000d03u,              // v_cvt_f32_u32 v0,v3
        0xbf810000u,
    };
    const auto cross_wave_output = [&](const std::vector<uint32_t>& code) {
        const auto module = recompile_valu(
            code.data(), code.size(), 1, 0, nullptr, 0,
            prosper::gpu::kDefaultComputePgmRsrc1, false, 256);
        if (module.empty()) return std::vector<float>{};
        std::vector<float> input(256);
        for (uint32_t lane = 0; lane < 256; ++lane)
            input[lane] = static_cast<float>(lane * 4u);
        return prosper::test::run_compute(module, input, 256, 256, {}, {}, nullptr, 256);
    };
    const auto cross_wave_matches = [&](const std::vector<float>& output) {
        if (output.size() != 256) return false;
        for (uint32_t lane = 0; lane < 256; ++lane)
            if (output[lane] != static_cast<float>(64u + ((lane ^ 64u) * 4u)))
                return false;
        return true;
    };
    if (!cross_wave_matches(cross_wave_output(cross_wave_guest))) {
        std::fprintf(stderr, "four guest waves did not exchange LDS across one workgroup\n");
        return 1;
    }
    cross_wave_guest[6] = 4u; // wrong peer: stays in the same wave
    if (cross_wave_matches(cross_wave_output(cross_wave_guest))) {
        std::fprintf(stderr, "cross-wave LDS negative control did not distinguish workgroups\n");
        return 1;
    }

    // Kena's captured merged-NGG program narrows EXEC through a VCC mask and SAVEEXEC, rather than
    // through CMPX. Use the integer VOPC encoding (0x7d82), not the float compare (0x7c02): the
    // latter can flush our small integer bit patterns as subnormal floats and vacate the mask.
    auto saveexec = kGuest;
    saveexec[10] = 0x7d820900u; // v_cmp_lt_u32 vcc,v0,v4
    saveexec[11] = 0xbe80246au; // s_and_saveexec_b64 s[0:1],vcc
    const auto saved_result = run(saveexec);
    if (mismatches(saved_result, true)) {
        std::fprintf(stderr, "VCC/SAVEEXEC did not preserve guest-wave prefix\n");
        return 1;
    }

    auto wrong_peer = kGuest;
    wrong_peer[5] = 0x3a040088u; // exchange with lane^2, not lane^1
    const auto peer_result = run(wrong_peer);
    if (peer_result.size() != kWaveSize || mismatches(peer_result) != kWaveSize) {
        std::fprintf(stderr, "peer-address control did not distinguish all lanes\n");
        return 1;
    }

    auto wrong_mask = kGuest;
    wrong_mask[8] = 0x7e0802a0u; // threshold 32 bytes: 8 live lanes instead of 16
    const auto mask_result = run(wrong_mask);
    if (mask_result.size() != kWaveSize || mismatches(mask_result) != 8) {
        std::fprintf(stderr, "EXEC-mask control did not isolate lanes 8..15\n");
        return 1;
    }

    const auto exports = exported_guest();
    const auto exported = run_exports(exports);
    if (export_mismatches(exported, true)) {
        std::fprintf(stderr, "guest PRIM/POS0/POS1.z/PARAM0 exports did not survive the wave\n");
        return 1;
    }
    // The Kena launch experiment must preserve both raw initial GS offset VGPRs. Test the
    // compile-only input contract with deliberately distinct lane words, then swap the fields:
    // a hardwired vertex ID or an input-stride mistake must fail the per-lane comparison.
    constexpr std::array<uint32_t, 3> packed_export = {
        0xf80000c3u, 0x03020100u, // EXP POS0.xy, initial v0/v1
        0xbf810000u,
    };
    if (!recompile_ngg_exports_for_test(packed_export.data(), packed_export.size(), 0,
                                        0, nullptr, 4, 0, {}, true).empty()) {
        std::fprintf(stderr, "packed-offset probe accepted missing per-lane inputs\n");
        return 1;
    }
    const auto packed_module = recompile_ngg_exports_for_test(
        packed_export.data(), packed_export.size(), 2, 0, nullptr, 4, 0, {}, true);
    if (packed_module.empty()) {
        std::fprintf(stderr, "packed-offset probe did not compile\n");
        return 1;
    }
    std::vector<float> packed_inputs(kWaveSize * 2u);
    for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
        packed_inputs[lane * 2u] = std::bit_cast<float>(0x10000u + lane * 7u);
        packed_inputs[lane * 2u + 1u] = std::bit_cast<float>(0x20000u + lane * 11u);
    }
    const auto packed_readback = [&](const std::vector<float>& input) {
        return prosper::test::run_compute(packed_module, input, kWaveSize,
                                          kWaveSize * prosper::gpu::kNggExportProbeWords);
    };
    const auto packed_matches = [&](const std::vector<float>& output) {
        if (output.size() != kWaveSize * prosper::gpu::kNggExportProbeWords) return false;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            const size_t base = static_cast<size_t>(lane) * prosper::gpu::kNggExportProbeWords;
            if (std::bit_cast<uint32_t>(output[base + 1u]) != 0x10000u + lane * 7u ||
                std::bit_cast<uint32_t>(output[base + 2u]) != 0x20000u + lane * 11u)
                return false;
        }
        return true;
    };
    if (!packed_matches(packed_readback(packed_inputs))) {
        std::fprintf(stderr, "packed-offset probe lost the two per-lane input words\n");
        return 1;
    }
    for (uint32_t lane = 0; lane < kWaveSize; ++lane)
        std::swap(packed_inputs[lane * 2u], packed_inputs[lane * 2u + 1u]);
    if (packed_matches(packed_readback(packed_inputs))) {
        std::fprintf(stderr, "packed-offset swapped-field control did not distinguish inputs\n");
        return 1;
    }
    // A complete four-wave launch supplies the nine initial VGPRs and a wave-uniform s3. The
    // synthetic stream reads the registers before defining them, so this checks the actual module
    // input path rather than only its declared input-buffer size.
    constexpr std::array<uint32_t, 8> full_launch_guest = {
        0x7e120203u,              // v_mov_b32 v9,s3
        0xf80000cfu, 0x03020100u, // EXP POS0, initial v0..v3
        0xf80008dfu, 0x07060504u, // EXP POS1, initial v4..v7
        0xf8000203u, 0x00000908u, // EXP PARAM0.xy, initial v8 and s3 copied to v9
        0xbf810000u,
    };
    if (!recompile_ngg_exports_for_test(full_launch_guest.data(), full_launch_guest.size(),
                                        2, 0, nullptr, 4, 0, {}, true, true).empty()) {
        std::fprintf(stderr, "four-wave probe accepted an incomplete launch record\n");
        return 1;
    }
    const auto full_module = recompile_ngg_exports_for_test(
        full_launch_guest.data(), full_launch_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true);
    if (full_module.empty()) {
        std::fprintf(stderr, "four-wave launch probe did not compile\n");
        return 1;
    }
    constexpr uint32_t kFullLanes = 256;
    std::vector<float> full_inputs(kFullLanes * 10u);
    for (uint32_t lane = 0; lane < kFullLanes; ++lane) {
        const size_t base = static_cast<size_t>(lane) * 10u;
        for (uint32_t reg = 0; reg < 9u; ++reg)
            full_inputs[base + reg] = std::bit_cast<float>(0x3f000000u + lane * 16u + reg);
        full_inputs[base + 9u] =
            std::bit_cast<float>(0x40004040u | ((lane / 64u) << 24u));
    }
    const auto full_readback = [&](const std::vector<float>& input) {
        return prosper::test::run_compute(
            full_module, input, kFullLanes,
            kFullLanes * prosper::gpu::kNggExportProbeWords,
            {}, {}, nullptr, kFullLanes);
    };
    const auto full_matches = [&](const std::vector<float>& output) {
        if (output.size() != kFullLanes * prosper::gpu::kNggExportProbeWords) return false;
        for (uint32_t lane = 0; lane < kFullLanes; ++lane) {
            const size_t base = static_cast<size_t>(lane) * prosper::gpu::kNggExportProbeWords;
            for (uint32_t reg = 0; reg < 8u; ++reg)
                if (std::bit_cast<uint32_t>(output[base + 1u + reg]) !=
                    0x3f000000u + lane * 16u + reg)
                    return false;
            if (std::bit_cast<uint32_t>(output[base + 9u]) !=
                    0x3f000000u + lane * 16u + 8u ||
                std::bit_cast<uint32_t>(output[base + 10u]) !=
                (0x40004040u | ((lane / 64u) << 24u)))
                return false;
        }
        return true;
    };
    if (!full_matches(full_readback(full_inputs))) {
        std::fprintf(stderr, "four-wave launch lost initial VGPRs or per-wave s3\n");
        return 1;
    }
    for (uint32_t reg = 0; reg < 10u; ++reg) {
        auto changed = full_inputs;
        const uint32_t lanes_to_change = reg == 9u ? 64u : 1u; // keep s3 wave-uniform
        for (uint32_t lane = 0; lane < lanes_to_change; ++lane) {
            const size_t slot = static_cast<size_t>(lane) * 10u + reg;
            changed[slot] = std::bit_cast<float>(std::bit_cast<uint32_t>(changed[slot]) ^ 1u);
        }
        if (full_matches(full_readback(changed))) {
            std::fprintf(stderr, "four-wave launch did not distinguish input word %u\n", reg);
            return 1;
        }
    }
    // Kena's main program counts a VOPC-saved s[6:7] mask. One 64-invocation workgroup is one
    // guest wave, independent of the driver's native subgroup width. The export makes the count
    // observable without relying on an image or the game's resource table.
    std::vector<uint32_t> count_guest = {
        0x7e000f00u,              // v_cvt_u32_f32 v0,v0: input is lane * 4
        0x7e0202ffu, 160u,        // v_mov_b32 v1,160: only lane 40 matches
        0x7d8402f9u, 0x06068600u, // v_cmp_eq_u32_sdwa s[6:7],v0,v1
        0xbe841006u,              // s_bcnt1_i32_b64 s4,s[6:7]
        0x7e0c0204u,              // v_mov_b32 v6,s4
        0xf8000941u, 0x00000006u, // EXP PRIM,v6
        0xbf810000u,
    };
    const auto count_output = [&](const std::vector<uint32_t>& code) {
        const auto module = recompile_ngg_exports_for_test(code.data(), code.size(), 1);
        if (module.empty()) return std::vector<float>{};
        std::vector<float> input(kWaveSize);
        for (uint32_t lane = 0; lane < kWaveSize; ++lane)
            input[lane] = static_cast<float>(lane * 4);
        return prosper::test::run_compute(
            module, input, kWaveSize, kWaveSize * prosper::gpu::kNggExportProbeWords);
    };
    const auto count_matches = [&](const std::vector<float>& output, uint32_t value) {
        if (output.size() != kWaveSize * prosper::gpu::kNggExportProbeWords) return false;
        for (uint32_t lane = 0; lane < kWaveSize; ++lane)
            if (std::bit_cast<uint32_t>(output[lane * prosper::gpu::kNggExportProbeWords]) != value) {
                std::fprintf(stderr, "count lane=%u got=%u expected=%u\n", lane,
                             std::bit_cast<uint32_t>(
                                 output[lane * prosper::gpu::kNggExportProbeWords]), value);
                return false;
            }
        return true;
    };
    if (!count_matches(count_output(count_guest), 1u)) {
        std::fprintf(stderr, "portable saved-mask count missed lane 40 or its other consumers\n");
        return 1;
    }
    // Native Wave64 is optional. Where Vulkan 1.3 can require four complete 64-lane subgroups,
    // check that each guest wave gets its own saved-mask count inside one 256-lane LDS workgroup.
    // The ordinary runner cannot substitute its advertised/default subgroup width for this test.
    if (prosper::test::default_compute_required_subgroup_supported(64u, 255u)) {
        std::fprintf(stderr, "native Wave64 gate accepted an incomplete workgroup\n");
        return 1;
    }
    auto native_count_guest = count_guest;
    native_count_guest.insert(native_count_guest.begin() + 5, 0xbf8a0000u); // safe phase barrier
    const auto native_module = recompile_ngg_exports_for_test(
        native_count_guest.data(), native_count_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true);
    if (native_module.empty()) {
        std::fprintf(stderr, "native four-wave saved-mask program did not compile\n");
        return 1;
    }
    // pc 7 copies the saved-mask count into v6 after the barrier. The two sideband words
    // must report the value and an execution hit without moving the 13 export words.
    const auto traced_native_module = recompile_ngg_exports_for_test(
        native_count_guest.data(), native_count_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true, 7u, 6u);
    const auto missed_pc = recompile_ngg_exports_for_test(
        native_count_guest.data(), native_count_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true, 4u, 6u);
    const auto missing_vgpr = recompile_ngg_exports_for_test(
        native_count_guest.data(), native_count_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true, 7u, 250u);
    if (traced_native_module.empty() || !missed_pc.empty() || !missing_vgpr.empty()) {
        std::fprintf(stderr,
                     "NGG trace arms: reached=%zu missed-pc=%zu missing-vgpr=%zu\n",
                     traced_native_module.size(), missed_pc.size(), missing_vgpr.size());
        return 1;
    }
    if (prosper::test::default_compute_required_subgroup_supported(64u, 256u)) {
        std::vector<float> native_inputs(256u * 10u, 0.0f);
        for (uint32_t lane = 0; lane < 256u; ++lane)
            native_inputs[static_cast<size_t>(lane) * 10u] =
                static_cast<float>((lane % 64u) * 4u);
        const auto run_native = [&](const std::vector<float>& input) {
            return prosper::test::run_compute(
                native_module, input, 256u,
                256u * prosper::gpu::kNggExportProbeWords,
                {}, {}, nullptr, 256u, nullptr, nullptr, nullptr, 64u);
        };
        const auto native_matches = [&](const std::vector<float>& output,
                                        uint32_t wave_one_count) {
            if (output.size() != 256u * prosper::gpu::kNggExportProbeWords) return false;
            for (uint32_t lane = 0; lane < 256u; ++lane) {
                const uint32_t expected = lane / 64u == 1u ? wave_one_count : 1u;
                if (std::bit_cast<uint32_t>(
                        output[static_cast<size_t>(lane) * prosper::gpu::kNggExportProbeWords])
                    != expected) return false;
            }
            return true;
        };
        if (!native_matches(run_native(native_inputs), 1u)) {
            std::fprintf(stderr, "native four-wave saved-mask count crossed a wave boundary\n");
            return 1;
        }
        const auto run_trace = [&](const std::vector<float>& input) {
            return prosper::test::run_compute(
                traced_native_module, input, 256u,
                256u * prosper::gpu::kNggTraceProbeWords,
                {}, {}, nullptr, 256u, nullptr, nullptr, nullptr, 64u);
        };
        const auto trace_matches = [&](const std::vector<float>& output,
                                       uint32_t wave_one_count) {
            if (output.size() != 256u * prosper::gpu::kNggTraceProbeWords) return false;
            for (uint32_t lane = 0; lane < 256u; ++lane) {
                const size_t base = static_cast<size_t>(lane) *
                                    prosper::gpu::kNggTraceProbeWords;
                const uint32_t expected = lane / 64u == 1u ? wave_one_count : 1u;
                if (std::bit_cast<uint32_t>(output[base]) != expected ||
                    std::bit_cast<uint32_t>(
                        output[base + prosper::gpu::kNggTraceValueWord]) != expected ||
                    std::bit_cast<uint32_t>(
                        output[base + prosper::gpu::kNggTraceHitWord]) != 1u)
                    return false;
            }
            return true;
        };
        if (!trace_matches(run_trace(native_inputs), 1u)) {
            std::fprintf(stderr, "NGG trace missed the saved-mask milestone\n");
            return 1;
        }
        native_inputs[static_cast<size_t>(64u + 40u) * 10u] = 0.0f;
        if (!native_matches(run_native(native_inputs), 0u)) {
            std::fprintf(stderr, "native Wave64 one-wave mutation was not isolated\n");
            return 1;
        }
        if (!trace_matches(run_trace(native_inputs), 0u)) {
            std::fprintf(stderr, "NGG trace did not isolate the mutated wave\n");
            return 1;
        }
    } else {
        std::fprintf(stderr, "native Wave64 execution skipped: exact full subgroup unavailable\n");
    }
    // Kena's two BOUND_CTRL=1 row shifts reduce per-wave counts into lane 3. The
    // high 0x100 bit used by the portable dispatcher is metadata, not part of the
    // native subgroup shuffle amount; passing it through makes every source invalid.
    constexpr std::array<uint32_t, 8> bounded_dpp_guest = {
        0x7e140300u,              // v_mov_b32 v10,v0
        0xbf8a0000u,              // s_barrier: enter the same phased dispatcher as Kena
        0x4a1614fau, 0xff09110au, // v_add_nc_u32_dpp v11,v10,v10 row_shr:1 bound_ctrl:1
        0x4a1616fau, 0xff09120bu, // v_add_nc_u32_dpp v11,v11,v11 row_shr:2 bound_ctrl:1
        0xf8000941u, 0x0000000bu, // EXP PRIM,v11
    };
    std::vector<uint32_t> bounded_dpp_code(bounded_dpp_guest.begin(),
                                           bounded_dpp_guest.end());
    bounded_dpp_code.push_back(0xbf810000u); // s_endpgm
    const auto bounded_dpp_module = recompile_ngg_exports_for_test(
        bounded_dpp_code.data(), bounded_dpp_code.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true);
    if (bounded_dpp_module.empty()) {
        std::fprintf(stderr, "native bounded DPP reduction did not compile\n");
        return 1;
    }
    if (prosper::test::default_compute_required_subgroup_supported(64u, 256u)) {
        constexpr uint32_t first = 0x200040u, second = 0x20u;
        std::vector<float> input(256u * 10u, 0.0f);
        for (uint32_t wave = 0; wave < 4u; ++wave) {
            input[static_cast<size_t>(wave) * 64u * 10u] = std::bit_cast<float>(first);
            input[(static_cast<size_t>(wave) * 64u + 1u) * 10u] =
                std::bit_cast<float>(second);
        }
        const auto reduce = [&](const std::vector<float>& values) {
            return prosper::test::run_compute(
                bounded_dpp_module, values, 256u,
                256u * prosper::gpu::kNggExportProbeWords,
                {}, {}, nullptr, 256u, nullptr, nullptr, nullptr, 64u);
        };
        const auto reduced = reduce(input);
        if (reduced.size() != 256u * prosper::gpu::kNggExportProbeWords) {
            std::fprintf(stderr, "native bounded DPP dispatch failed\n");
            return 1;
        }
        for (uint32_t wave = 0; wave < 4u; ++wave) {
            const uint32_t lane = wave * 64u + 3u;
            if (std::bit_cast<uint32_t>(
                    reduced[static_cast<size_t>(lane) *
                            prosper::gpu::kNggExportProbeWords]) != first + second) {
                std::fprintf(stderr, "native bounded DPP lost wave %u lane-3 reduction\n",
                             wave);
                return 1;
            }
        }
        input[(64u + 0u) * 10u] = 0.0f;
        const auto changed = reduce(input);
        if (changed.size() != reduced.size() ||
            std::bit_cast<uint32_t>(changed[(64u + 3u) *
                                           prosper::gpu::kNggExportProbeWords]) != second ||
            std::bit_cast<uint32_t>(changed[(0u + 3u) *
                                           prosper::gpu::kNggExportProbeWords]) != first + second) {
            std::fprintf(stderr, "native bounded DPP mutation crossed guest waves\n");
            return 1;
        }
    }
    // A distinct VDST must survive an EXEC-off DPP instruction. Restore EXEC and export it to
    // expose a wrong SRC0 fallback; trace the intervening ordinary ALU so skipped lanes must
    // retain a zero hit even when their old destination value is nonzero.
    constexpr std::array<uint32_t, 12> masked_dpp_guest = {
        0x7e140300u,              // v_mov_b32 v10,v0 (DPP source)
        0x7e160301u,              // v_mov_b32 v11,v1 (old, distinct destination)
        0xbf8a0000u,              // s_barrier: force phased dispatcher
        0xbe80047eu,              // s_mov_b64 s[0:1],exec
        0x7da20900u,              // v_cmpx_lt_u32 v0,v4: only wave lanes 0..31
        0x4a1614fau, 0xff09110au, // Kena bounded ROW_SHR:1 into v11
        0x7e18030bu,              // v_mov_b32 v12,v11: ordinary ALU trace PC 7
        0xbefe0400u,              // s_mov_b64 exec,s[0:1]
        0xf8000941u, 0x0000000bu, // EXP PRIM,v11
        0xbf810000u,              // s_endpgm
    };
    constexpr uint64_t kDppTraceAddress = 0x3135000002ull;
    const auto masked_dpp_module = recompile_ngg_exports_for_test(
        masked_dpp_guest.data(), masked_dpp_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true, 7u, 12u);
    const auto cmpx_entry_trace = recompile_ngg_exports_for_test(
        masked_dpp_guest.data(), masked_dpp_guest.size(), 10, 0, nullptr, 4, 0,
        {}, true, true, true, 4u, 11u);
    const auto special_dpp_trace = recompile_ngg_exports_for_test(
        masked_dpp_guest.data(), masked_dpp_guest.size(), 10, 0, nullptr, 4, 0,
        {prosper::gpu::RecompileDiagnosticStage::Vertex, kDppTraceAddress},
        true, true, true, 5u, 11u);
    if (masked_dpp_module.empty() || cmpx_entry_trace.empty() ||
        !special_dpp_trace.empty() ||
        prosper::gpu::last_terminal_reject_reason(kDppTraceAddress).find(
            "no-ordinary-alu-milestone") == std::string::npos) {
        std::fprintf(stderr, "native bounded DPP trace accepted a special phase or lost ALU PC\n");
        return 1;
    }
    if (prosper::test::default_compute_required_subgroup_supported(64u, 256u)) {
        constexpr uint32_t old_destination = 0x76543210u;
        std::vector<float> input(256u * 10u, 0.0f);
        for (uint32_t lane = 0; lane < 256u; ++lane) {
            const size_t base = static_cast<size_t>(lane) * 10u;
            input[base] = std::bit_cast<float>(lane % 64u);
            input[base + 1u] = std::bit_cast<float>(old_destination);
            input[base + 4u] = std::bit_cast<float>(32u);
        }
        const auto output = prosper::test::run_compute(
            masked_dpp_module, input, 256u,
            256u * prosper::gpu::kNggTraceProbeWords,
            {}, {}, nullptr, 256u, nullptr, nullptr, nullptr, 64u);
        const auto cmpx_output = prosper::test::run_compute(
            cmpx_entry_trace, input, 256u,
            256u * prosper::gpu::kNggTraceProbeWords,
            {}, {}, nullptr, 256u, nullptr, nullptr, nullptr, 64u);
        if (output.size() != 256u * prosper::gpu::kNggTraceProbeWords) {
            std::fprintf(stderr, "native masked DPP dispatch failed\n");
            return 1;
        }
        if (cmpx_output.size() != output.size()) {
            std::fprintf(stderr, "native CMPX entry trace dispatch failed\n");
            return 1;
        }
        for (uint32_t lane = 0; lane < 256u; ++lane) {
            const size_t base = static_cast<size_t>(lane) *
                                prosper::gpu::kNggTraceProbeWords;
            if (lane % 64u >= 32u &&
                std::bit_cast<uint32_t>(output[base]) != old_destination) {
                std::fprintf(stderr, "native masked DPP lost old VDST lane %u\n", lane);
                return 1;
            }
            if (lane % 64u >= 32u &&
                std::bit_cast<uint32_t>(
                    output[base + prosper::gpu::kNggTraceHitWord]) != 0u) {
                std::fprintf(stderr, "native NGG trace marked skipped lane %u\n", lane);
                return 1;
            }
            if (lane % 64u < 32u &&
                std::bit_cast<uint32_t>(
                    output[base + prosper::gpu::kNggTraceHitWord]) != 1u) {
                std::fprintf(stderr, "native masked DPP missed active trace lane %u\n", lane);
                return 1;
            }
            if (std::bit_cast<uint32_t>(
                    cmpx_output[base + prosper::gpu::kNggTraceHitWord]) != 1u) {
                std::fprintf(stderr, "native CMPX trace lost entry-active lane %u\n", lane);
                return 1;
            }
        }
    }
    constexpr std::array<uint32_t, 5> skipped_barrier_guest = {
        0xbf840001u,              // s_cbranch_scc0 skips the barrier
        0xbf8a0000u,              // unsafe for a 256-lane workgroup
        0xf8000941u, 0x00000000u, // EXP PRIM, v0
        0xbf810000u,
    };
    constexpr uint64_t kSkippedBarrierAddress = 0x3135000000ull;
    if (!recompile_ngg_exports_for_test(
            skipped_barrier_guest.data(), skipped_barrier_guest.size(),
            10, 0, nullptr, 4, 0,
            {prosper::gpu::RecompileDiagnosticStage::Vertex, kSkippedBarrierAddress},
            true, true, true).empty() ||
        prosper::gpu::last_terminal_reject_reason(kSkippedBarrierAddress).find(
            "reason=barrier-phase-proof") == std::string::npos) {
        std::fprintf(stderr, "native four-wave probe admitted a skipped barrier\n");
        return 1;
    }
    constexpr std::array<uint32_t, 5> per_wave_guard_guest = {
        0xbe800380u, // s_mov_b32 s0, 0
        0xbf060003u, // s_cmp_eq_u32 s3, s0: s3 may differ between guest waves
        0xbf840001u, // s_cbranch_scc0 -> terminal s_endpgm
        0xbf8a0000u, // s_barrier, reached only by some waves
        0xbf810000u,
    };
    constexpr uint64_t kPerWaveGuardAddress = 0x3135000001ull;
    if (!recompile_ngg_exports_for_test(
            per_wave_guard_guest.data(), per_wave_guard_guest.size(),
            10, 0, nullptr, 4, 0,
            {prosper::gpu::RecompileDiagnosticStage::Vertex, kPerWaveGuardAddress},
            true, true, true).empty() ||
        prosper::gpu::last_terminal_reject_reason(kPerWaveGuardAddress).find(
            "reason=per-wave-terminal-guard") == std::string::npos) {
        std::fprintf(stderr, "native four-wave probe admitted a per-wave barrier guard\n");
        return 1;
    }
    count_guest[2] = 161u; // no lane matches; an any()/constant-one substitute must fail
    if (!count_matches(count_output(count_guest), 0u)) {
        std::fprintf(stderr, "portable saved-mask zero-count control failed\n");
        return 1;
    }
    count_guest[1] = 0x7e020300u; // v_mov_b32 v1,v0 (VGPR sources start at 0x100)
    count_guest.erase(count_guest.begin() + 2);
    count_guest[2] = 0x7d8400f9u; // v_cmp_eq_u32_sdwa s[6:7],v0,v0: all lanes match
    if (!count_matches(count_output(count_guest), 64u)) {
        std::fprintf(stderr, "portable saved-mask full-wave count control failed\n");
        return 1;
    }
    std::vector<uint32_t> row_guest = {
        0x7e000f00u,              // v_cvt_u32_f32 v0,v0
        0x7e140300u,              // v_mov_b32 v10,v0
        0xbf8a0000u,              // barrier forces a separate dispatcher phase
        0x4a1614fau,0xff09110au, // Kena: v_add_nc_u32_dpp v11,v10,v10 row_shr:1 BC1
        0xf8000941u,0x0000000bu, // EXP PRIM,v11
        0xbf810000u,
    };
    const auto row_output = count_output(row_guest);
    uint32_t row_bad = 0;
    if (row_output.size() == kWaveSize * prosper::gpu::kNggExportProbeWords)
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            const uint32_t peer = (lane & 15u) ? lane - 1u : 0u;
            const uint32_t expected = lane * 4u +
                ((lane & 15u) ? peer * 4u : 0u);
            const uint32_t got = std::bit_cast<uint32_t>(
                row_output[lane * prosper::gpu::kNggExportProbeWords]);
            if (got != expected && row_bad < 4u)
                std::fprintf(stderr, "ROW_SHR lane=%u got=%u expected=%u\n",
                             lane, got, expected);
            row_bad += got != expected;
        }
    else row_bad = kWaveSize;
    if (row_bad) {
        std::fprintf(stderr, "portable bounded ROW_SHR:1 mismatches=%u\n", row_bad);
        return 1;
    }
    row_guest[4] = 0xff09120au; // ROW_SHR:2 instead of ROW_SHR:1
    const auto wrong_row = count_output(row_guest);
    uint32_t distinguished = 0;
    if (wrong_row.size() == row_output.size())
        for (uint32_t lane = 0; lane < kWaveSize; ++lane)
            distinguished += wrong_row[lane * prosper::gpu::kNggExportProbeWords] !=
                row_output[lane * prosper::gpu::kNggExportProbeWords];
    if (distinguished < 20u) {
        std::fprintf(stderr, "ROW_SHR amount control distinguished only %u lanes\n", distinguished);
        return 1;
    }
    row_guest[4] = 0xff09000au; // control outside the admitted ROW_SHR family
    if (!recompile_ngg_exports_for_test(row_guest.data(), row_guest.size(), 1).empty()) {
        std::fprintf(stderr, "unsupported DPP control was silently accepted\n");
        return 1;
    }
    // The captured producer ends one CFG phase with scalar data in both physical VCC words,
    // crosses a barrier, then reads VCC_HI as scalar data. This smaller fixture uses a B32 select
    // so the untouched high word is observable at an implicit v_cndmask read. The terminal
    // scalar MUST proof must carry both words; merely retaining a Function-variable load would
    // turn a missing high word into a silently accepted zero mask.
    std::vector<uint32_t> scalar_vcc_phase = {
        0x7e000f00u,              // v_cvt_u32_f32 v0,v0: lane * 4
        0x7e0202c0u,              // v_mov_b32 v1,64
        0x7e140300u,              // v_mov_b32 v10,v0
        0x4a1614fau,0xff09110au, // bounded DPP: forces barrier-phase dispatcher
        0xbe8003c1u,              // s_mov_b32 s0,-1
        0xbeea0380u,              // s_mov_b32 vcc_lo,0
        0xbeeb0380u,              // s_mov_b32 vcc_hi,0
        0xbf8a0000u,              // barrier
        0xbf0000c1u,              // s_cmp_eq_u32 -1,s0: SCC true
        0x856a8000u,              // s_cselect_b32 vcc_lo,s0,0
        0x02060300u,              // v_cndmask_b32 v3,v0,v1
        0xf8000941u,0x00000003u, // EXP PRIM,v3
        0xbf810000u,
    };
    const auto phase_output = count_output(scalar_vcc_phase);
    uint32_t phase_bad = 0;
    if (phase_output.size() == kWaveSize * prosper::gpu::kNggExportProbeWords)
        for (uint32_t lane = 0; lane < kWaveSize; ++lane) {
            const uint32_t got = std::bit_cast<uint32_t>(
                phase_output[lane * prosper::gpu::kNggExportProbeWords]);
            const uint32_t expected = lane < 32u ? 64u : lane * 4u;
            if (got != expected && phase_bad < 4u)
                std::fprintf(stderr, "scalar-VCC lane=%u got=%u expected=%u\n",
                             lane, got, expected);
            phase_bad += got != expected;
        }
    else phase_bad = kWaveSize;
    if (phase_bad) {
        std::fprintf(stderr, "barrier scalar-VCC reconstruction mismatches=%u\n", phase_bad);
        return 1;
    }
    scalar_vcc_phase[7] = 0xbe810380u; // write s1, not VCC_HI: no complete pair proof
    if (!recompile_ngg_exports_for_test(
            scalar_vcc_phase.data(), scalar_vcc_phase.size(), 1).empty()) {
        std::fprintf(stderr, "missing terminal VCC_HI scalar proof was admitted\n");
        return 1;
    }
    scalar_vcc_phase[7] = 0xbeeb03c1u; // VCC_HI=-1: the high half now selects v1
    const auto high_output = count_output(scalar_vcc_phase);
    if (high_output.size() != phase_output.size()) {
        std::fprintf(stderr, "high-half scalar-VCC control did not execute\n");
        return 1;
    }
    uint32_t high_distinguished = 0;
    for (uint32_t lane = 32; lane < kWaveSize; ++lane)
        high_distinguished += high_output[lane * prosper::gpu::kNggExportProbeWords] !=
            phase_output[lane * prosper::gpu::kNggExportProbeWords];
    if (high_distinguished < 24u) {
        std::fprintf(stderr, "high-half scalar-VCC control distinguished only %u lanes\n",
                     high_distinguished);
        return 1;
    }
    auto wrong_layer = exports;
    wrong_layer[24] = 0x00060000u; // POS1.z reads v6, not the prefix in v5
    const auto layer_output = run_exports(wrong_layer);
    if (layer_output.size() != kWaveSize * prosper::gpu::kNggExportProbeWords ||
        export_mismatches(layer_output) != kWaveSize) {
        std::fprintf(stderr, "layer-export control did not distinguish all lanes\n");
        return 1;
    }
    auto wrong_export_peer = exports;
    wrong_export_peer[5] = 0x3a040088u; // peer LDS address lane^2
    const auto peer_output = run_exports(wrong_export_peer);
    if (peer_output.size() != kWaveSize * prosper::gpu::kNggExportProbeWords ||
        export_mismatches(peer_output) != kWaveSize) {
        std::fprintf(stderr, "exported peer-data control did not distinguish all lanes\n");
        return 1;
    }
    auto unsupported_export = exports;
    unsupported_export[23] = 0xf80008e4u; // POS2 instead of the supported POS1
    if (!recompile_ngg_exports_for_test(unsupported_export.data(), unsupported_export.size(), 1).empty()) {
        std::fprintf(stderr, "unsupported export target was silently accepted\n");
        return 1;
    }
    std::printf("guest wave, primitive and layer exports passed; peer/mask/layer controls distinguish\n");
    return 0;
}
