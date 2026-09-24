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
