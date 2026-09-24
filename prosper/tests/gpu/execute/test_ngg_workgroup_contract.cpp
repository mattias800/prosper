// The guest wave contract needed by merged-NGG producers: a lane exchanges an LDS value with its
// neighbour, reconverges after an EXEC-narrowed prefix count, and preserves the result. The current
// compute shell supplies the workgroup semantics; this does not admit a graphics program by itself.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
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
    const auto actual = run(kGuest, &portable_wave);
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
