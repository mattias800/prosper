// test_cfg_spilled_mask_halves — a Wave64 mask rebuilt from its two spilled halves stays a mask
// across a CFG-dispatcher block edge.
//
// Kena's compute program 0x5008ec0000 spills both words of a saved mask, s[20:21] =
// s_and_b64(...), into v18 lanes 7 and 6 (pc304/pc314), reuses s[20:21] as data, and much later
// reloads the halves with v_readlane into s[10:11] (pc1783/pc1785), moves them into VCC
// (pc1796 `s_mov_b64 vcc, s[10:11]`), branches on EXEC (pc1827 execz) and restores
// `s_mov_b64 exec, vcc` in the next block (pc1857). emit_alu already rebuilds the mask: the
// low-half slot holds the saved pair's Bool. But the dispatcher's Wave64 mask analysis treated the
// readlanes as plain scalar writes, so VCC was not a mask at the block edge, was not reloaded, and
// the restore was refused as an unresolved operand.
//
// The analysis now tracks which saved-mask instance each spill slot and each reloaded SGPR holds a
// half of (keyed by the defining pc, with every redefinition at that pc ending the older facts).
// `s_mov_b64 dst, s[N:N+1]` is a mask write when sN holds half 0 and sN+1 half 1 of the SAME
// instance. It is applied only with an exact native Wave64 subgroup, where the move also
// materializes the ballot words its dual-domain fact claims.
//
// Every kernel appends kTail (kernel 17d of test_rdna2_to_spirv), an irreducible loop that forces
// the CFG dispatcher. Execution needs a device that can REQUIRE a 64-lane compute subgroup (RADV
// yes, CI's lavapipe no), so it is gated; the compile-level assertions run everywhere. Kernels were
// assembled with llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include "fixtures/compute_runner.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;
constexpr uint32_t kOpSwitch = 251;

// v3 = 0; vcc = (40 > x); s[20:21] = exec & vcc; v18[7] = s21; v18[6] = s20; s20 = s21 = 0;
// s2 = v18[6]; s3 = v18[7]; vcc = s[2:3]; exec &= (16 > x); execz -> L; v3 = 7;
// L: exec = vcc; v3 += 100; exec = -1; out[x] = v3
// so lanes below 16 read 107, lanes 16..39 read 100 and the rest read 0. (s[8:11] is the output
// V#, so the reloads use s[2:3] rather than Kena's s[10:11].)
const uint32_t kReassembled[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xD7600002u, 0x00010D12u, 0xD7600003u,
    0x00010F12u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u, 0xBEFE046Au,
    0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The low half is spilled from one s_and_b64 and the high half from a SECOND one at another pc.
// The words are numerically the same here, but they are two instances, so the pair is not proven.
const uint32_t kHalvesOfTwoInstances[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010C14u, 0x87946A7Eu,
    0xD7610012u, 0x00010E15u, 0xBE940380u, 0xBE950380u, 0xD7600002u, 0x00010D12u,
    0xD7600003u, 0x00010F12u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u,
    0xBEFE046Au, 0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The positive with the reloads swapped: s2 gets the HIGH half and s3 the low one.
const uint32_t kSwappedHalves[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xD7600002u, 0x00010F12u, 0xD7600003u,
    0x00010D12u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u, 0xBEFE046Au,
    0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The positive with `s_mov_b32 s3, 0` between the reloads and the move into VCC.
const uint32_t kHighHalfOverwritten[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xD7600002u, 0x00010D12u, 0xD7600003u,
    0x00010F12u, 0xBE830380u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u,
    0xBEFE046Au, 0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
const uint32_t kTail[] = {
    0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
    0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
};

bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    for (size_t i = 5; i < spv.size();) {
        const uint32_t words = spv[i] >> 16u;
        if (words == 0 || i + words > spv.size()) return false;
        if ((spv[i] & 0xffffu) == opcode) return true;
        i += words;
    }
    return false;
}

// out[] is binding 3 (run_compute's cbuf1), addressed through the direct V# at s[8:11].
ShaderResourceTable output_table() {
    ShaderResourceTable table;
    ShaderResource out{};
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    table.resources.push_back(out);
    return table;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&prefix)[N], uint32_t native_subgroup_size = 64) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_subgroup_size;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config);
}

}   // namespace

TEST(CfgSpilledMaskHalves, AMaskRebuiltFromBothSpilledHalvesRestoresExecAcrossTheEdge) {
    const std::vector<uint32_t> spv = compile(kReassembled);
    ASSERT_FALSE(spv.empty()) << "VCC rebuilt from both proven halves is a mask at the block edge";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 16 ? 107u : lane < 40 ? 100u : 0u) << "lane " << lane;
}

TEST(CfgSpilledMaskHalves, HalvesThatAreNotOneProvenPairStayRefused) {
    EXPECT_TRUE(compile(kHalvesOfTwoInstances).empty())
        << "halves spilled from two different saved-mask instances are not one mask";
    EXPECT_TRUE(compile(kSwappedHalves).empty()) << "the halves must be low in sN, high in sN+1";
    EXPECT_TRUE(compile(kHighHalfOverwritten).empty())
        << "a write between the reload and the move ends the high half's fact";
    // A regression pin only: the portable dispatcher refuses this program for its own reasons, so
    // it stays refused even with the rule's native-Wave64 gate removed (measured).
    EXPECT_TRUE(compile(kReassembled, /*native_subgroup_size*/ 0).empty())
        << "without an exact native Wave64 subgroup the program stays refused";
}
