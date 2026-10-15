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
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
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
// The positive with `v_cmp_gt_u32 vcc, 8, v0` after the spills, so VCC holds a DIFFERENT set of
// lanes (x < 8) than the spilled mask (x < 40) when the halves are moved back into it. A move that
// left VCC untouched would read 0 for lanes 8..39 instead of 100.
const uint32_t kReassembledOverStaleVcc[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0x7D880088u, 0xD7600002u, 0x00010D12u,
    0xD7600003u, 0x00010F12u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u,
    0xBEFE046Au, 0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The positive with `s_branch 0` between the reloads and the move, forcing a dispatcher block edge:
// the reloaded halves' masks do not survive it, so the move must not count as a mask write.
const uint32_t kEdgeBetweenReloadAndMove[] = {
    0x7E060280u, 0x7D8800A8u, 0x87946A7Eu, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xD7600002u, 0x00010D12u, 0xD7600003u,
    0x00010F12u, 0xBF820000u, 0xBEEA0402u, 0x7DA80090u, 0xBF880001u, 0x7E060287u,
    0xBEFE046Au, 0x4A0606FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// Kena cs 0x5008e80000 (#5175): a lane-varying compare into an ordinary SGPR pair, both halves
// spilled (high first, as Kena's pc354/356), the pair reused as data, a dispatcher edge, the halves
// reloaded into s[2:3] and consumed as a mask by a B64 logical (Kena's pc778-782). An ordinary
// scalar spill and a uniform VCC branch on an SDWA scalar compare first (Kena's pc14 and pc17/29)
// leave the producer outside every participation proof, so its words are never retained: only the
// spilled Bool can carry the mask.
// s4 = 7; v19[0] = s4; vcc = (s4 < 1.0f); s_cbranch_vccz +0; v3 = 0; s[20:21] = (40 > x); v18[7] = s21; v18[6] = s20; s20 = s21 = 0;
// s_branch 0; s2 = v18[6]; s3 = v18[7]; vcc = exec & s[2:3]; v3 = vcc ? 100 : 0; out[x] = v3
const uint32_t kVopcHalvesIntoLogical[] = {
    0xBE840387u, 0xD7610013u, 0x00010004u, 0x7C03E4F9u, 0x86860004u, 0xBF860000u, 0x7E060280u,
    0xD4C40014u, 0x000200A8u, 0xD7610012u, 0x00010E15u, 0xD7610012u, 0x00010C14u, 0xBE940380u,
    0xBE950380u, 0xBF820000u, 0xD7600002u, 0x00010D12u, 0xD7600003u, 0x00010F12u, 0x87EA027Eu,
    0x7E0802FFu, 0x00000064u, 0x02060880u, 0xE0702000u, 0x80020300u,
};
// The same spill and reload, consumed by `s_mov_b64 exec, s[2:3]` (Kena's pc1791):
// ...; exec = s[2:3]; v3 = 100; exec = -1; out[x] = v3
const uint32_t kVopcHalvesIntoExec[] = {
    0xBE840387u, 0xD7610013u, 0x00010004u, 0x7C03E4F9u, 0x86860004u, 0xBF860000u, 0x7E060280u,
    0xD4C40014u, 0x000200A8u, 0xD7610012u, 0x00010E15u, 0xD7610012u, 0x00010C14u, 0xBE940380u,
    0xBE950380u, 0xBF820000u, 0xD7600002u, 0x00010D12u, 0xD7600003u, 0x00010F12u, 0xBEFE0402u,
    0x7E0602FFu, 0x00000064u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The spilled pair is a B64 logical result whose words are demanded only by its own spill (Kena's
// pc300/301 -> pc304/314): s[24:25] = (40 > x); vcc = (7 < x); s[20:21] = s[24:25] & vcc; then as
// kVopcHalvesIntoLogical, so lanes 8..39 read 100.
const uint32_t kLogicalHalvesIntoLogical[] = {
    0xBE840387u, 0xD7610013u, 0x00010004u, 0x7C03E4F9u, 0x86860004u, 0xBF860000u, 0x7E060280u,
    0xD4C40018u, 0x000200A8u, 0x7D820087u, 0x87946A18u, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xBF820000u, 0xD7600002u, 0x00010D12u, 0xD7600003u,
    0x00010F12u, 0x87EA027Eu, 0x7E0802FFu, 0x00000064u, 0x02060880u, 0xE0702000u, 0x80020300u,
};
// The reloaded HIGH half read as scalar DATA: v3 = s3, so every lane reads the upper ballot word of
// (40 > x), 0xff.
const uint32_t kHighHalfReadAsData[] = {
    0xBE840387u, 0xD7610013u, 0x00010004u, 0x7C03E4F9u, 0x86860004u, 0xBF860000u,
    0x7E060280u, 0xD4C40014u, 0x000200A8u, 0xD7610012u, 0x00010E15u, 0xD7610012u,
    0x00010C14u, 0xBE940380u, 0xBE950380u, 0xBF820000u, 0xD7600002u, 0x00010D12u,
    0xD7600003u, 0x00010F12u, 0x7E060203u, 0xE0702000u, 0x80020300u,
};
// kVopcHalvesIntoLogical with the reloads swapped: s2 = high word, s3 = low word.
const uint32_t kSwappedHalvesIntoLogical[] = {
    0xBE840387u, 0xD7610013u, 0x00010004u, 0x7C03E4F9u, 0x86860004u, 0xBF860000u, 0x7E060280u,
    0xD4C40014u, 0x000200A8u, 0xD7610012u, 0x00010E15u, 0xD7610012u, 0x00010C14u, 0xBE940380u,
    0xBE950380u, 0xBF820000u, 0xD7600002u, 0x00010F12u, 0xD7600003u, 0x00010D12u, 0x87EA027Eu,
    0x7E0802FFu, 0x00000064u, 0x02060880u, 0xE0702000u, 0x80020300u,
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
std::vector<uint32_t> compile(const uint32_t (&prefix)[N], uint32_t native_subgroup_size = 64,
                              bool define_tail_input = true, bool observed_launch = true,
                              uint64_t program_address = 0) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    // Keep all branch offsets identical in the control: it replaces the producer by a relay of
    // s30, which no launch loads, so v1 is hard on every route. (Omitting it would leave v1
    // arbitrary under #5185 on every route, #5039.)
    code.push_back(define_tail_input ? 0x7e020280u : 0x7e02021eu);
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_subgroup_size;
    if (observed_launch) prosper::test::complete_compute_launch(config);
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config,
                             {RecompileDiagnosticStage::Compute, program_address});
}

}   // namespace

TEST(CfgSpilledMaskHalves, MissingTailCompareInputStaysRefused) {
    ASSERT_FALSE(compile(kReassembled, 64, true, true, 0x50610001u).empty())
        << last_terminal_reject_reason(0x50610001u);
    EXPECT_TRUE(compile(kReassembled, 64, false, true, 0x50610002u).empty());
    const std::string reason = last_terminal_reject_reason(0x50610002u);
    EXPECT_NE(reason.find("cfg-recompile-reject mode=unresolved-operand pc=25"), std::string::npos)
        << reason;
    EXPECT_NE(reason.find("words=7c020300"), std::string::npos) << reason;
}

TEST(CfgSpilledMaskHalves, HostSubgroupWidthDoesNotSupplyGuestLaunchWords) {
    ASSERT_FALSE(compile(kReassembled, 64, true, true, 0x50610003u).empty())
        << last_terminal_reject_reason(0x50610003u);
    EXPECT_TRUE(compile(kReassembled, 64, true, false, 0x50610004u).empty());
    const std::string reason = last_terminal_reject_reason(0x50610004u);
    EXPECT_NE(reason.find("reason=native-logical-data-words-unavailable pc=2 words=87946a7e"),
              std::string::npos)
        << reason;
}

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

TEST(CfgSpilledMaskHalves, TheRebuiltMaskReplacesAStaleVcc) {
    const std::vector<uint32_t> spv = compile(kReassembledOverStaleVcc);
    ASSERT_FALSE(spv.empty());
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 16   ? 107u
                             : lane < 40 ? 100u
                                         : 0u)
            << "lane " << lane << ": EXEC is the rebuilt mask, not the stale VCC (x < 8)";
}

TEST(CfgSpilledMaskHalves, HalvesThatAreNotOneProvenPairStayRefused) {
    EXPECT_TRUE(compile(kEdgeBetweenReloadAndMove).empty())
        << "a dispatcher edge between the reloads and the move ends the reloaded halves' facts";
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

namespace {
std::vector<uint32_t> run_native(const std::vector<uint32_t>& spv) {
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    return out;
}
}   // namespace

TEST(CfgSpilledMaskHalves, ReloadedHalvesFeedAB64LogicalAsTheSpilledMask) {
    const std::vector<uint32_t> spv = compile(kVopcHalvesIntoLogical, 64, true, true, 0x50610010u);
    ASSERT_FALSE(spv.empty()) << last_terminal_reject_reason(0x50610010u);
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    const auto out = run_native(spv);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 100u : 0u) << "lane " << lane;
}

TEST(CfgSpilledMaskHalves, ReloadedHalvesRestoreExec) {
    const std::vector<uint32_t> spv = compile(kVopcHalvesIntoExec, 64, true, true, 0x50610011u);
    ASSERT_FALSE(spv.empty()) << last_terminal_reject_reason(0x50610011u);
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    const auto out = run_native(spv);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 100u : 0u) << "lane " << lane;
}

TEST(CfgSpilledMaskHalves, ALogicalResultDemandedOnlyByItsSpillTravelsAsItsBool) {
    const std::vector<uint32_t> spv =
        compile(kLogicalHalvesIntoLogical, 64, true, true, 0x50610012u);
    ASSERT_FALSE(spv.empty()) << last_terminal_reject_reason(0x50610012u);
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    const auto out = run_native(spv);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane > 7 && lane < 40 ? 100u : 0u) << "lane " << lane;
}

TEST(CfgSpilledMaskHalves, AReloadedHighHalfReadsAsItsExactUpperWord) {
    const std::vector<uint32_t> spv = compile(kHighHalfReadAsData, 64, true, true, 0x50610013u);
    ASSERT_FALSE(spv.empty()) << last_terminal_reject_reason(0x50610013u);
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    const auto out = run_native(spv);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], 0xffu) << "lane " << lane << ": the upper word of (40 > x)";
}

TEST(CfgSpilledMaskHalves, AReloadedHalfWithoutTheOriginalLaunchStaysRefused) {
    // The reload publishes its word by a ballot of the spilled Bool; without the authenticated
    // physical launch there is no such word, and the program stays refused at that reload.
    EXPECT_TRUE(compile(kHighHalfReadAsData, 64, true, false, 0x50610014u).empty());
    EXPECT_TRUE(compile(kVopcHalvesIntoLogical, 64, true, false, 0x50610015u).empty());
    EXPECT_TRUE(compile(kVopcHalvesIntoLogical, /*native_subgroup_size*/ 0).empty())
        << "without an exact native Wave64 subgroup there is no exact ballot";
}

TEST(CfgSpilledMaskHalves, SwappedReloadedHalvesAreNotTheSpilledMask) {
    // s2 holds the high half and s3 the low one: no B64 predicate is rebuilt from them, so even
    // the Bool-half retry refuses, while the same program with the reloads in order compiles. The
    // recorded reason stays the first attempt's: a failed retry never replaces it.
    EXPECT_TRUE(compile(kSwappedHalvesIntoLogical, 64, true, true, 0x50610016u).empty());
    ASSERT_FALSE(compile(kVopcHalvesIntoLogical, 64, true, true, 0x50610017u).empty());
    const std::string reason = last_terminal_reject_reason(0x50610016u);
    EXPECT_NE(reason.find("pc=9 words=d7610012,00010e15"), std::string::npos) << reason;
}

// A folded scalar branch revokes the original physical-word proof for the first reload.
// Restoring the original stream must still allow the later, independent Bool-half spill retry.
TEST(CfgSpilledMaskHalves, ConstantBranchRecoveryAlsoRetriesWordlessMaskHalves) {
    std::vector<uint32_t> code{
        0xbe840380u,   // s4 = 0
        0xbe9e047eu,   // s[30:31] = exec
        0xd7610014u, 0x0001381eu,   // v20[28] = s30
        0xbf068004u, 0xbf840001u, 0x7e040281u,   // known scalar branch
        0xd760000eu, 0x00013914u,   // s14 = v20[28]
        0x7e08020eu,   // numeric DATA use of s14
    };
    code.insert(code.end(), std::begin(kVopcHalvesIntoLogical), std::end(kVopcHalvesIntoLogical));
    // Observe both recovered domains: add the original full-EXEC low word to the mask result.
    code.insert(code.end() - 2, 0x4a06060eu);   // v3 = s14 + v3
    code.push_back(0x7e020280u);
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(config);
    const auto table = output_table();
    constexpr uint64_t address = 0x50610018u;
    TerminalRejectCapture capture;
    const auto spv = recompile_compute(code.data(), code.size(), &table, config,
                                       {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(spv.empty()) << last_terminal_reject_reason(address);
    EXPECT_TRUE(has_opcode(spv, kOpSwitch));
    for (const auto& [tag, payload] : capture.take()) {
        EXPECT_NE(tag, "cfg-recompile-reject") << payload;
        EXPECT_NE(tag, "compute-cfg-reject") << payload;
        EXPECT_NE(tag, "recompile-reject") << payload;
    }
    bpermute_oracle::Interpreter vm(spv);
    vm.native_subgroup_width = 64;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 99u : UINT32_MAX) << "lane " << lane;

    config.dispatch_launch = {};
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, config,
                                  {RecompileDiagnosticStage::Compute, address + 1})
                    .empty());
    prosper::test::complete_compute_launch(config);
    config.native_subgroup_size = 0;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, config,
                                  {RecompileDiagnosticStage::Compute, address + 2})
                    .empty());
}
