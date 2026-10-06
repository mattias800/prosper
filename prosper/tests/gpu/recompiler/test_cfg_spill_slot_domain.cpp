// test_cfg_spill_slot_domain — the CFG dispatcher classifies a V_WRITELANE spill slot by what its
// source holds AT THAT WRITE, not by what the same SGPR holds anywhere in the program.
//
// The dispatcher persists a spill slot across block boundaries in one of two Function-variable
// domains: a uint for scalar data, a Bool for a wave mask. It used to call a slot a mask whenever
// the source SGPR was a mask key ANYWHERE in the program. Kena's compute program 0x5008ec0000
// spills s30 at pc451, when s30 is data from an s_buffer_load, and saves EXEC into s[30:31] at
// pc1246. The slot got only a Bool, the value emit_alu stored was dropped at the next block, and
// the reload's VALU consumer at pc554 was refused.
//
// The Wave64 mask analysis now records each write whose source is, on every path, ordinary scalar
// data (a MUST scalar word, no mask or ambiguous pair, and no V_READLANE that may have defined it)
// and gives those slots the data domain. Anything it cannot decide keeps the old rule.
//
// The reload side (#4600): the analysis also carries what each slot holds on every path
// (rdna2_spill_slot_domain), and types each V_READLANE by it. A data reload is a scalar word, a
// mask reload is a mask (so a data read of it is the ballot word on a native Wave64 subgroup and
// refused on the portable one), and a reload that is a mask on only some paths is refused at its
// first read. A slot persisted in both Function variables is reloaded from the one it last got.
//
// The portable programs take the portable-readlane dispatcher route (a structured forward SCC if
// plus `v_readlane_b32 s2, v0, 0` of an ordinary VGPR); the native ones append kTail, an
// irreducible loop. Either way the spill really crosses a dispatcher block edge between the write
// (block 0) and the reload (the join block). The kernels were assembled with
// llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
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

// x = lane; s30 = 7; v20[28] = s30; if (s4 == 0) v2 = 1;
// join: s14 = v20[28]; s2 = v0[0]; v3 = float(s14 + x); s[30:31] = exec (s30 is a mask LATER).
const uint32_t kDataSpillOfALaterMask[] = {
    0x7E000F00u, 0xBE9E0387u, 0xD7610014u, 0x0001381Eu, 0xBF068004u,
    0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u, 0xD7600002u,
    0x00010100u, 0x4A06000Eu, 0x7E060D03u, 0xBE9E047Eu, 0xBF810000u,
};
// The mask-domain arm. Restoring EXEC from a reloaded mask spill needs an exact native Wave64
// subgroup (the portable dispatcher refuses `s_mov_b64 exec, s[14:15]` there), so it compiles
// through recompile_compute with native_subgroup_size = 64 and executes only where the device can
// REQUIRE a 64-lane subgroup (RADV yes, CI's lavapipe no). It spills a saved mask (lanes < 40),
// restores EXEC from the reload in the join block and writes 1 from the lanes it enables. The
// irreducible kTail (kernel 17d of test_rdna2_to_spirv) forces the CFG dispatcher. Under native
// Wave64 `s_mov_b64 s[30:31], vcc` is dual-domain, so s30 IS a MUST scalar word at the spill: only
// the mask-pair exclusion keeps this slot a mask slot.
//
// v3 = 0; vcc = (40 > x); s[30:31] = vcc; v20[28] = s30; if (s4 == 0) v2 = 1;
// join: s14 = v20[28]; exec = s[14:15]; v3 = 1; exec = -1; out[x] = v3 (1 below lane 40, else 0)
const uint32_t kMaskSpill[] = {
    0x7E060280u, 0x7D8800A8u, 0xBE9E046Au, 0xD7610014u, 0x0001381Eu,
    0xBF068004u, 0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u,
    0xBEFE040Eu, 0x7E060281u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// #4600 shape 1: the saved mask is spilled, reloaded and spilled AGAIN in one block before the
// dispatcher edge; the join block restores EXEC from the second slot. Native Wave64 only, like
// kMaskSpill, whose output it must match.
// v3 = 0; vcc = (40 > x); s[30:31] = vcc; v21[5] = s30; s30 = v21[5]; v20[28] = s30;
// if (s4 == 0) v2 = 1; join: s14 = v20[28]; exec = s[14:15]; v3 = 1; exec = -1; out[x] = v3
const uint32_t kMaskReSpill[] = {
    0x7E060280u, 0x7D8800A8u, 0xBE9E046Au, 0xD7610015u, 0x00010A1Eu, 0xD760001Eu, 0x00010B15u,
    0xD7610014u, 0x0001381Eu, 0xBF068004u, 0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u,
    0xBEFE040Eu, 0x7E060281u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// Shape 1 with the reload in an SGPR that is not otherwise a mask key: the second spill slot is a
// mask slot only because the analysis proves the write stores a Bool. Native Wave64 only.
// ... v21[5] = s30; s40 = v21[5]; v20[28] = s40; ... join as kMaskReSpill
const uint32_t kMaskReSpillThroughAnotherSgpr[] = {
    0x7E060280u, 0x7D8800A8u, 0xBE9E046Au, 0xD7610015u, 0x00010A1Eu, 0xD7600028u, 0x00010B15u,
    0xD7610014u, 0x00013828u, 0xBF068004u, 0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u,
    0xBEFE040Eu, 0x7E060281u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// #4600 shape 2: a VALU DATA read of a word reloaded from a mask slot, on the portable route.
// x = lane; s[30:31] = exec; v20[28] = s30; if (s4 == 0) v2 = 1;
// join: s14 = v20[28] (EXEC_LO = 0xffffffff); s2 = v0[0]; v3 = float(s14 + x)
const uint32_t kDataReadOfAMaskReload[] = {
    0x7E000F00u, 0xBE9E047Eu, 0xD7610014u, 0x0001381Eu, 0xBF068004u, 0xBF840001u, 0x7E040281u,
    0xD760000Eu, 0x00013914u, 0xD7600002u, 0x00010100u, 0x4A06000Eu, 0x7E060D03u, 0xBF810000u,
};
// The native Wave64 form of shape 2: a native subgroup can form the ballot word, so the reloaded
// mask's data read is the real EXEC_LO. A second branch puts a dispatcher edge between the reload
// and the read, so the read sees what crossed the edge. Native Wave64 only, like kMaskSpill.
// v3 = 0; s[30:31] = exec; v20[28] = s30; if (s4 == 0) v2 = 1;
// J: s14 = v20[28]; if (s5 == 0) v2 = 2; K: v3 = s14 + x; out[x] = v3 (x - 1 mod 2^32)
const uint32_t kNativeDataReadOfAMaskReload[] = {
    0x7E060280u, 0xBE9E047Eu, 0xD7610014u, 0x0001381Eu, 0xBF068004u,
    0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u, 0xBF068005u,
    0xBF840001u, 0x7E040282u, 0x4A06000Eu, 0xE0702000u, 0x80020300u,
};
// A slot that holds data on one path and a mask on the other: the reload has no typed value at
// the join, because the dispatcher's two Function variables carry no runtime tag. Portable route.
// x = lane; s[30:31] = exec; s12 = 7; v20[28] = s12; if (s4 == 0) v20[28] = s30;
// join: s14 = v20[28]; s2 = v0[0]; v3 = float(s14 + x)
const uint32_t kDataOrMaskSlotReload[] = {
    0x7E000F00u, 0xBE9E047Eu, 0xBE8C0387u, 0xD7610014u, 0x0001380Cu, 0xBF068004u,
    0xBF840002u, 0xD7610014u, 0x0001381Eu, 0xD760000Eu, 0x00013914u, 0xD7600002u,
    0x00010100u, 0x4A06000Eu, 0x7E060D03u, 0xBF810000u,
};
// A slot persisted in BOTH Function variables, that holds a mask at the reload: data first, then
// the saved mask over it in the same block. The reload must take the Bool, not the uint variable,
// which holds the zero placeholder. Native Wave64 only, like kMaskSpill.
// v3 = 0; s[30:31] = exec; s12 = 7; v20[28] = s12; v20[28] = s30; if (s4 == 0) v2 = 1;
// join: s14 = v20[28]; v3 = s14 + x; out[x] = v3 (x - 1 mod 2^32)
const uint32_t kDataThenMaskSlotReload[] = {
    0x7E060280u, 0xBE9E047Eu, 0xBE8C0387u, 0xD7610014u, 0x0001380Cu,
    0xD7610014u, 0x0001381Eu, 0xBF068004u, 0xBF840001u, 0x7E040281u,
    0xD760000Eu, 0x00013914u, 0x4A06000Eu, 0xE0702000u, 0x80020300u,
};
// Barrier-phased compute (#4607 review): EXEC = lanes < 40 is spilled into v20 lanes 0/1 in the
// first phase, and the phase after an unguarded `s_barrier` reloads both halves and restores EXEC.
// Each phase is compiled by its own dispatcher, the second starting from the first's terminal
// state, so the spill slots reach it as initial state with both Function variables loaded. Native
// Wave64 only.
// v3 = 0; vcc = (40 > x); exec = vcc; v20[0] = exec_lo; v20[1] = exec_hi; exec = -1; s_barrier;
// s14 = v20[0]; s15 = v20[1]; exec = s[14:15]; v3 = 1; exec = -1; out[x] = v3 (1 below 40)
const uint32_t kExecSpilledAcrossABarrier[] = {
    0x7E060280u, 0x7D8800A8u, 0xBEFE046Au, 0xD7610014u, 0x0001007Eu, 0xD7610014u,
    0x0001027Fu, 0xBEFE04C1u, 0xBF8A0000u, 0xD760000Eu, 0x00010114u, 0xD760000Fu,
    0x00010314u, 0xBEFE040Eu, 0x7E060281u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
};
// The Wave32 sibling of shape 1 (#4607 review): a one-word (B32) mask spilled and reloaded into
// the register it came from, then read as data in the same block. Straight-line, native Wave32.
// s20 = (20 > x); v21[5] = s20; s20 = v21[5]; v3 = s20 + x; out[x] = v3 (0xfffff + x)
const uint32_t kWave32MaskReloadDataRead[] = {
    0xD4C40014u, 0x00020094u, 0xD7610015u, 0x00010A14u, 0xD7600014u,
    0x00010B15u, 0x4A060014u, 0xE0702000u, 0x80020300u, 0xBF810000u,
};
// The same Wave32 reload into VCC_LO, whose mask an implicit consumer reads through the VCC
// mirror (#4607 review): vcc = (20 > x); v20[3] = vcc_lo; vcc = (8 > x); vcc_lo = v20[3];
// v3 = vcc ? 1 : 0; out[x] = v3 (1 below lane 20)
const uint32_t kWave32VccReloadCndmask[] = {
    0x7E080281u, 0x7D880094u, 0xD7610014u, 0x0001066Au, 0x7D880088u, 0xD760006Au,
    0x00010714u, 0x02060880u, 0xE0702000u, 0x80020300u, 0xBF810000u,
};
const uint32_t kTail[] = {
    0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
    0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
};

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
std::vector<uint32_t> compile_native(const uint32_t (&prefix)[N], uint32_t wave = 64) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = wave;
    config.wave_size = wave;
    config.native_subgroup_size = wave;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config);
}

// A straight-line native Wave32 kernel (no kTail), one 32-lane wave.
template <size_t N>
std::vector<uint32_t> compile_wave32(const uint32_t (&code)[N]) {
    ComputeShaderConfig config;
    config.local_x = 32;
    config.wave_size = 32;
    config.native_subgroup_size = 32;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code, N, &table, config);
}

// Runs a native-subgroup kernel over one wave of `lanes`; empty when the device cannot require it.
std::vector<uint32_t> run_native(const std::vector<uint32_t>& spv, uint32_t lanes) {
    std::vector<uint32_t> out;
    if (!prosper::test::default_compute_required_subgroup_supported(lanes, lanes)) return out;
    prosper::test::run_compute(spv, std::vector<float>(lanes, 0.0f), lanes, lanes, {},
                               std::vector<uint32_t>(lanes, 0xdeadbeefu), &out, lanes, nullptr,
                               nullptr, nullptr, lanes);
    return out;
}

// Every terminal refusal reason recorded while compiling `code` on the portable route.
template <size_t N>
std::string portable_refusals(const uint32_t (&code)[N], std::vector<uint32_t>& spv) {
    TerminalRejectCapture capture;
    spv = recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 3);
    std::string all;
    for (const auto& [tag, payload] : capture.take()) {
        all += tag;
        all += ' ';
        all += payload;
        all += '\n';
    }
    return all;
}

bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    for (size_t i = 5; i < spv.size();) {
        const uint32_t words = spv[i] >> 16u;
        if (words == 0 || i + words > spv.size()) return false;
        if ((spv[i] & 0xffffu) == opcode) return true;
        i += words;
    }
    return false;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N]) {
    return recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 3);
}

}  // namespace

TEST(CfgSpillSlotDomain, DataSpilledBeforeTheSameSgprBecomesAMaskSurvivesTheBlockEdge) {
    const std::vector<uint32_t> spv = compile(kDataSpillOfALaterMask);
    ASSERT_FALSE(spv.empty()) << "a data spill of an SGPR that is a mask only later must compile";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    const std::vector<float> got = prosper::test::run_compute(spv, input, kLanes, kLanes);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], static_cast<float>(lane + 7u))
            << "lane " << lane << ": the reloaded spill must be the stored 7";
}

TEST(CfgSpillSlotDomain, AMaskSpillKeepsTheMaskDomain) {
    const std::vector<uint32_t> spv = compile_native(kMaskSpill);
    ASSERT_FALSE(spv.empty()) << "a spilled saved mask restores EXEC across the block edge";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 1u : 0u) << "lane " << lane;
}

// #4600 shape 1: a reload that is spilled again is still the saved mask. It used to restore an
// EXEC with no lanes, so every lane read 0.
TEST(CfgSpillSlotDomain, AReSpilledMaskReloadKeepsTheMask) {
    const std::vector<uint32_t> spv = compile_native(kMaskReSpill);
    ASSERT_FALSE(spv.empty()) << "a native Wave64 dispatcher holds the reload as the mask";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 1u : 0u) << "lane " << lane;
}

// A reloaded mask re-spilled from an SGPR that is not a static mask key is persisted as a mask. The
// static rule made that slot a data slot, so the Bool was lost at the edge and EXEC restored empty.
TEST(CfgSpillSlotDomain, AMaskReSpilledThroughAnotherSgprKeepsTheMask) {
    const std::vector<uint32_t> spv = compile_native(kMaskReSpillThroughAnotherSgpr);
    ASSERT_FALSE(spv.empty()) << "a native Wave64 dispatcher holds the reload as the mask";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 1u : 0u) << "lane " << lane;
}

// #4600 shape 2: the reloaded word is EXEC_LO's 0xffffffff, so v3 = x - 1 (mod 2^32). It used to
// compile and read the word as 0 (v3 = x). The portable dispatcher cannot form a ballot word of a
// Wave64 mask, so it must refuse, and at the data read itself: the reload crossed the edge as a
// mask, and operand_bits declines a portable data read of one.
TEST(CfgSpillSlotDomain, APortableDataReadOfAMaskReloadIsRefused) {
    std::vector<uint32_t> spv;
    const std::string reasons = portable_refusals(kDataReadOfAMaskReload, spv);
    EXPECT_TRUE(spv.empty()) << "a portable data read of a reloaded mask has no word to read";
    EXPECT_NE(reasons.find("cfg-recompile-reject mode=unresolved-operand pc=11 words=4a06000e"),
              std::string::npos)
        << "the refusal is the data read of s14, v_add_nc_u32 v3, s14, v0\n"
        << reasons;
}

// The native form of shape 2: the reloaded mask crosses a dispatcher edge as a mask, and its data
// read takes the mask's low ballot word, EXEC_LO = 0xffffffff. Under the bug the word crossed as
// the zero placeholder of its uint variable, so out[x] = x.
TEST(CfgSpillSlotDomain, ANativeDataReadOfAMaskReloadIsTheBallotWord) {
    const std::vector<uint32_t> spv = compile_native(kNativeDataReadOfAMaskReload);
    ASSERT_FALSE(spv.empty()) << "a native Wave64 dispatcher can form the reloaded mask's word";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane - 1u) << "lane " << lane << ": s14 is EXEC_LO, 0xffffffff";
}

// A reload whose slot is data on one path and a mask on another is refused at its first read. It
// used to compile and read the uint variable, the zero placeholder on the mask path.
TEST(CfgSpillSlotDomain, AReloadOfADataOrMaskSlotIsRefused) {
    std::vector<uint32_t> spv;
    const std::string reasons = portable_refusals(kDataOrMaskSlotReload, spv);
    EXPECT_TRUE(spv.empty()) << "neither Function variable holds the reloaded word on every path";
    EXPECT_NE(reasons.find("pc=13 reason=wave64-ambiguous-mask-read"), std::string::npos)
        << reasons;
}

// A slot that has both a data and a mask Function variable is reloaded from the one it last
// received. It used to reload both, and the data read took the uint variable's zero: out[x] = x.
TEST(CfgSpillSlotDomain, AReloadTakesTheDomainTheSlotLastReceived) {
    const std::vector<uint32_t> spv = compile_native(kDataThenMaskSlotReload);
    ASSERT_FALSE(spv.empty()) << "a native Wave64 dispatcher can form the reloaded mask's word";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    if (!prosper::test::default_compute_required_subgroup_supported(64u, kLanes))
        GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    std::vector<uint32_t> out;
    prosper::test::run_compute(spv, std::vector<float>(kLanes, 0.0f), kLanes, kLanes, {},
                               std::vector<uint32_t>(kLanes, 0xdeadbeefu), &out, kLanes, nullptr,
                               nullptr, nullptr, 64u);
    ASSERT_EQ(out.size(), kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane - 1u) << "lane " << lane << ": s14 is EXEC_LO, 0xffffffff";
}

// A spill slot that reaches a barrier phase as initial state keeps the dispatcher's untyped
// behaviour. The slot-domain analysis used to seed every initial Bool slot as an untyped-half mask,
// so under native Wave64 both reloads were ambiguous and the restore was refused (#4607 review).
TEST(CfgSpillSlotDomain, AnExecSpilledAcrossABarrierIsRestored) {
    const std::vector<uint32_t> spv = compile_native(kExecSpilledAcrossABarrier);
    ASSERT_FALSE(spv.empty()) << "the phase after the barrier restores EXEC from its spill";
    EXPECT_TRUE(has_opcode(spv, kOpSwitch)) << "it lowered through the CFG dispatcher";
    const std::vector<uint32_t> out = run_native(spv, kLanes);
    if (out.empty()) GTEST_SKIP() << "the device cannot require a 64-lane compute subgroup";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_EQ(out[lane], lane < 40 ? 1u : 0u) << "lane " << lane;
}

// Shape 1's Wave32 sibling (#4607 review). record_scalar_write ended the destination's B32 marker
// on any scalar write except a B32 mask writer, and with it the Bool the reload had just
// published, so the data read took the untracked SGPR's silent 0: out[x] = x. That loop is the
// same on main, so the defect predates #4607 and is independent of its Wave64 preserve rule. This
// fixes it within one block only; across a dispatcher edge the read is still 0 (#4613).
TEST(CfgSpillSlotDomain, AWave32MaskReloadedIntoItsRegisterKeepsItsWord) {
    constexpr uint32_t kWave32 = 32;
    const std::vector<uint32_t> spv = compile_wave32(kWave32MaskReloadDataRead);
    ASSERT_FALSE(spv.empty()) << "a native Wave32 subgroup can form the reloaded mask's word";
    const std::vector<uint32_t> out = run_native(spv, kWave32);
    if (out.empty()) GTEST_SKIP() << "the device cannot require a 32-lane compute subgroup";
    for (uint32_t lane = 0; lane < kWave32; ++lane)
        EXPECT_EQ(out[lane], 0xfffffu + lane) << "lane " << lane << ": s20 is the mask, lanes < 20";
}

// A Wave32 mask reloaded into VCC_LO must also reach the implicit VCC consumers. The rule above
// kept the reload's Bool on VCC_LO but left the mirror rs.vcc at the later compare, so v_cndmask
// selected on lanes < 8 instead of the reloaded lanes < 20 (#4607 review). Main refused it.
TEST(CfgSpillSlotDomain, AWave32MaskReloadedIntoVccReachesCndmask) {
    constexpr uint32_t kWave32 = 32;
    const std::vector<uint32_t> spv = compile_wave32(kWave32VccReloadCndmask);
    ASSERT_FALSE(spv.empty()) << "the reloaded VCC_LO mask is a one-word mask";
    const std::vector<uint32_t> out = run_native(spv, kWave32);
    if (out.empty()) GTEST_SKIP() << "the device cannot require a 32-lane compute subgroup";
    for (uint32_t lane = 0; lane < kWave32; ++lane)
        EXPECT_EQ(out[lane], lane < 20 ? 1u : 0u) << "lane " << lane << ": vcc is the reload";
}
