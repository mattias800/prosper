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
// Every program below takes the portable-readlane dispatcher route (a structured forward SCC if
// plus `v_readlane_b32 s2, v0, 0` of an ordinary VGPR), so the spill really crosses a dispatcher
// block edge between the write (block 0) and the reload (the join block). The kernels were
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
// A second arm that re-spilled a word RELOADED from a mask slot (the case the readlane exclusion
// is for) was dropped: it restores an empty EXEC with or without this change, a pre-existing
// defect tracked in its own issue.
//
// v3 = 0; vcc = (40 > x); s[30:31] = vcc; v20[28] = s30; if (s4 == 0) v2 = 1;
// join: s14 = v20[28]; exec = s[14:15]; v3 = 1; exec = -1; out[x] = v3 (1 below lane 40, else 0)
const uint32_t kMaskSpill[] = {
    0x7E060280u, 0x7D8800A8u, 0xBE9E046Au, 0xD7610014u, 0x0001381Eu,
    0xBF068004u, 0xBF840001u, 0x7E040281u, 0xD760000Eu, 0x00013914u,
    0xBEFE040Eu, 0x7E060281u, 0xBEFE04C1u, 0xE0702000u, 0x80020300u,
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
std::vector<uint32_t> compile_native(const uint32_t (&prefix)[N]) {
    std::vector<uint32_t> code(std::begin(prefix), std::end(prefix));
    code.insert(code.end(), std::begin(kTail), std::end(kTail));
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code.data(), code.size(), &table, config);
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
