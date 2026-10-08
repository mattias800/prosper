// test_lane_slot_carry — constant-lane V_WRITELANE spill slots across structured joins.
//
// Kena's level load lost the Vulkan device: pixel program 0x5007ad0000 keeps its loop counter in a
// spill slot (`v_writelane_b32 v20, s16, 40` at the latch, `v_readlane_b32 s17, v20, 40` at the
// header), and the structured loop emitter gave every register a header phi except the slots. The
// recompiled exit test read the preheader counter forever -- an infinite loop on the GPU.
//
// The kernels are hand-written and assembled with llvm-mc -mcpu=gfx1030 (encodings and branch
// displacements are the assembler's). The loop kernels carry a SECOND, ordinary SGPR trip guard
// (at most 64 iterations), so a regression terminates with a wrong answer instead of hanging the
// GPU this test runs on.
#include "gpu/recompiler/rdna2_lane_slot_carry.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include "fixtures/compute_runner.h"

#include <cstddef>
#include <iterator>
#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kLanes = 64;

// counter = 0 and count = 5 spilled to v20 lanes 40/16; while (counter < count && guard < 64)
// { v1 += 2; ++guard; ++counter (reload, add, re-spill) }; out = v1 + reload(counter) + lane.
// One loop: the counted-loop route. Expected 10 + 5 + lane.
const uint32_t kCounterInSlot[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE850385u, 0xD7610014u, 0x00012005u,
    0x7E020280u, 0xBE8B0380u, 0xD7600006u, 0x00012114u, 0xD7600007u, 0x00015114u, 0xBF040607u,
    0x850A8081u, 0xBF0AC00Bu, 0x850C8081u, 0x870A0C0Au, 0xBF07800Au, 0xBF840008u, 0x4A020282u,
    0x800B810Bu, 0xD7600006u, 0x00015114u, 0x80068106u, 0xD7610014u, 0x00015006u, 0xBF82FFEDu,
    0xD7600008u, 0x00015114u, 0x4A020208u, 0x4A020300u, 0x7E060D01u, 0xBF810000u,
};
// The same outer loop with count = 3 around an inner `for (j = 0; j < 2; ++j) v1 += 1`, as in Kena's
// program (inner loop at pc 231..378 of the outer 197..866). Two loops: the general structured
// route. Expected 6 + 3 + lane.
const uint32_t kCounterInSlotNested[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE850383u, 0xD7610014u, 0x00012005u,
    0x7E020280u, 0xBE8B0380u, 0xD7600006u, 0x00012114u, 0xD7600007u, 0x00015114u, 0xBF040607u,
    0x850A8081u, 0xBF0AC00Bu, 0x850C8081u, 0x870A0C0Au, 0xBF07800Au, 0xBF84000Du, 0x800B810Bu,
    0xBE8D0380u, 0xBF0A820Du, 0xBF840003u, 0x4A020281u, 0x800D810Du, 0xBF82FFFBu, 0xD7600006u,
    0x00015114u, 0x80068106u, 0xD7610014u, 0x00015006u, 0xBF82FFE8u, 0xD7600008u, 0x00015114u,
    0x4A020208u, 0x4A020300u, 0x7E060D01u, 0xBF810000u,
};
// counter = 0 in v20[40]; loop: v20[41] = counter + 1000 IN THE CONDITION REGION; exit when
// counter >= 3 (or 64 trips); body: v20[41] = 7, ++counter. After the loop v20[41] is the value the
// last check wrote (1003), not the body's 7: the exit leaves from the check block. Expected
// 1003 + lane.
const uint32_t kExitTakesTheCheckValue[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE8B0380u, 0xD7600007u, 0x00015114u,
    0xB78703E8u, 0xD7610014u, 0x00015207u, 0xD7600007u, 0x00015114u, 0xBF0A8307u, 0x850A8081u,
    0xBF0AC00Bu, 0x850C8081u, 0x870A0C0Au, 0xBF07800Au, 0xBF840008u, 0x800B810Bu, 0xBE8F0387u,
    0xD7610014u, 0x0001520Fu, 0x80078107u, 0xD7610014u, 0x00015007u, 0xBF82FFEAu, 0xD7600008u,
    0x00015314u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
// slot v20[40] = 0; for (i = 0; i < 3; ++i) { v1 += reload(slot); re-spill slot + 1; then
// v_mov_b32 v20, 5 }. The ordinary write ends the spill lifetime inside the body, so on hardware
// trips 2 and 3 reload VECTOR data, 5 (10 + lane). The overwrite is deliberately not the seed (0):
// with 0 a phi closed by its seed would give the hardware answer by coincidence.
const uint32_t kBodyEndsTheSpillLifetime[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE800380u, 0x7E020280u, 0xBF0A8300u,
    0xBF840009u, 0xD7600006u, 0x00015114u, 0x4A020206u, 0x80068106u, 0xD7610014u, 0x00015006u,
    0x7E280285u, 0x80008100u, 0xBF82FFF5u, 0x4A020300u, 0x7E060D01u, 0xBF810000u,
};
// The same lifetime end, but the loop never reloads the slot: v1 += 2 per trip, the slot is
// spilled and then overwritten (v_mov_b32 v20, 5), and nothing reads it. Expected 6 + lane.
const uint32_t kBodyEndsAnUnreloadedSpill[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE800380u, 0x7E020280u,
    0xBF0A8300u, 0xBF840006u, 0x4A020282u, 0xD7610014u, 0x00015000u, 0x7E280285u,
    0x80008100u, 0xBF82FFF8u, 0x4A020300u, 0x7E060D01u, 0xBF810000u,
};
// slot v20[3] = 7; if (s9 == 1) v_mov_b32 v20, v0 (ends the array on the taken arm); then an
// ORDINARY read v1 = v20 + v0. On the skipped edge V_WRITELANE erased v20's vector value, so the
// merge has nothing honest to give that read.
const uint32_t kOrdinaryReadAfterEndedArray[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u,
    0xBF840001u, 0x7E280300u, 0x4A020114u, 0x7E060D01u, 0xBF810000u,
};
// As above, without the read (v1 = v0): a dead conflict must still compile.
const uint32_t kEndedArrayNotRead[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u,
    0xBF840001u, 0x7E280300u, 0x7E020300u, 0x7E060D01u, 0xBF810000u,
};
// #4740 review probe, a Wave64 FRAGMENT program on the structured path: v12[1] is spilled on the
// taken arm only, reloaded into s[4:5] after the merge, and projected onto lane bits by
// s_and_b64 with a compare mask. On the skipped edge the slot is the merge's placeholder 0, which
// a projection must not consume (#4725). The control spills v12[1] before the branch as well
// (replacing the two s_nops), so both edges hold a real word.
const uint32_t kFragmentOneEdgeSlotProjected[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c2000au, 0x00010080u, 0xbf800000u, 0xbf800000u,
    0xbf068000u, 0xbf850003u, 0xbe9503c1u, 0xd761000cu, 0x00010215u, 0xd7600004u,
    0x0001030cu, 0xd7600005u, 0x0001030cu, 0x87860a04u, 0xd5010001u, 0x0019e480u,
    0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};
const uint32_t kFragmentBothEdgesSlotProjected[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c2000au, 0x00010080u, 0xd761000cu, 0x00010200u,
    0xbf068000u, 0xbf850003u, 0xbe9503c1u, 0xd761000cu, 0x00010215u, 0xd7600004u,
    0x0001030cu, 0xd7600005u, 0x0001030cu, 0x87860a04u, 0xd5010001u, 0x0019e480u,
    0x7e000280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
};
// slot v20[40] = 0 (data) before the loop; the body re-spills EXEC_LO into the same lane.
const uint32_t kBodyFlipsTheSlotToAMask[] = {
    0x7E000F00u, 0xBE840380u, 0xD7610014u, 0x00015004u, 0xBE800380u, 0x7E020280u,
    0xBF0A8300u, 0xBF840007u, 0xD7600006u, 0x00015114u, 0x4A020206u, 0xD7610014u,
    0x0001507Eu, 0x80008100u, 0xBF82FFF7u, 0x4A020300u, 0x7E060D01u, 0xBF810000u,
};
// slot v20[3] = 7; s9 = K; if (s9 == 1) slot = 100; out = reload(slot) + lane.
const uint32_t kIfTaken[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u,
    0xBF068109u, 0xBF840003u, 0xB0040064u, 0xD7610014u, 0x00010604u,
    0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
const uint32_t kIfSkipped[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890380u,
    0xBF068109u, 0xBF840003u, 0xB0040064u, 0xD7610014u, 0x00010604u,
    0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
// slot = 7; if (s9 == 1) slot = 100; else slot = 50; out = reload(slot) + lane.
const uint32_t kIfElseThen[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u, 0xBF840004u,
    0xB0040064u, 0xD7610014u, 0x00010604u, 0xBF820003u, 0xB0040032u, 0xD7610014u, 0x00010604u,
    0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
const uint32_t kIfElseElse[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890380u, 0xBF068109u, 0xBF840004u,
    0xB0040064u, 0xD7610014u, 0x00010604u, 0xBF820003u, 0xB0040032u, 0xD7610014u, 0x00010604u,
    0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
// slot = 7 (data); if (s9 == 1) v_writelane v20, exec_lo, 3 (a MASK into the same lane);
// out = reload(slot) + lane. No single phi type joins data and a mask.
const uint32_t kIfMaskReloaded[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u, 0xBF840002u,
    0xD7610014u, 0x0001067Eu, 0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
// As kIfMaskReloaded without the reload: the conflicting slot is dead after the merge.
const uint32_t kIfMaskDead[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u,
    0xBF840002u, 0xD7610014u, 0x0001067Eu, 0x7E020300u, 0x7E060D01u, 0xBF810000u,
};
// As kIfMaskReloaded with a DATA source (s9) in the arm: the reload is well defined. Expected
// 1 + lane.
const uint32_t kIfDataReloaded[] = {
    0x7E000F00u, 0xBE840387u, 0xD7610014u, 0x00010604u, 0xBE890381u, 0xBF068109u, 0xBF840002u,
    0xD7610014u, 0x00010609u, 0xD7600008u, 0x00010714u, 0x4A020008u, 0x7E060D01u, 0xBF810000u,
};
// A kernel with no spill slots, run first: an empty result there means no Vulkan device, while an
// empty result for a kernel under test means the device refused its SPIR-V.
const uint32_t kDeviceProbe[] = {0x7E000F00u, 0x7E020300u, 0x7E060D01u, 0xBF810000u};

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N]) {
    return prosper::gpu::recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 3);
}

std::vector<float> lane_indices() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    return input;
}

bool have_device() {
    return !prosper::test::run_compute(compile(kDeviceProbe), lane_indices(), kLanes, kLanes)
                .empty();
}

// Compile and run `code`; every lane must read `base + lane`.
template <size_t N>
void expect_lanes(const uint32_t (&code)[N], uint32_t base, const char* what) {
    const std::vector<uint32_t> spv = compile(code);
    ASSERT_FALSE(spv.empty()) << what << ": must recompile";
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes);
    ASSERT_EQ(got.size(), kLanes) << what << ": the device refused the recompiled SPIR-V";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], static_cast<float>(base + lane)) << what << ", lane " << lane;
}

}   // namespace

// The Kena shape: the exit test reads the loop counter from a spill slot written at the latch.
// Without a header phi for the slot the counter never moves and only the guard ends the loop
// (64 trips: 128 + 0 + lane).
TEST(LaneSlotCarry, CountedLoopCarriesItsSpilledCounter) {
    if (!have_device()) GTEST_SKIP() << "no Vulkan compute device";
    expect_lanes(kCounterInSlot, 15, "counter spilled across a counted loop");
}

TEST(LaneSlotCarry, StructuredLoopCarriesItsSpilledCounterAroundAnInnerLoop) {
    if (!have_device()) GTEST_SKIP() << "no Vulkan compute device";
    expect_lanes(kCounterInSlotNested, 9, "counter spilled across an outer structured loop");
}

// A slot written in both the condition region and the body leaves the loop with the condition
// region's value. The body's would also be an SSA id that does not dominate the merge.
TEST(LaneSlotCarry, LoopExitTakesTheConditionRegionValue) {
    if (!have_device()) GTEST_SKIP() << "no Vulkan compute device";
    expect_lanes(kExitTakesTheCheckValue, 1003, "slot rewritten by the check and the body");
}

// A body that ends a carried slot's lifetime closes the phi with the loop-invariant seed (the
// lane holds vector data on the next trip); a body that re-spills a mask into a data slot has no
// phi type and refuses.
TEST(LaneSlotCarry, LoopBodyLifetimeEndAndDomainFlip) {
    EXPECT_TRUE(compile(kBodyEndsTheSpillLifetime).empty())
        << "a slot the loop reloads must not be closed with its seed once the body ends its "
           "lifetime: hardware reloads the lane's vector data (10 + lane), the seed gives 0 + lane";
    if (have_device())
        expect_lanes(kBodyEndsAnUnreloadedSpill, 6, "control: lifetime ended, slot never reloaded");
    else
        EXPECT_FALSE(compile(kBodyEndsAnUnreloadedSpill).empty());
    EXPECT_TRUE(compile(kBodyFlipsTheSlotToAMask).empty())
        << "a slot entering as data and leaving the body as a mask must refuse";
}

// After an if merge where one edge ended the spill array, an ORDINARY read of the VGPR refuses
// instead of reading the slot edge's placeholder in every lane.
TEST(LaneSlotCarry, OrdinaryReadOfAnArrayEndedOnOneEdgeRefuses) {
    EXPECT_FALSE(compile(kEndedArrayNotRead).empty()) << "control: the merge itself compiles";
    EXPECT_TRUE(compile(kOrdinaryReadAfterEndedArray).empty())
        << "an ordinary read of a VGPR whose vector value one edge erased must refuse";
}

// A slot written on one if-edge only is a fabricated word on the other edge (#4725's mark), so
// the fragment projection of its reload must refuse.
TEST(LaneSlotCarry, FragmentProjectionOfAOneEdgeSlotRefuses) {
    ASSERT_FALSE(prosper::gpu::recompile_fragment(kFragmentBothEdgesSlotProjected,
                                                  std::size(kFragmentBothEdgesSlotProjected))
                     .empty())
        << "control: with the slot written on both edges the projection compiles, or the next "
           "check is void";
    EXPECT_TRUE(prosper::gpu::recompile_fragment(kFragmentOneEdgeSlotProjected,
                                                 std::size(kFragmentOneEdgeSlotProjected))
                    .empty())
        << "projecting the merge's placeholder onto lane bits must refuse";
}

// A one-arm if: the skipped edge keeps the entry slot, the taken edge the arm's. Without a merge
// phi the taken arm's value (an SSA id that does not dominate the merge) reached both paths.
TEST(LaneSlotCarry, OneArmIfJoinsTheSlotFromBothEdges) {
    if (!have_device()) GTEST_SKIP() << "no Vulkan compute device";
    expect_lanes(kIfTaken, 100, "slot written in the taken arm");
    expect_lanes(kIfSkipped, 7, "slot written in the skipped arm");
}

// An if/else: emission rolls the state back to the branch before the else arm, which used to
// discard the then arm's slot write outright.
TEST(LaneSlotCarry, IfElseJoinsTheSlotFromBothArms) {
    if (!have_device()) GTEST_SKIP() << "no Vulkan compute device";
    expect_lanes(kIfElseThen, 100, "slot written in the then arm");
    expect_lanes(kIfElseElse, 50, "slot written in the else arm");
}

// Data on one edge and a mask on the other cannot be joined: the slot is dropped at the merge, so
// a reload refuses while a merge that never reloads it still compiles.
TEST(LaneSlotCarry, ADataMaskConflictRefusesOnlyItsReload) {
    if (have_device())
        expect_lanes(kIfDataReloaded, 1, "control: data written in the arm");
    else
        ASSERT_FALSE(compile(kIfDataReloaded).empty()) << "control: data written in the arm";
    EXPECT_FALSE(compile(kIfMaskDead).empty())
        << "a dead data/mask conflict at a merge must not refuse the program";
    EXPECT_TRUE(compile(kIfMaskReloaded).empty())
        << "a reload of a slot that is data on one edge and a mask on the other must refuse";
}

// The join itself, on hand-built edge states. An executed kernel cannot show the data/mask drop:
// in compute a reload of a mask-domain slot yields no scalar either way, so a merge that kept the
// mask (and lost the data edge's value) would refuse just the same. Stages that DO reload a mask
// slot as data (exact Wave64 halves) would then read the wrong value silently.
TEST(LaneSlotCarry, JoinPhisDiffersPlaceholdsAbsenceAndDropsConflicts) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin(0);
    b.is_compute = true;
    const uint32_t seven = b.uconst(7), hundred = b.uconst(100);
    LaneSlotEdge first, second;
    first.block = 11;
    second.block = 12;
    first.data[20][3] = seven;   // differs: phi
    second.data[20][3] = hundred;
    first.data[20][4] = seven;   // equal: no phi
    second.data[20][4] = seven;
    first.data[20][5] = seven;   // one edge only: phi against the placeholder
    first.data[21][0] = seven;   // data on one edge, a mask on the other: dropped
    second.mask[21][0] = b.btrue();
    first.data[22][0] = seven;   // the other edge ended this array's lifetime
    second.invalidated.insert(22);
    first.data[23][0] = seven;   // an earlier merge left the other edge's array EMPTY (ended)
    second.data[23];
    RegState rs;
    join_lane_slots(b, rs, first, second, /*branch_pc*/ 6);

    ASSERT_TRUE(rs.vgpr_lane_slots.contains(20));
    const auto& v20 = rs.vgpr_lane_slots.at(20);
    ASSERT_EQ(v20.size(), 3u);
    EXPECT_NE(v20.at(3), seven);
    EXPECT_NE(v20.at(3), hundred) << "differing values must be joined by a new phi";
    EXPECT_EQ(v20.at(4), seven) << "equal values need no phi";
    EXPECT_NE(v20.at(5), seven) << "a slot absent on one edge is a phi against the placeholder";
    EXPECT_TRUE(rs.lane_slot_merge_placeholder.contains({20, 5}))
        << "and that placeholder is marked fabricated, like an SGPR absent on one edge";
    EXPECT_FALSE(rs.lane_slot_merge_placeholder.contains({20, 3}))
        << "a slot both edges wrote is not fabricated";
    EXPECT_FALSE(rs.lane_slot_merge_placeholder.contains({20, 4}));
    // An ended or dropped array stays as an EMPTY data entry: operand_bits refuses an ordinary
    // read of a spill array, and V_READLANE refuses a lane the array does not hold.
    for (int vgpr : {21, 22, 23}) {
        ASSERT_TRUE(rs.vgpr_lane_slots.contains(vgpr)) << "v" << vgpr << " must stay an array";
        EXPECT_TRUE(rs.vgpr_lane_slots.at(vgpr).empty()) << "v" << vgpr << " keeps no slot";
        EXPECT_FALSE(rs.vgpr_lane_mask_slots.contains(vgpr)) << "v" << vgpr << " keeps no mask";
    }
}
