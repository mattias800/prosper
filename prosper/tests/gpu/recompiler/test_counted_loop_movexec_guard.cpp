// test_counted_loop_movexec_guard — a wave-empty EXEC guard around a counted loop,
// in the form Black Flag's luma/edge filter uses (#4816): the narrowing write goes
// straight into EXEC (`s_mov_b64 exec, vcc`, `s_or_b64 exec, sN, vcc`) and the restore
// is the constant full mask (`s_mov_b64 exec, -1`), with no saved-mask register.
//
// The counted-loop guard proof knew two encodings of "save EXEC, narrow it, skip the
// region if no lane is left" (s_and/or_saveexec, and save-then-v_cmpx). A direct EXEC
// write before `s_cbranch_execz` matched neither, so every such guard was declined and
// the dispatch skipped — even though the restore the guest executes explicitly makes the
// model exact whatever EXEC held at entry, and the region is VALU-only.
//
// The kernels below are hand-written and assembled with llvm-mc -mcpu=gfx1030 (encodings
// and branch displacements are the assembler's, not hand-computed). The positives are
// one edit away from the cmpx-guard positives (same region, same loop, same
// displacements; only the guard words differ), so a refusal can only come from the guard
// form. Each negative is ONE change to a positive:
//   * the restore replaced with s_nop (no full-EXEC restore to linearize to);
//   * the narrowing write retargeted to s[6:7] (EXEC never narrowed; not this idiom);
//   * the restore replaced with s_mov_b64 exec, 0 (only the observed full restore proves);
//   * an s_dcache_inv inside the region (a scalar memory writer would run for a skipped wave).
// The positives that can run are executed: lanes the compare switches off must keep their
// prior value, and every lane must be live again after the restore.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_counted_loop_guard.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"

#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;

// P1. x = (uint)input; out = 0; VCC = (x < 32); EXEC = VCC; skip if empty;
// sum 0..4 in a counted loop; out = float(sum); EXEC = -1 at the branch target.
// The execz is at pc 4, its target (the restore) at pc 14.
const uint32_t kMovExecGuardAtTarget[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBEFE046Au, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE04C1u, 0xBF810000u,
};
// P2. As P1, with the or-idiom narrow: s[4:5] = 0, EXEC = s[4:5] | VCC.
// The execz is at pc 6, its target (the restore) at pc 16.
const uint32_t kOrExecGuardAtTarget[] = {
    0x7E000F00u, 0x7E060280u, 0xBE840380u, 0xBE850380u, 0x7D8800A0u, 0x88FE6A04u,
    0xBF880009u, 0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u,
    0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u, 0xBEFE04C1u, 0xBF810000u,
};
// N_norestore. As P1, with s_nop where the restore was: nothing re-establishes full EXEC.
const uint32_t kNoRestore[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBEFE046Au, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBF800000u, 0xBF810000u,
};
// N_sgprdst. As P1, with the narrowing write retargeted to s[6:7]: EXEC is never narrowed.
const uint32_t kNarrowWritesSgpr[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBE86046Au, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE04C1u, 0xBF810000u,
};
// N_restorezero. As P1, with s_mov_b64 exec, 0 as the restore: only full restores prove.
const uint32_t kRestoreZero[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBEFE046Au, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE0480u, 0xBF810000u,
};
// N_dcache. As P1, with s_dcache_inv inside the loop body: a scalar memory writer in the
// region would run for a wave the guest skipped. The execz is at pc 4.
const uint32_t kScalarMemoryWriterInRegion[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBEFE046Au, 0xBF88000Bu, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850005u, 0x4A020200u, 0xF4800000u,
    0x00000000u, 0x80008100u, 0xBF82FFF9u, 0x7E060D01u, 0xBEFE04C1u, 0xBF810000u,
};

// P1r. As P1, plus v_add_f32 v3, 1.0, v3 after the restore: every lane must be live again there.
const uint32_t kGuardThenWriteAfterRestore[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBEFE046Au, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE04C1u, 0x060606F2u, 0xBF810000u,
};

// N_nested. P1's direct guard inside an outer s_and_saveexec guard. The inner restore sets EXEC
// to -1, not to the outer mask, so when the wave skips the outer region the linearized inner
// restore would run the rest of it (v_add after pc 16) on every lane. Hand-placed displacements:
// the outer execz at pc 4 targets its restore at pc 18, the inner at pc 6 targets pc 16.
const uint32_t kDirectGuardInsideSaveexecGuard[] = {
    0x7E000F00u, 0x7E060280u, 0x7D8800A0u, 0xBE8A246Au, 0xBF88000Du, 0xBEFE046Au, 0xBF880009u,
    0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE04C1u, 0x060606F2u, 0xBEFE040Au, 0xBF810000u,
};

// Decode `code`, find its counted loop and report whether the guard branch at `execz_pc` was proven.
template <size_t N>
bool guard_proven(const uint32_t (&code)[N], uint32_t execz_pc) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, N, ins);
    const CountedLoop loop = detect_counted_loop(ins);
    EXPECT_TRUE(loop.found) << "fixture must contain a counted loop";
    std::unordered_set<uint32_t> safe;
    mark_counted_loop_exec_guards(ins, loop, safe);
    return safe.count(execz_pc) != 0;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N],
                              const ShaderResourceTable* table = nullptr) {
    return recompile_valu(code, N, /*num_inputs*/ 1, /*out_vgpr*/ 3, table);
}

std::vector<float> lane_indices() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    return input;
}

}   // namespace

TEST(CountedLoopMovExecGuard, AcceptsTheDirectExecNarrowForms) {
    ASSERT_TRUE(guard_proven(kMovExecGuardAtTarget, 4))
        << "control: the proof must see the s_mov_b64 exec, vcc guard";
    EXPECT_FALSE(compile(kMovExecGuardAtTarget).empty())
        << "s_mov_b64 exec, vcc -> s_cbranch_execz around a counted loop must lower";
    ASSERT_TRUE(guard_proven(kOrExecGuardAtTarget, 6))
        << "control: the proof must see the s_or_b64 exec guard";
    EXPECT_FALSE(compile(kOrExecGuardAtTarget).empty())
        << "s_or_b64 exec, s, vcc -> s_cbranch_execz around a counted loop must lower";
}

TEST(CountedLoopMovExecGuard, RefusesWhatTheGuardCannotProve) {
    EXPECT_FALSE(guard_proven(kNoRestore, 4))
        << "no full-EXEC restore at the target leaves the narrowing unproven";
    EXPECT_TRUE(compile(kNoRestore).empty())
        << "without the restore the narrowed branch must still refuse";
    EXPECT_FALSE(guard_proven(kNarrowWritesSgpr, 4))
        << "a narrowing write that never touches EXEC is not this idiom";
    EXPECT_FALSE(guard_proven(kRestoreZero, 4))
        << "only the observed full restore proves the region boundary";
    EXPECT_TRUE(compile(kRestoreZero).empty()) << "a zero restore must still refuse";
}

TEST(CountedLoopMovExecGuard, ProofRefusesScalarMemoryWritersInTheRegion) {
    ASSERT_TRUE(guard_proven(kMovExecGuardAtTarget, 4)) << "control: the plain form is proven";
    EXPECT_FALSE(guard_proven(kScalarMemoryWriterInRegion, 4))
        << "a scalar memory writer in the region would run for a wave the guest skipped";
}

TEST(CountedLoopMovExecGuard, SwitchedOffLanesKeepTheirValueAndReturnAtTheRestore) {
    const std::vector<uint32_t> spv = compile(kGuardThenWriteAfterRestore);
    ASSERT_FALSE(spv.empty());
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    // Inside the guard only x < 32 add the loop's sum (10); after the restore every lane adds 1.
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], lane < 32 ? 11.0f : 1.0f) << "lane " << lane;
}

TEST(CountedLoopMovExecGuard, ADirectGuardIsNotABalancedRegionForAnOuterGuard) {
    EXPECT_TRUE(guard_proven(kDirectGuardInsideSaveexecGuard, 6))
        << "control: the inner direct guard is proven on its own";
    EXPECT_FALSE(guard_proven(kDirectGuardInsideSaveexecGuard, 4))
        << "its full-mask restore would widen EXEC inside an outer region the wave skipped";
}
