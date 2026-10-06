// test_counted_loop_cmpx_guard — a wave-empty EXEC guard around a counted loop, in the form UE4's
// NGG vertex shaders use (#4427).
//
// The counted-loop lowering already accepted "s_and_saveexec_b64 sN, cond -> s_cbranch_execz ->
// s_mov_b64 exec, sN" around a counted loop (the Evergate shape, test_recompile_coverage). Kena's
// lighting-loop vertex programs encode the same guard differently, and all 13 of them were refused at
// the s_cbranch_execz:
//
//     s_mov_b64 s[34:35], exec          ; save EXEC early
//     ...
//     v_cmpx_nlt_f32 ...                ; narrow EXEC directly before the branch
//     s_cbranch_execz target            ; skip the region if no lane is left
//     ...  buffer_load / image_sample   ; the region READS through descriptors
//     ...  counted loop (s_cbranch_scc1 back-edge)
//   target:
//     s_load_dwordx4 ...                ; UE4 schedules the next scalar load first
//     s_mov_b64 exec, s[34:35]          ; restore
//
// The kernels below are hand-written and assembled with llvm-mc -mcpu=gfx1030 (encodings and branch
// displacements are the assembler's, not hand-computed). Each negative is ONE change to a positive,
// so a refusal can only come from the property that changed:
//   * the saved mask overwritten between the save and the narrowing compare;
//   * a vector instruction in the window between the branch target and the restore;
//   * a buffer STORE in the guarded region, paired with the same store placed after the restore,
//     which must still compile (so the refusal is about where the store is, not the store itself).
// The positives that can run are executed: lanes the compare switches off must keep their prior
// value, and a buffer read inside the region must deliver its data to the lanes that stay on.
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

// P1. x = (uint)input; out = 0; save EXEC; EXEC &= (x < 32); skip if empty;
// sum 0..4 in a counted loop; out = float(sum); restore at the branch target.
const uint32_t kGuardAtTarget[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u, 0xB0020005u, 0xBE800380u,
    0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u,
    0xBEFE0404u, 0xBF810000u,
};
// P2. As P1, with an s_waitcnt at the branch target: the restore is one instruction later.
const uint32_t kRestoreAfterTarget[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u, 0xB0020005u, 0xBE800380u,
    0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u,
    0xBF8CC07Fu, 0xBEFE0404u, 0xBF810000u,
};
// N_window. As P2, but the window instruction is v_mov_b32 v5, 0 (a vector write).
const uint32_t kVectorInRestoreWindow[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u, 0xB0020005u, 0xBE800380u,
    0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u,
    0x7E0A0280u, 0xBEFE0404u, 0xBF810000u,
};
// N_clobber. As P1, with s_mov_b32 s5, 0 between the save and the narrowing compare.
const uint32_t kSavedMaskClobbered[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0xBE850380u, 0x7DA800A0u, 0xBF880009u, 0xB0020005u,
    0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu,
    0x7E060D01u, 0xBEFE0404u, 0xBF810000u,
};
// P3. As P1, plus buffer_load_dword v4, v0, s[8:11], 0 idxen inside the region; out = sum + v4.
const uint32_t kBoundedReadInRegion[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF88000Du, 0xE0302000u, 0x80020400u,
    0xBF8C3F70u, 0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u,
    0x80008100u, 0xBF82FFFBu, 0x4A020304u, 0x7E060D01u, 0xBEFE0404u, 0xBF810000u,
};
// N_store. As P3, but the region holds buffer_store_dword v0, v0, s[8:11], 0 idxen instead of the
// load, and the v_add that consumed the loaded value is dropped with it.
const uint32_t kStoreInRegion[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF88000Cu, 0xE0702000u, 0x80020000u,
    0xBF8C3F70u, 0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u,
    0x80008100u, 0xBF82FFFBu, 0x7E060D01u, 0xBEFE0404u, 0xBF810000u,
};
// C_store. As P1, with the same store placed after the restore (outside the guarded region).
const uint32_t kStoreAfterRestore[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u, 0xB0020005u, 0xBE800380u,
    0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u,
    0xBEFE0404u, 0xE0702000u, 0x80020000u, 0xBF810000u,
};

// P1r. As P1, plus v_add_f32 v3, 1.0, v3 after the restore: every lane must be live again there.
const uint32_t kGuardThenWriteAfterRestore[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u, 0xB0020005u, 0xBE800380u,
    0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u, 0xBF82FFFBu, 0x7E060D01u,
    0xBEFE0404u, 0x060606F2u, 0xBF810000u,
};
// Direct tests of mark_counted_loop_exec_guards. In a compute stage the emitter refuses these
// shapes anyway, so recompiling them cannot show which check fired; the proof is asked directly.
// D_window. As P2, plus `s_cmp_eq_u32 s1, 0; s_cbranch_scc1` jumping onto the restore (inside the
// window after the branch target). The guarded execz is at pc 6.
const uint32_t kBranchIntoRestoreWindow[] = {
    0x7E000F00u, 0xBF068001u, 0xBF85000Eu, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u,
    0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBF8CC07Fu, 0xBEFE0404u, 0xBF810000u,
};
// D_save. As P1, plus a branch onto the v_cmpx (between the save and the guarded execz at pc 6).
const uint32_t kBranchBetweenSaveAndNarrow[] = {
    0x7E000F00u, 0xBF068001u, 0xBF850002u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF880009u,
    0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE0404u, 0xBF810000u,
};
// D_smem. As P1, plus s_dcache_inv (SMEM 0x20, a memory writer to the classifier) in the region.
const uint32_t kScalarMemoryWriterInRegion[] = {
    0x7E000F00u, 0x7E060280u, 0xBE84047Eu, 0x7DA800A0u, 0xBF88000Bu, 0xF4800000u, 0x00000000u,
    0xB0020005u, 0xBE800380u, 0x7E020280u, 0xBF090200u, 0xBF850003u, 0x4A020200u, 0x80008100u,
    0xBF82FFFBu, 0x7E060D01u, 0xBEFE0404u, 0xBF810000u,
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

// The buffer at s[8:11] is the runner's `cbuf` (binding 2), one dword per lane.
ShaderResourceTable buffer_table() {
    ShaderResourceTable table;
    ShaderResource buffer{};
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 1;
    buffer.binding = 2;
    buffer.stride = 4;
    buffer.size = kLanes * 4;
    buffer.sgpr_base = 8;
    table.resources.push_back(buffer);
    return table;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N], const ShaderResourceTable* table = nullptr) {
    return recompile_valu(code, N, /*num_inputs*/1, /*out_vgpr*/3, table);
}

std::vector<float> lane_indices() {
    std::vector<float> input(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) input[lane] = static_cast<float>(lane);
    return input;
}

}  // namespace

TEST(CountedLoopCmpxGuard, AcceptsTheSaveThenNarrowForm) {
    EXPECT_FALSE(compile(kGuardAtTarget).empty())
        << "s_mov_b64 sN, exec ... v_cmpx -> s_cbranch_execz around a counted loop must lower";
    EXPECT_FALSE(compile(kRestoreAfterTarget).empty())
        << "the restore may follow the branch target by scalar-only instructions";
    const ShaderResourceTable table = buffer_table();
    EXPECT_FALSE(compile(kBoundedReadInRegion, &table).empty())
        << "a descriptor-bounded buffer read inside the guarded region is EXEC-predicated";
}

TEST(CountedLoopCmpxGuard, RefusesWhatTheGuardCannotProve) {
    EXPECT_TRUE(compile(kSavedMaskClobbered).empty())
        << "a write to the saved mask between save and narrow leaves the restore unproven";
    EXPECT_TRUE(compile(kVectorInRestoreWindow).empty())
        << "a vector write between the branch target and the restore is outside the guard";
    const ShaderResourceTable table = buffer_table();
    ASSERT_FALSE(compile(kStoreAfterRestore, &table).empty())
        << "control: the store itself must lower outside the region, or the next check is void";
    EXPECT_TRUE(compile(kStoreInRegion, &table).empty())
        << "a store inside the skipped region would run for a wave the guest skipped";
}

TEST(CountedLoopCmpxGuard, SwitchedOffLanesKeepTheirValueAndReturnAtTheRestore) {
    const std::vector<uint32_t> spv = compile(kGuardThenWriteAfterRestore);
    ASSERT_FALSE(spv.empty());
    const std::vector<float> got = prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    // Inside the guard only x < 32 add the loop's sum (10); after the restore every lane adds 1.
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], lane < 32 ? 11.0f : 1.0f) << "lane " << lane;
}

TEST(CountedLoopCmpxGuard, ProofRefusesEntryIntoTheGuardAndScalarMemoryWriters) {
    ASSERT_TRUE(guard_proven(kGuardAtTarget, 4)) << "control: the plain form is proven";
    ASSERT_TRUE(guard_proven(kRestoreAfterTarget, 4)) << "control: a restore after the target";
    EXPECT_FALSE(guard_proven(kBranchIntoRestoreWindow, 6))
        << "a branch landing between the target and the restore skips part of the window";
    EXPECT_FALSE(guard_proven(kBranchBetweenSaveAndNarrow, 6))
        << "a branch landing between the save and the branch reaches the narrow without the save";
    EXPECT_FALSE(guard_proven(kScalarMemoryWriterInRegion, 4))
        << "a scalar memory writer in the region would run for a wave the guest skipped";
}

TEST(CountedLoopCmpxGuard, ReadInsideTheRegionReachesTheActiveLanes) {
    const ShaderResourceTable table = buffer_table();
    const std::vector<uint32_t> spv = compile(kBoundedReadInRegion, &table);
    ASSERT_FALSE(spv.empty());
    std::vector<uint32_t> buffer(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) buffer[lane] = 1000u + lane;
    const std::vector<float> got =
        prosper::test::run_compute(spv, lane_indices(), kLanes, kLanes, buffer);
    if (got.empty()) GTEST_SKIP() << "no Vulkan compute device";
    for (uint32_t lane = 0; lane < kLanes; ++lane)
        EXPECT_FLOAT_EQ(got[lane], lane < 32 ? static_cast<float>(10u + 1000u + lane) : 0.0f)
            << "lane " << lane;
}
