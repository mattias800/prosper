// test_scc_branch_uniformity — a SCALAR branch may only be called workgroup-uniform when the value
// it compares is the same for every wave in the workgroup along every path.
//
// Why this exists (#3713). A generated compute uber-shader selects between barrier-separated
// variants with `s_cbranch_scc` on a constant read once from a constant buffer, so its `s_barrier`s
// sit inside SCC regions. `ForwardIf::uniform_workgroup` lets `top_level_pc` see through such a
// region, and that flag was only ever set for VCC branches — so those programs were rejected whole.
// `scc_branch_is_workgroup_uniform` now proves the scalar case.
//
// The failure mode this guards is SILENT. A branch wrongly called uniform lets a guest `s_barrier`
// sit inside a region only some waves enter, so waves wait for a barrier their peers never reach —
// wrong pixels or a hang, with no reject and no diagnostic. That is the same shape as #3596, which
// is why this file is built as PAIRS: every positive has a negative that must stay unproved, so the
// suite cannot pass by the proof simply never firing.
//
// Two of the negatives below were RED when this proof was first written, and both were found in
// review rather than by me:
//
//   * `s_bcnt1_i32_b64` was on the "steps over" list, borrowed from `rdna2_emit_cfg.cpp`'s
//     SCC-LIVENESS walks. That list answers "is SCC still SCALAR-VALUED", which an SCC writer can
//     satisfy; this proof asks "is SCC UNMODIFIED", which it cannot. `s_bcnt1_i32_b64` is
//     `D.i = CountOneBits(S0.u64); SCC = (D.i != 0)` (RDNA2 ISA 70648), so the walk stepped over a
//     ballot popcount of a lane mask and proved the branch from an older compare.
//
//   * The "sole definition in the program" rule had no dominance requirement. "Exactly one
//     definition, so the use saw that value or the untouched launch value, and both are uniform" is
//     false: each alternative is uniform alone, but the CHOICE between them need not be.
//
// WHAT THIS DOES NOT CLAIM. Not that the proof is sound in general — the obligations it rests on
// (EXEC full at the branch, operand kinds, vector writers, the depth cap) are unchanged and
// unexercised here. It establishes that the guest shape is admitted and that these specific
// counterexamples are not.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static std::vector<Rdna2Inst> walk(const uint32_t* code, size_t dwords) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code, dwords, ins);
    return ins;
}

int main() {
    printf("scc_branch_uniformity\n");

    // ---- the guest shape, positive -------------------------------------------------------------
    // `s_buffer_load_dword s16, s[8:11], null` at entry, compared against a literal, with
    // `s_mov_b64 exec, -1` sitting BETWEEN the compare and the branch — which is what the real
    // shader emits and what made four of its five guards unprovable before the walk stepped over
    // SCC-preserving SOP1s.
    const uint32_t uniform_guard[] = {
        0xF4200404u, 0xFA000000u,   // s_buffer_load_dword s16, s[8:11], null
        0xBF0B8110u,                // s_cmp_le_u32 s16, 1
        0xBEFE04C1u,                // s_mov_b64 exec, -1
        0xBF850001u,                // s_cbranch_scc1 +1
        0xBF800000u,                // s_nop
        0xBF810000u,                // s_endpgm
    };
    {
        const std::vector<Rdna2Inst> ins = walk(uniform_guard, std::size(uniform_guard));
        uint32_t branch_pc = 0;
        for (const auto& in : ins)
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x05u) branch_pc = in.pc;
        CHECK(branch_pc != 0, "positive fixture decodes an s_cbranch_scc1");
        CHECK(scc_branch_is_workgroup_uniform(ins, branch_pc),
              "a scalar branch on a constant-buffer load from launch SGPRs is workgroup-uniform");
    }

    // ---- negative: the SCC comes from a ballot popcount -----------------------------------------
    // An older, innocuous-looking `s_cmp_eq_i32 s5, 0` sits above a `s_bcnt1_i32_b64 s6, vcc`, which
    // writes SCC from a LANE MASK. Stepping over the popcount would prove the branch from the older
    // compare. The walk must stop at it.
    const uint32_t ballot_popcount_guard[] = {
        0x7D020300u,                // v_cmp_lt_i32 vcc, v0, v1  (any VOPC writing VCC)
        0xBF008005u,                // s_cmp_eq_i32 s5, 0       <- the decoy uniform compare
        0xBE86106Au,                // s_bcnt1_i32_b64 s6, vcc  <- writes SCC from the lane mask
        0xBEFE04C1u,                // s_mov_b64 exec, -1
        0xBF840001u,                // s_cbranch_scc0 +1
        0xBF800000u,
        0xBF810000u,
    };
    {
        const std::vector<Rdna2Inst> ins = walk(ballot_popcount_guard, std::size(ballot_popcount_guard));
        uint32_t branch_pc = 0;
        bool saw_bcnt = false;
        for (const auto& in : ins) {
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x04u) branch_pc = in.pc;
            if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x10u) saw_bcnt = true;
        }
        CHECK(saw_bcnt, "negative fixture really contains s_bcnt1_i32_b64 (the arm is not vacuous)");
        CHECK(!sop1_opcode_leaves_scc_unmodified(0x10u),
              "s_bcnt1_i32_b64 is not on the leaves-SCC-unmodified list");
        CHECK(!scc_branch_is_workgroup_uniform(ins, branch_pc),
              "a branch whose SCC is a popcount of a lane mask is NOT workgroup-uniform");
    }

    // ---- negative: the sole definition does not dominate the use --------------------------------
    // One definition of s16 in the whole program, and it sits inside a wave-varying branch arm. The
    // compare and the scalar branch are after the merge, so waves that skipped the arm compare the
    // untouched launch value while waves that took it compare the loaded one. Both values are
    // uniform; the CHOICE is not.
    const uint32_t non_dominating_guard[] = {
        0x7D020300u,                // v_cmp_lt_i32 vcc, v0, v1  (any VOPC writing VCC)
        0xBF860003u,                // s_cbranch_vccz -> the merge
        0xF4200404u, 0xFA000000u,   // s_buffer_load_dword s16, s[8:11], null   <- only in this arm
        0xBF800000u,                // s_nop
        0xBF0B8110u,                // s_cmp_le_u32 s16, 1                      <- merge
        0xBEFE04C1u,                // s_mov_b64 exec, -1
        0xBF850001u,                // s_cbranch_scc1 +1
        0xBF800000u,
        0xBF810000u,
    };
    {
        const std::vector<Rdna2Inst> ins = walk(non_dominating_guard, std::size(non_dominating_guard));
        uint32_t branch_pc = 0, vcc_branch_pc = 0;
        for (const auto& in : ins) {
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x05u) branch_pc = in.pc;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x06u) vcc_branch_pc = in.pc;
        }
        CHECK(vcc_branch_pc < branch_pc && branch_pc != 0,
              "negative fixture really places the load behind a wave-varying branch");
        CHECK(!scc_branch_is_workgroup_uniform(ins, branch_pc),
              "a scalar branch whose value is defined inside a wave-varying arm is NOT uniform");
    }

    // ---- positive that actually reaches sole_launch_derived_definition --------------------------
    // The first positive above does NOT exercise part 3 of this proof, and that was found in review:
    // with no branches at all, `block_start` walks to 0, so `last_writer` finds the load INSIDE the
    // branch's straight-line region and `uniform_scalar` takes its ordinary in-region SMEM case.
    // Replacing `sole_launch_derived_definition`'s body with `return false;` left every other
    // assertion in this file green — an arm sitting next to the property instead of on it.
    //
    // Reaching part 3 needs the definition OUTSIDE the branch's own block while still dominating it,
    // which is exactly the guest shape: the load sits before all control flow, and the compare that
    // consumes it sits in a later block whose start is a branch target. Here `s_cbranch_execz`
    // targets the compare, so `block_start` stops there and the load is out of region.
    const uint32_t out_of_region_guard[] = {
        0xF4200404u, 0xFA000000u,   // pc 0: s_buffer_load_dword s16, s[8:11], null  (sole def)
        0xBF880001u,                // pc 2: s_cbranch_execz -> pc 4
        0xBF800000u,                // pc 3: s_nop
        0xBF0B8110u,                // pc 4: s_cmp_le_u32 s16, 1   <- branch target: block starts here
        0xBEFE04C1u,                // pc 5: s_mov_b64 exec, -1
        0xBF850001u,                // pc 6: s_cbranch_scc1 -> pc 8
        0xBF800000u,                // pc 7: s_nop
        0xBF810000u,                // pc 8: s_endpgm
    };
    {
        const std::vector<Rdna2Inst> ins = walk(out_of_region_guard, std::size(out_of_region_guard));
        uint32_t branch_pc = 0, execz_pc = 0, load_pc = UINT32_MAX;
        for (const auto& in : ins) {
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x05u) branch_pc = in.pc;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08u) execz_pc = in.pc;
            if (in.fmt == Rdna2Format::SMEM && load_pc == UINT32_MAX) load_pc = in.pc;
        }
        // Without these the arm could pass while the fixture had drifted into the shape of the
        // first positive, which is how part 3 went untested in the first place.
        CHECK(load_pc < execz_pc && execz_pc < branch_pc && branch_pc != 0,
              "fixture really places the sole definition before all control flow, and the consuming "
              "compare in a later block");
        CHECK(scc_branch_is_workgroup_uniform(ins, branch_pc),
              "a sole launch-derived definition OUTSIDE the branch's block still proves uniform");
    }

    // ---- control: the load's ADDRESS is lane-local ----------------------------------------------
    // Same shape as the positive, dominating definition and all, except SBASE is produced by
    // `v_readfirstlane_b32` — an SGPR write carrying THIS lane's value. Without this arm the
    // positive could be passing because the address check never rejects anything.
    const uint32_t lane_local_base_guard[] = {
        0x7E100500u,                // v_readfirstlane_b32 s8, v0
        0xF4200404u, 0xFA000000u,   // s_buffer_load_dword s16, s[8:11], null
        0xBF0B8110u,                // s_cmp_le_u32 s16, 1
        0xBEFE04C1u,                // s_mov_b64 exec, -1
        0xBF850001u,                // s_cbranch_scc1 +1
        0xBF800000u,
        0xBF810000u,
    };
    {
        const std::vector<Rdna2Inst> ins = walk(lane_local_base_guard, std::size(lane_local_base_guard));
        uint32_t branch_pc = 0;
        bool saw_readfirstlane = false;
        for (const auto& in : ins) {
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x05u) branch_pc = in.pc;
            if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02u) saw_readfirstlane = true;
        }
        CHECK(saw_readfirstlane, "control fixture really contains v_readfirstlane_b32");
        CHECK(!scc_branch_is_workgroup_uniform(ins, branch_pc),
              "a scalar branch whose load address came from v_readfirstlane is NOT uniform");
    }

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
