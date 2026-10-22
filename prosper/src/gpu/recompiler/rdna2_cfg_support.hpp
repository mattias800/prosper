#pragma once
#include "gpu/recompiler/rdna2_cfg_scalar_merge.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_scalar_sources.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_forward_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_uniform_branch.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_divergent_detector.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_loop_admission.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_counted_loop.hpp" // IWYU pragma: export

// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that
// operate on them can live in their own translation units. These are INTERNAL to the
// recompiler: nothing outside src/gpu/recompiler/ should include this header.

// rdna2_cfg_support.hpp — helpers shared by rdna2_emit_cfg.cpp and rdna2_to_spirv.cpp.
// INTERNAL to src/gpu/recompiler/.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loaded_arbitrary.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_divergent_shapes.hpp" // IWYU pragma: export
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp" // IWYU pragma: export

namespace prosper::gpu {

uint32_t scalar_implicit_destination_read_width(const Rdna2Inst& in);
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source);
// The VALU sources that are lane masks: SRC2 of v_cndmask_b32_e64 and the VOP3B carry-in forms.
bool valu_lane_mask_source(const Rdna2Inst& in, uint32_t source);
// scalar_alu_source_words for a program of a known wave width: a Wave32 lane mask is ONE SGPR,
// where the width-free form charges every lane mask the Wave64 pair (#5202). Any other wave_size
// value (64, or unknown) keeps the pair.
uint32_t scalar_alu_source_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
// The words the source MARKS charge (scalar_source_marks and the dispatcher's
// spilled_vector_source_marks): the wave-aware width, except that a Wave32 VCC_LO lane mask still
// charges VCC_HI, the only witness the marks have of an unprovided entry VCC.
uint32_t scalar_source_mark_words(const Rdna2Inst& in, uint32_t source, uint32_t wave_size);
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
// Same proof, applied to a scalar branch's SCC instead of a VCC lane mask (#3713).
bool scc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);

// The header EXEC is a subset of its preheader only when every possible iteration preserves or
// narrows EXEC. This code-only certificate does not authenticate any carried register's data.
bool structured_loop_only_narrows_exec(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
bool narrowing_loop_exec_admission(const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                                   const RegState& entry);
// A wave-scalar word the loop [header, back edge] may write that is live at its exit: an SGPR,
// VCC or M0 word (a mask pair counts, as scalar data), or kLoopLiveScc. -1 when there is none.
// The narrowing certificate requires -1; rdna2_loop_wave_scalar_route.cpp routes the others.
constexpr int kLoopLiveScc = 253;
int loop_live_wave_scalar(const std::vector<Rdna2Inst>& ins, const DivLoop& loop);
// The launch and stream a fresh exact-dispatcher attempt of this program would compile: an
// authentic native Wave64 compute entry whose original words are `code`.
bool native_exact_retry_authentic(const SpirvCompute& b, const uint32_t* code, size_t dwords);
// A refused native store-loop header may select the existing exact dispatcher. This proves
// only eligibility to try a fresh builder, never DATA, scalar, resource or effect admission.
bool closed_compute_store_loop_data_retry(const SpirvCompute& b, const RegState& rs,
                                          const DivLoop& loop, const Rdna2Inst& header,
                                          uint32_t preheader_exec, const uint32_t* code,
                                          size_t dwords);
uint32_t make_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                      std::size_t& patch, bool admitted);
// The header EXEC phi is a subset of the preheader EXEC when every back edge carries a subset of
// it (induction over iterations). Two certificates grant that fact: the code-only compute one
// (narrowing_loop_exec_admission), and a checked one for fragment loops whose EXEC writes are
// all narrowing forms: assumed at the header, then verified at the back edge from the
// EXEC subset facts the body actually recorded. A failed check refuses the loop (#5039).
// Implemented in rdna2_loop_exec_subset.cpp.
class LoopExecSubset {
public:
    LoopExecSubset(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                   const RegState& entry);
    bool admitted() const { return certified_ || checked_; }
    void open(const RegState& preheader) {
        preheader_exec_ = preheader.exec;
        preheader_full_ = !preheader.exec_narrowed;
    }
    // At the back edge, with `rs` holding the body's final EXEC.
    bool holds(SpirvCompute& b, const RegState& rs) const;

private:
    bool certified_ = false;
    bool checked_ = false;
    uint32_t preheader_exec_ = 0;
    bool preheader_full_ = false;
};
// The EXEC phi at a one-armed structured if's merge. When the taken arm only narrowed the skipped
// edge's EXEC, the merge is a subset of it too (rdna2_loop_exec_subset.cpp).
uint32_t merge_exec_arm(SpirvCompute& b, RegState& rs, uint32_t skipped_exec,
                        uint32_t skipped_block, uint32_t taken_exec, uint32_t taken_block);
template <class PhiRecords>
void open_narrowing_loop_exec_phi(SpirvCompute& b, RegState& rs, uint32_t preheader,
                                  PhiRecords& phis, LoopExecSubset& proof) {
    std::size_t patch;
    proof.open(rs);
    const uint32_t phi = make_narrowing_loop_exec_phi(b, rs, preheader, patch, proof.admitted());
    phis.push_back({0, 4, phi, patch});
}

// A scalar VCCZ branch tests whether ANY active lane set VCC, not this lane's bit. Stages without
// fragment's forced wave64 vote may lower it as structured control only when the VCC producer is
// provably uniform across the wave. Dead Cells' light loops use the narrow compiler shape
//   v_cvt_i32_f32 vBOUND, sBOUND; ...; v_cmp_* vcc, sCOUNTER, vBOUND; s_cbranch_vccz EXIT
// so every active lane makes the same comparison. Unity also derives the bound through a short
// lane-local VALU chain fed only by scalar sources. Trace that chain backwards: uniform inputs stay
// uniform through ordinary VOP1/VOP2 arithmetic, while lane permutations, VCC-dependent selects,
// carry chains, interpolants, and unresolved VGPR inputs remain rejected.
// True when `in` provably cannot overwrite VCC — used by vcc_exit_is_wave_uniform's compare-finding
// walk (#590) to look past instructions scheduled between the VOPC and its consuming branch (real
// UE4 compute kernels hoist ~15 unrelated ALU ops in between). Anything not provably safe stops the
// walk conservatively: VOPC itself (the implicit dst), the VOP2 carry-chain ops (v_*_co_ci_u32,
// 0x28-0x2a), VOP3 windows that may carry a scalar dest (VOPC-as-VOP3 sdst at 0x000-0x13f, VOP3B
// carry-out / v_readlane at 0x300+ — the decoder does not expose their sdst), and any scalar or
// SMEM write whose dest range could cover SGPR 106/107 (the shaders use s106/s107 as scratch).
inline bool cannot_write_vcc(const Rdna2Inst& in) {
    auto sgpr_dst_misses_vcc = [&](uint32_t n_dwords) {
        if (in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) return true;
        const int lo = in.dst.value, hi = in.dst.value + (int)n_dwords - 1;
        return hi < 106 || lo > 107;
    };
    // The two lane reads write an SGPR although the decoder files their destination as a VGPR
    // operand (scalar_write_width knows them by opcode). Judge them by register number, or a
    // `v_readfirstlane_b32 vcc_lo` / `v_readlane_b32 vcc_lo` would pass as VCC-preserving (#4425).
    const auto lane_read_misses_vcc = [&] { return in.dst.value != 106 && in.dst.value != 107; };
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            if (in.opcode == 0x02) return lane_read_misses_vcc();   // v_readfirstlane_b32
            return true;
        case Rdna2Format::VINTRP:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::DS:
        case Rdna2Format::FLAT:
        case Rdna2Format::EXP:   // VGPR/memory dsts only
        case Rdna2Format::SOPP:   // branches/hints/waitcnt: no dst
        case Rdna2Format::SOPC: return true;   // writes SCC only
        case Rdna2Format::VOP2: return in.opcode < 0x28 || in.opcode > 0x2a;
        case Rdna2Format::VOP3P: return true;   // packed VALU: VGPR dst, no carry
        case Rdna2Format::VOP3:
            // gfx10 v_writelane_b32 writes only its VGPR lane slot. v_readlane_b32 writes the
            // scalar destination encoded in the decoder's dst field, so it is VCC-preserving as
            // long as that destination does not overlap s106:s107. UE4 schedules long scalar-spill
            // sequences of these two ops between a uniform VOPC and its s_cbranch_vccz.
            if (in.opcode == 0x361) return true;
            if (in.opcode == 0x360) return lane_read_misses_vcc();   // v_readlane_b32
            // Decoder storage has an SDST-shaped field for every VOP3 packet, but only VOP3B
            // operations architecturally write it. Treat ordinary VOP3A VALU as VCC-transparent;
            // otherwise a MAC between a uniform compare and VCCZ creates a false Wave64 vote.
            if (vop3_writes_mask_sdst(in)) return in.sdst.value != 106 && in.sdst.value != 107;
            return in.dst.kind == OperandKind::VGPR;
        case Rdna2Format::SOP1:
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPK: return sgpr_dst_misses_vcc(2);   // conservative 64-bit pair
        case Rdna2Format::SMEM: {
            // What SDATA receives, from the effects table (this used to read the width from
            // opcode & 7, which charged s_memtime sixteen SGPRs and every opcode 5-7 mod 8 none).
            if (!rdna2_instruction_effects(in).known()) return false;
            const uint32_t n = rdna2_smem_destination_dwords(in);
            return !n || sgpr_dst_misses_vcc(n);
        }
        default: return false;   // VOPC / unknown: stops the walk
    }
}

inline bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    const Rdna2Inst* compare = nullptr;
    for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
        if (it->pc >= branch_pc) continue;
        if (sopp_is_noop(*it)) continue;
        if (it->fmt == Rdna2Format::VOPC) {
            compare = &*it;
            break;
        }
        if (cannot_write_vcc(*it)) continue;   // (#590) look past provably-VCC-preserving instrs —
        break;   // VCC's value is frozen at the compare; only a real
        // rewrite between compare and branch invalidates it
    }
    if (!compare || compare->fmt != Rdna2Format::VOPC || vopc_is_cmpx(compare->opcode))
        return false;

    std::function<bool(const Operand&, uint32_t)> uniform_operand_at;
    uniform_operand_at = [&](const Operand& operand, uint32_t use_pc) -> bool {
        if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) {
            // "Lives in a scalar register" is NOT the same as "is wave-uniform" (#3607).
            // `v_readfirstlane_b32` writes an SGPR while producing THIS lane's value under prosper's
            // per-lane model -- its lowering is annotated SPECULATIVE(confidence: med) in
            // rdna2_emit_alu.cpp for exactly that reason -- so accepting the register unconditionally
            // lets a lane-varying value satisfy a wave-uniformity proof.
            //
            // `Special` is in this branch and NOT with the constants below, which is the half a
            // first version of this guard got wrong. `decode_src_field` maps 0..105 to `SGPR` but
            // sends VCC_LO/HI (106/107), EXEC, M0 and ttmp to `Special` -- so screening `SGPR` alone
            // left `v_readfirstlane_b32 s106, v1` writing VCC_LO and then satisfying the proof
            // through the very branch that used to assert "scalar-register sources (including VCC
            // halves) broadcast one wave value". This file's own comments note the shaders use
            // s106/s107 as scratch, so that is a reachable shape, and #3596's emitter-side sibling
            // already screens `Special` for exactly this reason. Matching on `value` works uniformly
            // across both kinds because the destination carries the raw register number.
            //
            // Refuse when any earlier instruction read this register out of a lane that way. Two
            // notes on the shape, both load-bearing:
            //
            //  - The match is on `dst.value`, NOT on `dst.kind`. The decoder builds every VOP1 dst as
            //    `vgpr(w >> 17)` (rdna2_decode.cpp:269), so this instruction's dst reports kind VGPR
            //    while its VALUE is the SGPR number -- which is what `rdna2_vgpr_write_count` returning
            //    0 here and the emitter's `rs.sreg[in.dst.value]` both rely on.
            //  - It does NOT stop at a later redefinition, so an SGPR that readfirstlane wrote and a
            //    subsequent `s_mov_b32` made uniform again keeps being refused. That is deliberate:
            //    the only way to stop at a redefinition is a "this instruction writes SGPR n"
            //    predicate, which does not exist here and would have to enumerate every SGPR-writing
            //    format by hand. A format missed by such a predicate would miss a *readfirstlane*
            //    write and return true -- the unsafe direction. Over-refusal costs an optimisation and
            //    cannot produce a wrong answer, because every caller treats false as "not proven
            //    uniform" and falls back to the general path.
            //
            // Deliberately narrow otherwise: anything else keeps the previous answer. The wider
            // question -- whether ANY scalar reached through arbitrary ALU is uniform -- is #3606.
            for (const auto& prior : ins)
                if (prior.pc < use_pc && prior.fmt == Rdna2Format::VOP1 && prior.opcode == 0x02u &&
                    prior.dst.value == operand.value)
                    return false;
            return true;
        }
        if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true;   // a constant is uniform by construction and nothing can write it
        if (operand.kind != OperandKind::VGPR) return false;
        for (auto it = ins.rbegin(); it != ins.rend(); ++it) {
            if (it->pc >= use_pc) continue;
            if (operand.value == rdna2_tfe_status_vgpr(*it)) return false;
            const uint32_t writes = rdna2_vgpr_write_count(*it);
            if (!writes || operand.value < it->dst.value ||
                operand.value >= it->dst.value + (int32_t)writes)
                continue;
            if (it->dst.value != operand.value || it->has_dpp) return false;
            // The write makes every currently active lane uniform. Keep the original conservative
            // EXEC rule: any later mask change could expose a lane that retained an older value.
            for (const auto& between : ins)
                if (between.pc > it->pc && between.pc < compare->pc &&
                    rdna2_instruction_may_change_exec(between))
                    return false;

            bool pure_lane_alu = false;
            if (it->fmt == Rdna2Format::VOP1) {
                pure_lane_alu = it->n_src == 1;
            } else if (it->fmt == Rdna2Format::VOP2) {
                // v_cndmask consumes VCC; the carry family consumes/produces VCC. Neither preserves
                // scalar-source uniformity as a simple lane-local expression.
                pure_lane_alu = it->n_src == 2 && it->opcode != 0x01 &&
                                (it->opcode < 0x28 || it->opcode > 0x2a);
            }
            if (!pure_lane_alu) return false;
            for (uint32_t i = 0; i < it->n_src; ++i)
                if (!uniform_operand_at(it->src[i], it->pc)) return false;
            return true;
        }
        return false;
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand_at(compare->src[i], compare->pc)) return false;
    return compare->n_src != 0;
}

// ---------------------------------------------------------------------------------------------
// Workgroup-uniform wave branch (#1554).
//
// A compute `s_cbranch_vccz`/`vccnz` decides per WAVE, so two waves of one workgroup may take
// different arms. A guest `S_BARRIER` inside such an arm violates Vulkan's uniform-participation
// requirement for OpControlBarrier, so detect_forward_ifs rejects that combination.
//
// Real kernels nevertheless build some of these masks entirely out of launch state:
//
//     s_load_dwordx4        s[8:11], s[12:13], 0x20    ; s12:s13 is entry data
//     s_buffer_load_dwordx2 s[16:17], s[8:11], 0xc
//     s_add_i32             vcc_lo, s17, -1            ; VCC_LO used as scalar scratch
//     s_cmp_gt_u32          s16, 0
//     s_cselect_b64         s[10:11], exec, 0
//     s_cmp_eq_u32          s14, vcc_lo
//     s_cselect_b64         vcc, exec, 0
//     s_and_b64             vcc, vcc, s[10:11]
//     s_cbranch_vccz        END
//
// With EXEC full, VCC is nonzero exactly when the scalar predicate holds, and that predicate reads
// only compute launch SGPRs. recompile_compute seeds those from push-constant user data and from
// WorkGroupId system SGPRs (see the entry seeding in recompile_compute), both of which are identical
// for every wave of a workgroup — so the branch is workgroup-uniform and its arm may synchronize.
//
// Two obligations are proved independently and mechanically:
//
//   (1) EXEC is full at the branch on EVERY path. Without it, VCCZ is also true for a wave whose
//       EXEC happens to be empty, which is a genuinely per-wave property. A forward must-dataflow
//       proves this, so a loop that restores `s_mov_b64 exec, -1` before its back-edge still
//       qualifies while an arm that narrows EXEC does not.
//   (2) VCC is `SCC ? EXEC : 0`, optionally combined by s_and_b64/s_or_b64; every contributing SCC
//       comes from a scalar compare, and every scalar input traces to launch data.
//
// Everything not proved stays rejected: EXEC or SCC read as data, V_READFIRSTLANE/V_READLANE,
// VOPC-written masks, SGPRs above the launch range, and any definition reaching the branch from
// outside its own straight-line region unless it provably still holds its entry value.
//
// CONFIDENCE: HIGH — the accepted provenance set is closed and every rejection is fail-visible.

// Forward "must" dataflow over the decoded stream. `fact[i]` is true only when the tracked property
// holds at entry to ins[i] along EVERY path that reaches it. `gen` re-establishes the property,
// `kill` destroys it, and anything else passes it through. Optimistic initialization followed by
// iteration to a fixpoint is the standard greatest-fixpoint formulation, which is what makes the
// result sound across back-edges. Returns an empty vector when the stream contains an edge this walk
// cannot resolve, so every caller treats "unanalyzable" as "not proved".
template <typename Gen, typename Kill>
std::vector<uint8_t> must_fact_at(const std::vector<Rdna2Inst>& ins, Gen&& gen, Kill&& kill,
                                  bool entry_holds) {
    if (ins.empty()) return {};
    std::unordered_map<uint32_t, size_t> index_of;
    for (size_t i = 0; i < ins.size(); ++i) index_of[ins[i].pc] = i;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };

    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) continue;
        if (is_branch(in)) {
            const uint32_t target = branch_target(in);
            auto found = index_of.find(target);
            if (found != index_of.end())
                succ[i].push_back(found->second);
            else if (target <= ins.back().pc)
                return {};   // an edge this walk cannot resolve
            // A target past the decoded stream is an early-out that terminates the wave; it
            // contributes no successor inside this function.
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x02)
            continue;   // s_branch: no fallthrough
        if (i + 1 < ins.size()) succ[i].push_back(i + 1);
    }

    std::vector<uint8_t> fact(ins.size(), 1);
    if (!entry_holds) fact[0] = 0;
    // Each round can only turn a 1 into a 0, so the fixpoint is reached in at most size()+1 rounds.
    for (size_t round = 0; round <= ins.size() + 1; ++round) {
        std::vector<uint8_t> next(ins.size(), 1);
        if (!entry_holds) next[0] = 0;
        for (size_t i = 0; i < ins.size(); ++i) {
            uint8_t out = fact[i];
            if (gen(ins[i]))
                out = 1;
            else if (kill(ins[i]))
                out = 0;
            if (out) continue;
            for (size_t s : succ[i]) next[s] = 0;
        }
        if (next == fact) return fact;
        fact.swap(next);
    }
    return fact;
}

// `s_mov_b64 exec, -1` is the only accepted full-mask definition. A 32-bit literal written to a
// 64-bit scalar destination is zero-extended, so it activates only the low half and is NOT a full
// wave64 mask; the inline -1 constant is sign-extended and is.
inline bool exec_write_sets_full_mask(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.n_src != 1) return false;
    if ((in.dst.kind != OperandKind::SGPR && in.dst.kind != OperandKind::Special) ||
        in.dst.value != 126)
        return false;
    return in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1;
}

inline bool reg_operand(const Operand& operand, int reg) {
    return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
           operand.value == reg;
}

inline uint32_t scalar_branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + static_cast<uint32_t>(in.simm16);
}

// Arbitrary CFG fallback. Some UE4 volume-lighting kernels and Astro Bot material shaders contain
// nested EXEC loops plus scalar/VCC branches. They are valid reducible machine CFGs, but deliberately
// exceed the narrow pattern structurizer below. Lower them as a structured dispatcher loop: one
// switch case per basic block, with the emulated register file persisted in Function variables
// between iterations. Graphics stages can branch directly on this invocation's SCC/VCC/EXEC bit;
// compute needs the guest-wave reduction described below.
// VCCZ/EXECZ reduce over the dispatch's 32/64-lane hardware wave, not the host Vulkan subgroup.
// Native subgroup widths are implementation-defined (llvmpipe is commonly 8), so every invocation
// remains in the dispatcher until the whole workgroup is done and exchanges its vote at the common
// switch merge. Dedicated Workgroup scratch keeps multiple hardware waves independent. The fallback
// is gated by emit_body to complex compute CFGs without guest barriers. Ordinary LDS reads, writes,
// and atomics remain valid while waves visit different cases. V_MBCNT is split into a dedicated
// common phase so every workgroup invocation reaches its synthesized barriers in uniform control flow.
// PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Bound the CFG DISPATCHER's loop at N iterations, on
// whichever back-edge it emits (direct or portable). PROSPER_CFG_TRIP_BOUND_PROGRAM selects one
// program; PROSPER_CFG_TRIP_BOUND_PHASE selects one phase and is REQUIRED — see
// emitted_loop_trip_bound for why a record spanning two phases has no true reading. A program that
// accesses GDS itself is refused, because the witness lives in guest-addressable storage.
//
// It does NOT cover the two structured loop emitters. An earlier revision of this comment claimed it
// did; it never called this helper from either, so the claim was false and would have made a null
// result from a structured-loop program read as evidence.
//
// A loop that never terminates hangs the GPU into a driver reset, which costs the whole process its
// compute backend and every later indirect draw — and from outside it is indistinguishable from a
// slow shader or from a defect anywhere else in the submit. Bounding the dispatcher's loop turns
// "is this a non-terminating loop at all?" into one run for the programs it covers.
//
// (This paragraph used to end "and covering all three emitters means the answer is not confined to
// one lowering path" — the very claim the paragraph above corrects. The correction landed two lines
// up and the boast survived underneath it, which is how a retracted statement keeps being read as
// current. `tests/shared/diagnostics/test_cfg_trip_bound.cpp` now pins both halves: a structured loop is unchanged when
// armed, a dispatcher loop is not.)
//
// Unset, nothing is emitted and modules are byte-identical. It is NOT a fix: truncating a guest
// program's control flow produces wrong results by construction, which is why it is opt-in and says
// so when it arms.
// See kComputeTripWitnessDword. Deliberately ignores the phase selector: if ANY phase of this program
// will be bounded, the witness destination has to be bound for the whole dispatch.
// The complete selector state, read fresh rather than cached in function-local statics.
//
// Two reasons it is a struct and not three scattered getenv sites. First, the shader cache is keyed
// on the program's CODE BYTES and never on its address, so a target and a non-target with the same
// body collide on one entry: whichever compiled first would serve the other, handing the bounded
// module to an excluded program or the unbounded module to the target. Either direction silently
// destroys the isolation that makes a bounded run evidence about ONE program. The cache key therefore
// mixes this whole struct in (see gpu_executor.cpp), and a struct is what makes that complete by
// construction — a future selector added here cannot be forgotten there.
//
// Second, function-local statics parse the environment exactly once per process, which makes the
// selectors untestable in-process: no test could arm a bound, assert, then disarm and assert again.
// So this parses on demand, and `TripBoundOperation` (rdna2_to_spirv.hpp) is what keeps one
// operation's reads coherent without freezing them for the process.
//
// Work counter for the operation scope. See trip_bound_parses().
inline std::atomic<uint64_t>& trip_bound_parse_counter() {
    static std::atomic<uint64_t> parses{0};
    return parses;
}

// The innermost TripBoundOperation's owned settings, or null when none is open on this thread.
// Per thread because compilation runs on whatever worker realized the draw.
inline const ComputeTripBoundSettings*& trip_bound_pin() {
    static thread_local const ComputeTripBoundSettings* pinned = nullptr;
    return pinned;
}

// THE COMMENT HERE USED TO SAY the re-reads happen "during RECOMPILATION only (never per draw or
// per dispatch — cf. #2214)", with CONFIDENCE: HIGH. That was wrong, and wrong in the way this
// project's charter warns about most: a true-sounding claim wearing a citation to the very issue
// about removing per-draw getenv, in a place nobody re-derives. Three of the callers are not in the
// recompiler at all — gpu_executor.cpp builds the shader-cache KEY from these settings, so they are
// read on every cache HIT as well as every miss, i.e. once or twice per draw. Measured on a 240 s
// routed Grand Theft Auto V window with the switch unset (#3714): 3,989,848 reads of
// PROSPER_CFG_TRIP_BOUND at `ec02a77f6`, ~16,600/s. That is nominally the largest remaining getenv
// name, but it leads PROSPER_GFXLOG by 0.39% while GFXLOG itself varies by 35,143 between two runs
// of the same route -- so read the two as jointly largest, not ranked. An unset name is the
// expensive case on glibc -- a full scan of environ -- so those are all misses.
//
// The body. `parse_trip_bound_settings()` and `compute_trip_bound_settings()` are declared in the
// public header and defined ONCE in rdna2_to_spirv.cpp on top of this. The reason is
// [basic.def.odr]/11: an `inline` function must be defined in every translation unit that uses it,
// and this header is internal to src/gpu/recompiler/, so a TU outside it -- the new
// `test_trip_bound_operation` -- cannot legally see a definition at all.
//
// Do NOT restate that as "no out-of-line copy survives". It does: on `ec02a77f6` gpu_executor.cpp
// calls `compute_trip_bound_settings()` through the public declaration WITHOUT including this
// header, and links. What was actually observed is narrower: linking test_trip_bound_operation
// against libprosper_core.a failed with `undefined reference to
// 'prosper::gpu::compute_trip_bound_settings()'` -- the PRE-EXISTING function, not any of the
// symbols this change adds, which is the only reason the observation bears on the inline question
// at all. The mechanism for it was not established and is deliberately not guessed at here.
inline ComputeTripBoundSettings parse_trip_bound_settings_impl() {
    trip_bound_parse_counter().fetch_add(1, std::memory_order_relaxed);
    ComputeTripBoundSettings settings;
    const char* spec = getenv("PROSPER_CFG_TRIP_BOUND");
    if (!spec || !*spec) return settings;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(spec, &end, 0);
    if (!end || *end || !parsed || parsed > 0xffffffffull) return settings;
    settings.bound = static_cast<uint32_t>(parsed);

    // PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR — restrict the bound to ONE program, leaving every other
    // recompiled module byte-identical. Without it, a bound low enough to be interesting truncates
    // MANY shaders, and any later change of behaviour has a second explanation: an earlier truncated
    // shader fed different data downstream. Targeting one program removes that alternative, which is
    // what turns "the run got further" into evidence about the program under test.
    if (const char* only = getenv("PROSPER_CFG_TRIP_BOUND_PROGRAM"); only && *only) {
        char* only_end = nullptr;
        const unsigned long long wanted = strtoull(only, &only_end, 0);
        if (only_end && !*only_end) settings.only_program = wanted;
    }
    // PROSPER_CFG_TRIP_BOUND_PHASE=K — bound only the K-th dispatcher of the selected program.
    //
    // A barrier-phased compute program emits one dispatcher per barrier-free phase, each covering a
    // different guest pc range. Bounding them together answers "does some loop in this program run
    // away" but not which, and they are not equivalent: an acyclic phase that needs a bound is a CFG
    // state-transition defect, while the phase containing the guest's own loop might merely be slow.
    if (const char* phase = getenv("PROSPER_CFG_TRIP_BOUND_PHASE"); phase && *phase) {
        char* phase_end = nullptr;
        const unsigned long long parsed_phase = strtoull(phase, &phase_end, 0);
        if (phase_end && !*phase_end && parsed_phase <= 0xfffffffeull)
            settings.only_phase = static_cast<uint32_t>(parsed_phase);
    }
    // PROSPER_CFG_TRIP_BOUND_ORDINAL=K — count only traversals about to dispatch ordinal K.
    // See ComputeTripBoundSettings::only_ordinal for why a per-loop cap is the thing that localizes
    // a runaway inside a program whose dispatcher serves several guest loops (#3193).
    if (const char* ordinal = getenv("PROSPER_CFG_TRIP_BOUND_ORDINAL"); ordinal && *ordinal) {
        char* ordinal_end = nullptr;
        const unsigned long long parsed_ordinal = strtoull(ordinal, &ordinal_end, 0);
        if (ordinal_end && !*ordinal_end && parsed_ordinal <= 0xfffffffeull)
            settings.only_ordinal = static_cast<uint32_t>(parsed_ordinal);
    }
    return settings;
}

struct BarrierPhasedCompute {
    size_t guard_index = 0;
    size_t end_index = 0;
    std::vector<size_t> barriers;
    bool guarded = false;
    bool found = false;
};

// Prove the SCC consumed by a terminal compute guard is identical for every wave in the workgroup,
// while allowing unrelated lane-local setup to precede it. The legacy barrier-phase proof required
// every instruction before the guard to be scalar; generated kernels commonly interleave vector
// address setup with a scalar descriptor load that independently produces the guard condition.
//
// This is deliberately a narrow backwards slice. The last SCC writer must be an SOPC, and every
// scalar word feeding that compare must trace through launch SGPRs, scalar ALU, or scalar memory.
// V_READFIRSTLANE/V_READLANE, vector mask writes, EXEC, and unresolved special registers terminate
// the proof. Existing all-scalar prefixes retain their established admission path below.
inline bool terminal_guard_scc_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins,
                                                    size_t guard_index) {
    // This is a textual backwards slice, not a reaching-definition analysis. Keep the prefix
    // branch-free so every scalar definition it finds dominates the terminal guard. In particular,
    // a VCC/EXEC branch may vary between guest waves even when the literal or SMEM value written in
    // one arm is itself uniform.
    for (size_t i = 0; i < guard_index; ++i) {
        const Rdna2Inst& prefix = ins[i];
        if ((prefix.fmt == Rdna2Format::SOPP && !sopp_is_noop(prefix)) ||
            (prefix.fmt == Rdna2Format::SOP1 && prefix.opcode >= 0x20u && prefix.opcode <= 0x22u))
            return false;
    }

    std::function<bool(int, uint32_t, uint32_t)> uniform_reg_at;
    auto operand_uniform_at = [&](const Operand& operand, uint32_t words, uint32_t use_pc,
                                  auto&& self) -> bool {
        switch (operand.kind) {
            case OperandKind::InlineInt:
            case OperandKind::InlineFloat:
            case OperandKind::Literal: return true;
            case OperandKind::Special:
                if (operand.value == 125) return true;   // SGPR_NULL
                if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                    return false;   // EXEC and SCC are not scalar launch data.
                [[fallthrough]];
            case OperandKind::SGPR:
                for (uint32_t word = 0; word < words; ++word)
                    if (!self(operand.value + static_cast<int>(word), use_pc, 0)) return false;
                return true;
            default: return false;
        }
    };

    uniform_reg_at = [&](int reg, uint32_t use_pc, uint32_t depth) -> bool {
        if (depth > 64) return false;
        for (size_t i = ins.size(); i-- > 0;) {
            const Rdna2Inst& writer = ins[i];
            if (writer.pc >= use_pc) continue;
            bool writes = false;
            for_each_scalar_write(writer, [&](int base, uint32_t width) {
                writes |= reg >= base && reg < base + static_cast<int>(width);
            });
            // Implicit VOPC writes target architectural VCC even when the decoder exposes it as a
            // Special operand rather than an ordinary SGPR destination.
            if ((reg == 106 || reg == 107) && writer.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(writer.opcode))
                writes = true;
            if (!writes) continue;

            auto uniform_operand = [&](const Operand& operand, uint32_t words) {
                switch (operand.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal: return true;
                    case OperandKind::Special:
                        if (operand.value == 125) return true;
                        if (operand.value == 126 || operand.value == 127 || operand.value == 253)
                            return false;
                        [[fallthrough]];
                    case OperandKind::SGPR:
                        for (uint32_t word = 0; word < words; ++word)
                            if (!uniform_reg_at(operand.value + static_cast<int>(word), writer.pc,
                                                depth + 1))
                                return false;
                        return true;
                    default: return false;
                }
            };

            switch (writer.fmt) {
                case Rdna2Format::SOP1: {
                    // GETPC is identical for every wave. Mask/saveexec operations consume EXEC and
                    // are rejected by rdna2_instruction_may_change_exec; ordinary scalar operations trace
                    // their source at the destination's architectural width.
                    if (writer.opcode == 0x1f) return true;
                    if (rdna2_instruction_may_change_exec(writer) || writer.n_src != 1)
                        return false;
                    return uniform_operand(writer.src[0],
                                           scalar_write_width(writer) == 2 ? 2u : 1u);
                }
                case Rdna2Format::SOP2: {
                    // Carry/cselect forms consume SCC in addition to their decoded operands. They are
                    // outside this narrow slice until SCC itself is represented as a recursive value.
                    if (writer.opcode == 0x04 || writer.opcode == 0x05 || writer.opcode == 0x0a ||
                        writer.opcode == 0x0b)
                        return false;
                    const uint32_t words = scalar_write_width(writer) == 2 ? 2u : 1u;
                    return writer.n_src == 2 && uniform_operand(writer.src[0], words) &&
                           uniform_operand(writer.src[1], words);
                }
                case Rdna2Format::SMEM: {
                    // Scalar-buffer loads use a four-dword descriptor; scalar-memory loads use a
                    // base pair. The scalar offset is one dword in both forms. Only a load's
                    // result is a function of its address operands.
                    if (!rdna2_smem_load_dwords(writer.opcode) || writer.n_src < 1) return false;
                    const uint32_t base_words = rdna2_smem_address_dwords(writer.opcode);
                    if (!uniform_operand(writer.src[0], base_words)) return false;
                    return writer.n_src < 2 || uniform_operand(writer.src[1], 1);
                }
                case Rdna2Format::SOPK:
                    // S_MOVK is a literal definition. Other SOPK data writers are read/modify/write
                    // or hardware-register reads and stay outside this proof.
                    return writer.opcode == 0x00;
                default: return false;
            }
        }
        // Ordinary compute launch SGPRs (user data and enabled system SGPRs) are uniform within one
        // workgroup. Special registers are numbered above this range and were handled explicitly.
        return reg >= 0 && reg <= 105;
    };

    for (size_t i = guard_index; i-- > 0;) {
        const Rdna2Inst& candidate = ins[i];
        if (sopp_is_noop(candidate)) continue;
        if (candidate.fmt == Rdna2Format::SOPC) {
            const uint32_t words = candidate.opcode == 0x12 || candidate.opcode == 0x13 ? 2u : 1u;
            for (uint32_t source = 0; source < candidate.n_src; ++source)
                if (!operand_uniform_at(candidate.src[source], words, candidate.pc, uniform_reg_at))
                    return false;
            return candidate.n_src != 0;
        }
        // Vector/memory operations cannot write SCC. Any scalar ALU or SOPK instruction might, so
        // stop rather than accidentally consuming an older compare through an unmodeled SCC writer.
        if (candidate.fmt == Rdna2Format::SOP1 || candidate.fmt == Rdna2Format::SOP2 ||
            candidate.fmt == Rdna2Format::SOPK)
            return false;
    }
    return false;
}

// Recognize barrier-separated compute phases. A generated kernel may wrap the phases in one proven
// workgroup-uniform terminal SCC guard, but the guard is not essential: an unconditional top-level
// barrier is itself a valid phase boundary. In either form, every explicit scalar edge must stay in
// one phase and indirect PC changes fail closed. Keeping this proof shared is important: emit_body
// uses it to split the shader, while the native subgroup policy uses the same result to make nested
// wave votes legal inside an individual phase.
inline BarrierPhasedCompute analyze_barrier_phased_compute(const std::vector<Rdna2Inst>& ins) {
    BarrierPhasedCompute result;
    result.end_index = ins.size();
    uint32_t end_pc = UINT32_MAX;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!ins[i].is_end) continue;
        result.end_index = i;
        end_pc = ins[i].pc;
        break;
    }

    result.guard_index = ins.size();
    size_t first_barrier = ins.size();
    size_t branch_count = 0;
    for (size_t i = 0; i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a && first_barrier == ins.size())
            first_barrier = i;
        if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03)
            continue;
        ++branch_count;
        if (result.guard_index == ins.size() && (in.opcode == 0x04 || in.opcode == 0x05) &&
            branch_target(in) == end_pc)
            result.guard_index = i;
    }

    bool guarded = result.guard_index < first_barrier && first_barrier < result.end_index;
    bool scalar_prefix = guarded;
    for (size_t i = 0; scalar_prefix && i < result.guard_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // Scalar ALU/loads retain workgroup-uniform values when entered from compute user data and
        // TGIDs. EXEC is lane state, so any explicit access to either half invalidates this proof.
        // VCC physical words may be scalar scratch provided no vector instruction enters the prefix.
        const bool scalar = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC ||
                            in.fmt == Rdna2Format::SMEM ||
                            (in.fmt == Rdna2Format::SOPP && sopp_is_noop(in));
        if (!scalar) {
            scalar_prefix = false;
            break;
        }
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
                   (operand.value == 126 || operand.value == 127);
        };
        if (is_exec(in.dst)) scalar_prefix = false;
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (is_exec(in.src[source])) scalar_prefix = false;
    }
    if (guarded && !scalar_prefix)
        guarded = terminal_guard_scc_is_workgroup_uniform(ins, result.guard_index);

    // If a candidate terminal guard was not proved uniform, include it in the ordinary edge proof.
    // Since it jumps across every barrier to S_ENDPGM, that proof rejects the conditional barrier.
    // A proved guard is deliberately excluded: every invocation in the workgroup takes it together.
    result.guarded = guarded;
    const size_t body_begin = guarded ? result.guard_index + 1 : 0;
    // Safety comes from the cross-edge, trap and EXEC-save checks below. A straight-line kernel
    // whose only control-flow instruction is a barrier satisfies that proof trivially; requiring
    // an unrelated branch-count threshold dropped GTA V's partial-workgroup barrier passes.
    bool valid = result.end_index < ins.size();

    for (size_t i = body_begin; valid && i < result.end_index; ++i)
        if (ins[i].fmt == Rdna2Format::SOPP && ins[i].opcode == 0x0a) result.barriers.push_back(i);
    valid = valid && !result.barriers.empty();

    for (size_t i = body_begin; valid && i < result.end_index; ++i) {
        const Rdna2Inst& in = ins[i];
        // A trap deactivates its guest wave in the dispatcher. Emitting a host workgroup barrier
        // afterward would then require an invocation that has left the phase to participate.
        // Final-phase traps are fine because no outer barrier follows them.
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12 && i < result.barriers.back())
            valid = false;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) valid = false;
    }
    // A phase boundary is valid only when no guest branch jumps across it in either direction.
    for (size_t barrier_index : result.barriers) {
        const uint32_t barrier_pc = ins[barrier_index].pc;
        for (size_t i = body_begin; valid && i < result.end_index; ++i) {
            const Rdna2Inst& in = ins[i];
            if (in.fmt != Rdna2Format::SOPP || in.opcode < 0x02 || in.opcode > 0x09 ||
                in.opcode == 0x03 || in.opcode == 0x0a)
                continue;
            const uint32_t target = branch_target(in);
            if ((in.pc < barrier_pc && target >= barrier_pc) ||
                (in.pc > barrier_pc && target <= barrier_pc))
                valid = false;
        }
    }
    result.found = valid;
    return result;
}

}   // namespace prosper::gpu
