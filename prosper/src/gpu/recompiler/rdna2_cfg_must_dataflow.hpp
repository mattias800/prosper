// Must-fact dataflow and the full-EXEC-mask predicate for CFG proofs.
#pragma once
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {

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

bool exec_write_sets_full_mask(const Rdna2Inst& in);
}   // namespace prosper::gpu
