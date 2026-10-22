// Existing VCC-preservation and recursive wave-uniform exit proof.
#include "gpu/recompiler/rdna2_cfg_wave_uniform.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace prosper::gpu {
namespace {
std::optional<bool> cfg_wave_scalar_operand(const Operand& operand,
                                            const std::vector<Rdna2Inst>& ins, uint32_t& use_pc) {
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
    return std::nullopt;
}

bool cfg_wave_operand_at(const std::vector<Rdna2Inst>& ins, const Rdna2Inst*& compare,
                         std::function<bool(const Operand&, uint32_t)>& uniform_operand_at,
                         const Operand& operand, uint32_t& use_pc) {
    if (auto result = cfg_wave_scalar_operand(operand, ins, use_pc)) return *result;
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
            pure_lane_alu =
                it->n_src == 2 && it->opcode != 0x01 && (it->opcode < 0x28 || it->opcode > 0x2a);
        }
        if (!pure_lane_alu) return false;
        for (uint32_t i = 0; i < it->n_src; ++i)
            if (!uniform_operand_at(it->src[i], it->pc)) return false;
        return true;
    }
    return false;
}

}   // namespace

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
bool cannot_write_vcc(const Rdna2Inst& in) {
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

bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
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
        return cfg_wave_operand_at(ins, compare, uniform_operand_at, operand, use_pc);
    };
    for (uint32_t i = 0; i < compare->n_src; ++i)
        if (!uniform_operand_at(compare->src[i], compare->pc)) return false;
    return compare->n_src != 0;
}

}   // namespace prosper::gpu
