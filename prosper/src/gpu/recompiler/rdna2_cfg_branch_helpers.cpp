// Shared CFG branch helpers moved from rdna2_cfg_support.hpp.
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
bool sopp_is_noop(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPP) return false;
    switch (in.opcode) {
        case 0x00:   // s_nop
        case 0x0c:   // s_waitcnt
        case 0x20:   // s_inst_prefetch
        case 0x21:   // s_clause
        case 0x22:   // s_wait_idle
            // NOTE: s_waitcnt_vscnt is NOT SOPP on gfx10 — it is SOPK opcode 0x17
            // (round-trip: llvm-mc gfx1010 encodes it 0xBBFD0000). Handled in the SOPK case.
            return true;
        default: return false;
    }
}

// WATERFALL loops (#273 — DOLL's skinned scene VS): the readfirstlane-uniformize idiom
//   L: v_readfirstlane s4, vIDX ; v_cmpx_eq_u32 s4, vIDX ; s_mov m0, s4 ; v_movrels …
//      s_andn2_b64 REMAIN, REMAIN, exec ; s_mov_b64 exec, REMAIN ; s_cbranch_scc1 L
// iterates once per DISTINCT per-lane index, with EXEC selecting the matching lanes. In the
// per-invocation model each lane's iteration IS its own: readfirstlane returns MY value, the cmpx
// keeps me active, and the andn2 leaves MY remaining-mask empty — so the loop body runs EXACTLY
// ONCE per invocation and the backward branch (whose SCC = "any lane remains", a cross-lane
// reduction) is never taken. A BACKWARD s_cbranch_scc0/scc1 whose SCC was produced by a 64-bit
// wave-mask op is therefore safe to DROP (linearize), exactly like the forward kill-mask branches.
// The SCC producer may sit a few instructions back (the shape interposes `s_mov_b64 exec, REMAIN`,
// which does not write SCC) — walk back over non-SCC-writing instructions to find it.
// CONFIDENCE: MED — exec-diff kernel T19 locks the once-through semantics.
std::unordered_set<uint32_t> waterfall_branches(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> out;
    for (size_t i = 0; i < ins.size(); i++) {
        const Rdna2Inst& in = ins[i];
        if (in.is_end) break;
        if (in.fmt != Rdna2Format::SOPP || (in.opcode != 0x04 && in.opcode != 0x05) ||
            in.simm16 >= 0)
            continue;
        // Walk back to the most recent SCC-writing instruction, skipping ops known not to touch SCC
        // (scalar moves, all vector ALU incl. VOPC — it writes VCC/EXEC, not SCC — memory, hints).
        for (size_t j = i; j-- > 0;) {
            const Rdna2Inst& p = ins[j];
            bool skip_ok = (p.fmt == Rdna2Format::SOP1 &&
                            (p.opcode == 0x03 || p.opcode == 0x04)) ||   // s_mov_b32/b64
                           p.fmt == Rdna2Format::VOP1 ||
                           p.fmt == Rdna2Format::VOP2 || p.fmt == Rdna2Format::VOP3 ||
                           p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::MIMG ||
                           p.fmt == Rdna2Format::MUBUF || p.fmt == Rdna2Format::MTBUF ||
                           p.fmt == Rdna2Format::FLAT || sopp_is_noop(p);
            if (skip_ok) continue;
            // The SCC producer: a 64-bit wave-mask logical op (s_and/or/xor/andn2_b64, SCC = result!=0).
            if (p.fmt == Rdna2Format::SOP2 &&
                (p.opcode == 0x0f || p.opcode == 0x11 || p.opcode == 0x13 || p.opcode == 0x15) &&
                p.dst.kind == OperandKind::SGPR && p.dst.value <= 106)
                out.insert(in.pc);
            break;   // any other SCC producer (s_cmp etc.): a REAL loop condition — not a waterfall
        }
    }
    return out;
}
uint32_t branch_target(const Rdna2Inst& in) {
    return in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16;
}

// The compare that decides a bottom-tested SCC loop (`do { ... s_cmp ... } while (SCC)`), or
// UINT32_MAX when it cannot be named. The back-edge reads SCC, so the loop's continue predicate is
// whatever SCC holds when control reaches it. That is exactly one compare only when the walk back
// from the back-edge to it is straight-line: every instruction in between is one that cannot write
// SCC, none is a branch, and no branch anywhere lands strictly after the compare (a landing there
// could arrive with another SCC). The writer must be an SOPC; other scalar ALU may write SCC as a
// side effect (carry, nonzero result) and is refused rather than modeled.
// Kena's (PPSA01802) blur fragment programs put `s_cmp_lt_u32 vcc_lo, 5` twenty dwords above the
// back-edge, with vector ALU, a waitcnt and an image sample in between (#4518).
uint32_t scc_do_while_condition_pc(const std::vector<Rdna2Inst>& ins, uint32_t header_pc,
                                   uint32_t backedge_pc) {
    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };
    size_t back = ins.size();
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].pc == backedge_pc) {
            back = i;
            break;
        }
    if (back == ins.size()) return UINT32_MAX;
    uint32_t writer = UINT32_MAX;
    for (size_t j = back; j-- > 0;) {
        const Rdna2Inst& p = ins[j];
        if (p.pc < header_pc) return UINT32_MAX;
        if (p.fmt == Rdna2Format::SOPC) {
            writer = p.pc;
            break;
        }
        const bool cannot_write_scc =
            (p.fmt == Rdna2Format::SOP1 && (p.opcode == 0x03 || p.opcode == 0x04)) ||
            p.fmt == Rdna2Format::VOP1 || p.fmt == Rdna2Format::VOP2 ||
            p.fmt == Rdna2Format::VOP3 || p.fmt == Rdna2Format::VOP3P ||
            p.fmt == Rdna2Format::VOPC || p.fmt == Rdna2Format::VINTRP ||
            p.fmt == Rdna2Format::SMEM || p.fmt == Rdna2Format::DS || p.fmt == Rdna2Format::MUBUF ||
            p.fmt == Rdna2Format::MTBUF || p.fmt == Rdna2Format::MIMG ||
            p.fmt == Rdna2Format::FLAT || sopp_is_noop(p);
        if (!cannot_write_scc) return UINT32_MAX;   // a branch, or a possible SCC side effect
    }
    if (writer == UINT32_MAX) return UINT32_MAX;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (!is_branch(in)) continue;
        const uint32_t target = branch_target(in);
        if (target > writer && target <= backedge_pc) return UINT32_MAX;
    }
    return writer;
}

}   // namespace prosper::gpu
