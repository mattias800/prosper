#include "gpu/recompiler/rdna2_exec_skip_region.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {
namespace {

// A scalar destination written in the region is acceptable only when no path from the merge reads
// it before redefining it. `dwords` consecutive registers are checked. EXEC (126/127) and M0 (124)
// are never provably dead here: they have implicit readers everywhere, so any write is a live-out.
bool scalar_destination_dead(const std::vector<Rdna2Inst>& ins, uint32_t target_pc,
                             const Operand& destination, int dwords) {
    if (destination.kind != OperandKind::SGPR && destination.kind != OperandKind::Special)
        return false;
    if (destination.value == 125) return true;   // NULL: writes are discarded
    for (int k = 0; k < dwords; ++k) {
        const int reg = destination.value + k;
        if (reg > 107) return false;             // M0, EXEC, and anything unnamed
        if (!sgpr_dead_at_merge(ins, target_pc, reg)) return false;
    }
    return true;
}

int smem_load_dwords(uint32_t opcode) {
    switch (opcode) {
        case 0x0:
        case 0x8: return 1;
        case 0x1:
        case 0x9: return 2;
        case 0x2:
        case 0xA: return 4;
        case 0x3:
        case 0xB: return 8;
        case 0x4:
        case 0xC: return 16;
        default: return 0;
    }
}

} // namespace

ExecSkipRegionEffects classify_exec_skip_region(const std::vector<Rdna2Inst>& ins,
                                                uint32_t branch_pc, uint32_t target_pc) {
    ExecSkipRegionEffects fx;
    if (target_pc <= branch_pc) {   // not a forward skip: nothing here is proven
        fx.foreign_exit = true;
        return fx;
    }
    const auto dead = [&](const Operand& destination, int dwords) {
        return scalar_destination_dead(ins, target_pc, destination, dwords);
    };
    for (const auto& in : ins) {
        if (in.pc <= branch_pc || in.pc >= target_pc) continue;
        if (in.is_end || in.synthetic_terminator || rdna2_escapes_decoded_effects(in)) {
            fx.foreign_exit = true;
            continue;
        }
        switch (in.fmt) {
            case Rdna2Format::VOP1:
                if (in.has_dpp || in.has_modifier) fx.wave_side_effect = true;
                // v_readfirstlane_b32: the decoder keeps the SGPR index in a VGPR-kinded `dst`.
                if (in.opcode == 0x02 && !dead({OperandKind::SGPR, in.dst.value}, 1))
                    fx.scalar_live_out = true;
                break;
            case Rdna2Format::VOP2:
                if (in.has_dpp || in.has_modifier) fx.wave_side_effect = true;
                // v_add/sub/subrev_co_ci_u32 write the carry-out to VCC implicitly.
                if (in.opcode >= 0x28 && in.opcode <= 0x2A && !dead({OperandKind::Special, 106}, 2))
                    fx.scalar_live_out = true;
                break;
            case Rdna2Format::VOPC:
                if (in.has_dpp || in.has_modifier) fx.wave_side_effect = true;
                // v_cmpx writes EXEC only; the others write the VCC/SGPR pair in `dst`.
                if (vopc_is_cmpx(in.opcode) || !dead(in.dst, 2)) fx.scalar_live_out = true;
                break;
            case Rdna2Format::VOP3:
                if (in.has_dpp || in.has_modifier) fx.wave_side_effect = true;
                // 0x182 is v_readfirstlane_b32 and 0x360 v_readlane_b32 in their VOP3 spelling.
                if ((in.opcode == 0x182 || in.opcode == 0x360) &&
                    !dead({OperandKind::SGPR, in.dst.value}, 1))
                    fx.scalar_live_out = true;
                if (in.sdst.kind != OperandKind::None && !dead(in.sdst, 2))
                    fx.scalar_live_out = true;
                break;
            case Rdna2Format::VOP3P:
            case Rdna2Format::VINTRP: break;   // VGPR destinations only
            case Rdna2Format::SOPP:
                // Only pure hints. Every branch (the way out of the region, or a nested region
                // the emitter would have to structure separately) and every message, barrier,
                // sleep, trap or trace operation is refused.
                switch (in.opcode) {
                    case 0x00:
                    case 0x0c:
                    case 0x20:
                    case 0x21: break;
                    case 0x02:
                    case 0x04:
                    case 0x05:
                    case 0x06:
                    case 0x07:
                    case 0x08:
                    case 0x09: fx.foreign_exit = true; break;
                    default: fx.wave_side_effect = true; break;
                }
                break;
            case Rdna2Format::SOPK:
                // vmcnt/expcnt/lgkmcnt waits are counters the synchronous model never waits on.
                // s_waitcnt_vscnt is a publication barrier, s_setreg changes wave state, and the
                // rest are scalar ALU writes with an SCC/SGPR effect.
                if (in.opcode >= 0x18 && in.opcode <= 0x1A) break;
                if (in.opcode == kSopkOpcodeWaitcntVscnt || in.opcode == kSopkOpcodeSetregB32)
                    fx.wave_side_effect = true;
                else
                    fx.scalar_live_out = true;
                break;
            case Rdna2Format::SOP1:
            case Rdna2Format::SOP2:
            case Rdna2Format::SOPC:
                // SCC has no liveness proof, and the destination is rarely provably dead: the
                // VALU-only region is the case this certificate exists for.
                fx.scalar_live_out = true;
                break;
            case Rdna2Format::SMEM: {
                const int n = smem_load_dwords(in.opcode);
                if (rdna2_instruction_may_write_memory(in) || n == 0 || !dead(in.dst, n))
                    fx.scalar_memory_effect = true;
                break;
            }
            default:   // DS, MUBUF, MTBUF, MIMG, FLAT, EXP, Unknown
                fx.unclassified = true;
                break;
        }
    }
    return fx;
}

}   // namespace prosper::gpu
