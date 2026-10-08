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
        if (!sgpr_dead_at_merge(ins, target_pc, reg, ScalarMergeProof::AnyRead, nullptr,
                                /*vopc_sgpr_pair_kills=*/true))
            return false;
    }
    return true;
}

// Cross-lane and lane-addressed VALU: its result depends on which lanes of the 64-lane guest wave
// are active, and the fragment emitter lowers it per invocation (v_readfirstlane returns the
// invocation's own lane), so it is wrong on a narrower host even when its destination is dead.
bool is_cross_lane_valu(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::VOP1) return in.opcode == 0x02;          // v_readfirstlane_b32
    if (in.fmt != Rdna2Format::VOP3) return false;
    switch (in.opcode) {
        case 0x182:                                                     // readfirstlane (VOP3 form)
        case 0x360: case 0x361:                                         // v_readlane / v_writelane
        case 0x365: case 0x366:                                         // v_mbcnt_lo / v_mbcnt_hi
        case 0x377: case 0x378:                                         // v_permlane16 / x16
            return true;
        default: return false;
    }
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
            case Rdna2Format::VOP2:
            case Rdna2Format::VOPC:
            case Rdna2Format::VOP3:
                // DPP crosses lanes, and an SDWA/modifier form is not one the emitter models.
                if (in.has_dpp || in.has_modifier || is_cross_lane_valu(in))
                    fx.wave_side_effect = true;
                if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2A &&
                    !dead({OperandKind::Special, 106}, 2))
                    fx.scalar_live_out = true;   // v_*_co_ci_u32 carry-out to VCC
                if (in.fmt == Rdna2Format::VOPC &&
                    (vopc_is_cmpx(in.opcode) || !dead(in.dst, 2)))
                    fx.scalar_live_out = true;   // v_cmpx writes EXEC; others write VCC/SGPR pair
                if (in.fmt == Rdna2Format::VOP3 && in.sdst.kind != OperandKind::None &&
                    !dead(in.sdst, 2))
                    fx.scalar_live_out = true;   // VOP3B carry/flag out
                break;
            case Rdna2Format::VOP3P:
            case Rdna2Format::VINTRP:
                break;   // VGPR destinations only
            case Rdna2Format::SOPP:
                // Only pure hints. Every branch (the way out of the region, or a nested region
                // the emitter would have to structure separately) and every message, barrier,
                // sleep, trap or trace operation is refused.
                switch (in.opcode) {
                    case 0x00: case 0x0c: case 0x20: case 0x21: break;
                    case 0x02: case 0x04: case 0x05: case 0x06: case 0x07: case 0x08: case 0x09:
                        fx.foreign_exit = true;
                        break;
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
                // gfx10.3 has no scalar stores or atomics; every non-load opcode (cache control,
                // s_memtime, s_atc_probe, and any encoding left over from older ISAs) has width 0
                // and is refused here, which is also what keeps this fail-closed for new opcodes.
                const int n = smem_load_dwords(in.opcode);
                if (n == 0 || !dead(in.dst, n)) fx.scalar_memory_effect = true;
                break;
            }
            default:   // DS, MUBUF, MTBUF, MIMG, FLAT, EXP, Unknown
                fx.unclassified = true;
                break;
        }
    }
    return fx;
}

} // namespace prosper::gpu
