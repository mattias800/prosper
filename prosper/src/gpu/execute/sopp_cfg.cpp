#include "gpu/execute/sopp_cfg.hpp"

namespace prosper::gpu {

// SOPP direct branches: s_branch (0x02) and the s_cbranch_* family (0x04..0x09). Hoisted so the two
// executor CFG proofs share one definition of "is a branch" and one target computation — #2181
// unified four private copies of the VOPC cmpx windows for the same reason, and #2120 is the
// cautionary tale for a forked predicate.
bool sopp_is_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
           (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
}
bool sopp_is_unconditional_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && in.opcode == 0x02;
}
// GFX10 branch target: PC of the branch + its own length + the signed dword displacement. Direction
// is deliberately NOT filtered here — a predecessor tally that only counts forward edges is not a
// predecessor tally (#2202 review, B2).
int64_t sopp_branch_target(const Rdna2Inst& in) {
    return static_cast<int64_t>(in.pc) + static_cast<int64_t>(in.len_dwords) +
           static_cast<int64_t>(in.simm16);
}
// Indirect control transfer: s_setpc_b64 / s_swappc_b64 / s_rfe_b64 (SOP1 0x20/0x21/0x22) and
// s_call_b64 (SOPK 0x16). Encodings round-tripped through llvm-mc -mcpu=gfx1030, never read off a
// table (see SONIC_CROSSWORLDS_STATUS.md § Ruled out, the 0x305 trap). A shader containing one has a
// CFG no static scan over SOPP displacements can represent.
bool has_indirect_control_flow(const std::vector<Rdna2Inst>& instructions) {
    for (const Rdna2Inst& in : instructions) {
        if (in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == 0x20 || in.opcode == 0x21 || in.opcode == 0x22))
            return true;
        if (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16) return true;
    }
    return false;
}

} // namespace prosper::gpu
