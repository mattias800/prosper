// sop1.cpp — SOP1 lowering: the route chain, then the plain scalar-data tail.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_sop1(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return Sop1Lowering({b, rs, in, ok, allow_wave}).lower();
}

bool Sop1Lowering::lower() {
    if (auto routed = route_cmov_b64_odd_pair()) return *routed;
    if (auto routed = route_wqm_numeric_special_destination()) return *routed;
    if (auto routed = route_wqm_numeric()) return *routed;
    if (auto routed = route_quadmask_b64()) return *routed;
    if (auto routed = route_ff1_b64_native64()) return *routed;
    if (auto routed = route_ff1_b32_data()) return *routed;
    if (auto routed = route_ff1_b32_wave32()) return *routed;
    if (auto routed = route_exec_half_from_vcc_half()) return *routed;
    if (auto routed = route_exec_half_immediate()) return *routed;
    if (auto routed = route_mov_b32_mask()) return *routed;
    if (auto routed = route_mov_b32_exec_lo_full()) return *routed;
    if (auto routed = route_saveexec_b32()) return *routed;
    if (auto routed = route_wqm_b32()) return *routed;
    if (auto routed = route_not_b32_mask()) return *routed;
    if (auto routed = route_bcnt1_b64()) return *routed;
    if (auto routed = route_b64_mask_ops()) return *routed;
    if (auto routed = route_not_vcc_half_self()) return *routed;
    if (auto routed = route_getpc_b64()) return *routed;
    if (auto routed = route_bcnt1_b64_one_lane()) return *routed;
    if (auto routed = route_ff1_b64_one_lane()) return *routed;
    if (auto routed = route_flbit()) return *routed;
    if (auto routed = route_bitset()) return *routed;
    if (auto routed = route_brev_exec_lo_wave32()) return *routed;
    if (auto routed = route_bitreplicate_b64_b32()) return *routed;
    if (auto routed = route_reject_exec_half_data()) return *routed;
    if (auto routed = route_entry_m0_save()) return *routed;
    if (auto routed = route_entry_m0_restore()) return *routed;
    return lower_data();
}

bool Sop1Lowering::lower_data() {
    if (in.opcode == 0x2E) return emit_s_movrels_b32(b, rs, in, ok);   // s_movrels_b32
    // A write of anything else ends this register's entry-M0 lifetime.
    rs.sreg_entry_m0.erase(in.dst.value);
    uint32_t a = val(in.src[0]); uint32_t& d = rs.sreg[in.dst.value];
    const uint32_t partial_vcc = partial_vcc_word_predicate(b, rs, in, a);
    // An ordinary scalar-data write starts a new lifetime for this physical word.  Do not
    // let an earlier Wave32 mask save alias that new value in later mask-domain moves.
    // A numeric half starts a new physical generation. A complete VCC self-copy must
    // not recover the departed B64 predicate when numeric uniformity is unproved.
    // An independently tagged B32 low mask does not own the adjacent high DATA word.
    retire_scalar_data_word_mask(rs, in.dst.value);
    switch (in.opcode) {
        case 0x03: {                                // s_mov_b32
            publish_s_mov_b32_word(rs, in, a, partial_vcc);
            break;
        }
        case 0x07:                                  // s_not_b32 changes the descriptor word
            // ISA op 7: "D = ~S0; SCC = (D != 0)". (s_brev_b32 below has NO SCC write.)
            d = b.iun(Op_Not, a); rs.sreg_srt.erase(in.dst.value);
            rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0)); break;
        case kSop1OpcodeBrevB32:                     // s_brev_b32
            d = b.iun(Op_BitReverse, a); rs.sreg_srt.erase(in.dst.value); break;
        case 0x34: {                                // s_abs_i32
            // Two's-complement absolute value.  OpISub deliberately preserves the ISA's
            // INT_MIN -> INT_MIN wraparound; SCC is set iff the resulting bits are nonzero.
            const uint32_t negative = b.scmp(Op_SLessThan, a, b.uconst(0));
            d = b.sel(negative, b.ibin(Op_ISub, b.uconst(0), a), a);
            rs.scc = b.ucmp(Op_INotEqual, d, b.uconst(0));
            rs.sreg_srt.erase(in.dst.value);
            break;
        }
        default: ok = false;
    }
    return true;
}

} // namespace prosper::gpu::ir
