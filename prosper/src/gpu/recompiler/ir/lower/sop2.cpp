// sop2.cpp — SOP2 lowering: the route chain, emit_alu's shared operand predicates, then the plain
// scalar-data tail.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

bool lower_sop2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave) {
    return Sop2Lowering({b, rs, in, ok, allow_wave}).lower();
}

bool Sop2Lowering::lower() {
    gtav_wave32_vcchi_scalar_packet =
        allows_compute_scalar_vcc_bridge(b) && b.native_subgroup_size == 32 &&
        is_gtav_wave32_vcchi_scalar_packet(in);
    if (auto routed = route_wave64_vcc_lo_cselect()) return *routed;
    if (auto routed = route_b32_mask_cselect()) return *routed;
    if (auto routed = route_cselect_b32_vcc_lo_bridge()) return *routed;
    if (auto routed = route_exec_lo_logical_data()) return *routed;
    if (auto routed = route_exec_lo_bfm_data()) return *routed;
    mixed_wave32_mask_scalar_data =
        b.is_compute && b.wave_size == 32 && b.native_subgroup_size == 32 &&
        in.dst.kind == OperandKind::SGPR && in.dst.value <= 105 &&
        wave32_mask_operand(in.src[0]) != wave32_mask_operand(in.src[1]) &&
        exact_wave32_scalar_dword(in.src[0]) &&
        exact_wave32_scalar_dword(in.src[1]);
    if (auto routed = route_wave32_b32_mask_logical()) return *routed;
    if (auto routed = route_pack_halfwords()) return *routed;
    if (auto routed = route_bfm_b64()) return *routed;
    if (auto routed = route_cselect_b64()) return *routed;
    if (auto routed = route_b64_mask_logical()) return *routed;
    if (auto routed = route_lshr_b64_exec_pack()) return *routed;
    if (auto routed = route_shift_b64()) return *routed;
    if (auto routed = route_bfe_u64()) return *routed;
    if (auto routed = route_vcc_half_b32_logical()) return *routed;
    return lower_data();
}

bool Sop2Lowering::wave32_mask_operand(const Operand& source) const {
    // Inline integer/literal operands are representable in both domains. Classify
    // them as masks so a pure mask operation preserves its Bool lifetime; only a
    // genuinely mixed mask/raw-data operation should take the scalar route below.
    if (source.kind == OperandKind::InlineInt ||
        source.kind == OperandKind::Literal)
        return true;
    if (source.kind == OperandKind::Special && source.value == 126)
        return true; // EXEC_LO
    return (source.kind == OperandKind::SGPR ||
            source.kind == OperandKind::Special) &&
        rs.sreg_bool_b32.contains(source.value);
}

bool Sop2Lowering::wave32_live_mask_operand(const Operand& source) const {
    if (source.kind == OperandKind::Special && source.value == 126)
        return true; // EXEC_LO
    return (source.kind == OperandKind::SGPR ||
            source.kind == OperandKind::Special) &&
        rs.sreg_bool_b32.contains(source.value);
}

bool Sop2Lowering::exact_wave32_scalar_dword(const Operand& source) const {
    switch (source.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal:
            return true;
        case OperandKind::SGPR:
            return rs.sreg.contains(source.value) ||
                rs.sreg_input.contains(source.value) ||
                (b.is_compute && b.wave_size == 32 &&
                 b.native_subgroup_size == 32 &&
                 rs.sreg_bool_b32.contains(source.value) &&
                 rs.sreg_bool.contains(source.value));
        case OperandKind::Special:
            if (source.value == 125) return true; // SGPR_NULL
            if (source.value == 253) return rs.scc != 0;
            if (source.value >= 106 && source.value <= 124)
                return rs.sreg.contains(source.value);
            return b.is_compute && b.wave_size == 32 &&
                b.native_subgroup_size == 32 &&
                (source.value == 126 || source.value == 127) && rs.exec;
        default:
            return false;
    }
}

bool Sop2Lowering::scalar_data_operand(const Operand& source) const {
    switch (source.kind) {
        case OperandKind::InlineInt:
        case OperandKind::InlineFloat:
        case OperandKind::Literal:
            return true;
        case OperandKind::SGPR:
            return rs.sreg.contains(source.value) ||
                rs.sreg_input.contains(source.value);
        case OperandKind::Special:
            if (source.value == 125) return true; // SGPR_NULL
            if (source.value == 253) return rs.scc != 0;
            return source.value >= 106 && source.value <= 124 &&
                rs.sreg.contains(source.value);
        default:
            return false;
    }
}

ScalarOpIds Sop2Lowering::emit_data_op(uint32_t s0, uint32_t s1) {
    const ScalarOpDef* def = find_scalar_op(in.fmt, in.opcode);
    if (!def) {
        ok = false;
        return {};
    }
    return emit_scalar_op(b, *def, s0, s1, rs.scc);
}

bool Sop2Lowering::lower_data() {
    // 32-bit scalar DATA writes into EXEC halves are rejected (see the SOP1 uint path).
    if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
    uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t& d = rs.sreg[in.dst.value];
    // Capture both operands before ending the old physical-word mask lifetime. These
    // ordinary DATA writers are separate from the dedicated mask producers above.
    retire_scalar_data_word_mask(rs, in.dst.value);
    const ScalarOpDef* def = find_scalar_op(in.fmt, in.opcode);
    // An operation that reads SCC (s_cselect, s_addc, s_subb) cannot use a poisoned SCC: a mask op
    // left it unrepresentable.
    if (!def || (rdna2_reads_scc_as_data(in.fmt, in.opcode) == Rdna2Answer::Yes && !rs.scc)) {
        ok = false;
        return true;
    }
    const ScalarOpIds result = emit_scalar_op(b, *def, a, c, rs.scc);
    d = result.d;
    if (result.scc) rs.scc = result.scc;
    // Every modeled SOP2 operation changes the scalar bits (including cselect unless both
    // inputs happen to be identical). A previous descriptor tag on the destination is stale;
    // retaining it can bind an unrelated later SRSRC to the old buffer.
    if (ok) rs.sreg_srt.erase(in.dst.value);
    return true;
}

} // namespace prosper::gpu::ir
