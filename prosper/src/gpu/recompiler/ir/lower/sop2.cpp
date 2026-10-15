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

uint32_t Sop2Lowering::logical_u32(uint32_t a, uint32_t c) {
    return in.opcode == kSop2OpcodeAndB32 ? b.ibin(Op_BitwiseAnd, a, c)
         : in.opcode == kSop2OpcodeOrB32 ? b.ibin(Op_BitwiseOr, a, c)
         : in.opcode == kSop2OpcodeXorB32 ? b.ibin(Op_BitwiseXor, a, c)
         : in.opcode == kSop2OpcodeAndn2B32
               ? b.ibin(Op_BitwiseAnd, a, b.iun(Op_Not, c))
         : in.opcode == kSop2OpcodeOrn2B32
               ? b.ibin(Op_BitwiseOr, a, b.iun(Op_Not, c))
         : in.opcode == kSop2OpcodeNandB32
               ? b.iun(Op_Not, b.ibin(Op_BitwiseAnd, a, c))
         : in.opcode == kSop2OpcodeNorB32
               ? b.iun(Op_Not, b.ibin(Op_BitwiseOr, a, c))
               : b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c));
}

uint32_t Sop2Lowering::bitfield_mask_u32(uint32_t width, uint32_t offset) {
    width = b.ibin(Op_BitwiseAnd, width, b.uconst(31));
    offset = b.ibin(Op_BitwiseAnd, offset, b.uconst(31));
    const uint32_t ones = b.ibin(
        Op_ISub,
        b.ibin(Op_ShiftLeftLogical, b.uconst(1), width), b.uconst(1));
    return b.ibin(Op_ShiftLeftLogical, ones, offset);
}

bool Sop2Lowering::lower_data() {
    // 32-bit scalar DATA writes into EXEC halves are rejected (see the SOP1 uint path).
    if (in.dst.value == 126 || in.dst.value == 127) { ok = false; return true; }
    uint32_t a = val(in.src[0]), c = val(in.src[1]); uint32_t& d = rs.sreg[in.dst.value];
    // Capture both operands before ending the old physical-word mask lifetime. These
    // ordinary DATA writers are separate from the dedicated mask producers above.
    retire_scalar_data_word_mask(rs, in.dst.value);
    auto scc_nz = [&](uint32_t v){ rs.scc = b.ucmp(Op_INotEqual, v, b.uconst(0)); };  // SCC = (result != 0)
    switch (in.opcode) {
        case 0x00: d = b.ibin(Op_IAdd, a, c); rs.scc = b.ucmp(Op_ULessThan, d, a); break;  // s_add_u32 (SCC=carry)
        case 0x01: d = b.ibin(Op_ISub, a, c); rs.scc = b.ucmp(Op_ULessThan, a, c); break;  // s_sub_u32 (SCC=borrow)
        // Signed add/sub: two's-complement result is bit-identical to the unsigned op; SCC is
        // signed OVERFLOW. add ovf = operands same sign AND result sign differs: (~(a^c))&(a^d);
        // sub ovf = operands differ in sign AND result sign differs from a: (a^c)&(a^d). Bit 31 = ovf.
        case 0x02: { d = b.ibin(Op_IAdd, a, c);                                              // s_add_i32
                     uint32_t o = b.ibin(Op_BitwiseAnd, b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c)),
                                                       b.ibin(Op_BitwiseXor, a, d));
                     rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
        case 0x03: { d = b.ibin(Op_ISub, a, c);                                              // s_sub_i32
                     uint32_t o = b.ibin(Op_BitwiseAnd, b.ibin(Op_BitwiseXor, a, c),
                                                       b.ibin(Op_BitwiseXor, a, d));
                     rs.scc = b.ucmp(Op_INotEqual, b.ibin(Op_ShiftRightLogical, o, b.uconst(31)), b.uconst(0)); break; }
        case 0x04: {   // s_addc_u32: dst = src0 + src1 + SCC(carry-in); SCC = carry-out. The high
                       // half of a 64-bit add (pairs with s_add_u32's SCC). Round-trip llvm-mc
                       // gfx1010: 0x82252580 → s_addc_u32 s37, 0, s37. CONFIDENCE: HIGH.
            if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op: carry-in unknown
            uint32_t cin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
            uint32_t s1 = b.ibin(Op_IAdd, a, c);
            uint32_t k1 = b.ucmp(Op_ULessThan, s1, a);        // wrap in a+c
            d = b.ibin(Op_IAdd, s1, cin);
            uint32_t k2 = b.ucmp(Op_ULessThan, d, s1);        // wrap in +cin
            rs.scc = b.bsel(k1, b.btrue(), k2); break;
        }
        case 0x05: {   // s_subb_u32: dst = src0 - src1 - SCC(borrow-in); SCC = borrow-out.
            if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op: borrow-in unknown
            const uint32_t bin = b.sel(rs.scc, b.uconst(1), b.uconst(0));
            const uint32_t difference = b.ibin(Op_ISub, a, c);
            const uint32_t first_borrow = b.ucmp(Op_ULessThan, a, c);
            d = b.ibin(Op_ISub, difference, bin);
            const uint32_t second_borrow = b.ucmp(Op_ULessThan, difference, bin);
            rs.scc = b.bsel(first_borrow, b.btrue(), second_borrow); break;
        }
        // s_min/max (0x06 min_i32, 0x07 min_u32, 0x08 max_i32, 0x09 max_u32): SCC = "src0 was
        // selected", STRICT in both directions per the ISA pseudocode (S_MIN: SCC = S0 < S1;
        // S_MAX_I32: "D.i = (S0.i > S1.i) ? S0.i : S1.i; SCC = (S0.i > S1.i)", doc 70648
        // sec 12.1 ops 6-9) — on a tie SCC = 0 for both min and max. An earlier change (#397)
        // flipped max to non-strict `>=` quoting a pseudocode line that is not in the actual
        // document; the 2026-07 ISA audit (#879) re-derived the strict form from the PDF.
        // D is unaffected either way (the tie value is identical). Round-trip llvm-mc
        // gfx1010: 0x83000201/0x83800201/0x84000201/0x84800201. CONFIDENCE: HIGH.
        case 0x06: rs.scc = b.scmp(Op_SLessThan, a, c);    d = b.sext2(Glsl_SMin, a, c); break;
        case 0x07: rs.scc = b.ucmp(Op_ULessThan, a, c);    d = b.uext2(Glsl_UMin, a, c); break;
        case 0x08: rs.scc = b.scmp(Op_SGreaterThan, a, c); d = b.sext2(Glsl_SMax, a, c); break;
        case 0x09: rs.scc = b.ucmp(Op_UGreaterThan, a, c); d = b.uext2(Glsl_UMax, a, c); break;
        case 0x0A:   // s_cselect_b32: SCC ? src0 : src1 (reads SCC, writes none)
            if (!rs.scc) { ok = false; break; }   // SCC poisoned by a mask op
            d = b.sel(rs.scc, a, c); break;
        case 0x0E: d = b.ibin(Op_BitwiseAnd, a, c); scc_nz(d); break;   // s_and_b32
        case 0x10: d = b.ibin(Op_BitwiseOr,  a, c); scc_nz(d); break;   // s_or_b32
        case 0x12: d = b.ibin(Op_BitwiseXor, a, c); scc_nz(d); break;   // s_xor_b32
        case 0x14: d = b.ibin(Op_BitwiseAnd, a, b.iun(Op_Not, c));       // s_andn2_b32
                   scc_nz(d); break;
        case 0x16: d = b.ibin(Op_BitwiseOr, a, b.iun(Op_Not, c));        // s_orn2_b32
                   scc_nz(d); break;
        case 0x18: d = b.iun(Op_Not, b.ibin(Op_BitwiseAnd, a, c));       // s_nand_b32
                   scc_nz(d); break;
        case 0x1A: d = b.iun(Op_Not, b.ibin(Op_BitwiseOr, a, c));        // s_nor_b32
                   scc_nz(d); break;
        case 0x1C: d = b.iun(Op_Not, b.ibin(Op_BitwiseXor, a, c));       // s_xnor_b32
                   scc_nz(d); break;
        case 0x1E: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshl_b32
                     d = b.ibin(Op_ShiftLeftLogical, a, sh); scc_nz(d); break; }  // dst = src0 << (src1 & 31)
        case kSop2OpcodeBfmB32:
            d = bitfield_mask_u32(a, c); break; // no SCC write
        case 0x20: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_lshr_b32
                     d = b.ibin(Op_ShiftRightLogical, a, sh); scc_nz(d); break; }  // dst = src0 >> (src1 & 31)
        case 0x22: { uint32_t sh = b.ibin(Op_BitwiseAnd, c, b.uconst(31));   // s_ashr_i32
                     d = b.sbin(Op_ShiftRightArithmetic, a, sh); scc_nz(d); break; }  // dst = src0 >>a (src1 & 31)
        case 0x26: d = b.ibin(Op_IMul, a, c); break;         // s_mul_i32 (low 32 bits; no SCC)
        case 0x2E: case 0x2F: case 0x30: case 0x31: {
            // s_lshl{1,2,3,4}_add_u32 = (src0<<N)+src1; SCC is unsigned carry-out of the
            // FULL 64-bit sum (RDNA2 ISA ops 46-49: ((u64)S0<<N)+S1 >= 2^32). If any of
            // S0's top N bits are set, the shift alone overflowed; otherwise overflow is
            // exactly the ordinary low-dword addition wrap (D < S1).
            const uint32_t shift = in.opcode - 0x2Du;
            uint32_t shifted = b.ibin(Op_ShiftLeftLogical, a, b.uconst(shift));
            d = b.ibin(Op_IAdd, shifted, c);
            uint32_t shifted_out = b.ucmp(Op_INotEqual,
                b.ibin(Op_ShiftRightLogical, a, b.uconst(32u - shift)), b.uconst(0));
            uint32_t wrapped = b.ucmp(Op_ULessThan, d, c);
            rs.scc = b.bsel(shifted_out, b.btrue(), wrapped); break;
        }
        case kSop2OpcodePackLlB32B16: { // s_pack_ll_b32_b16: D={S1[15:0],S0[15:0]}
            const uint32_t lo = b.ibin(Op_BitwiseAnd, a, b.uconst(0xFFFFu));
            const uint32_t hi = b.ibin(Op_ShiftLeftLogical,
                b.ibin(Op_BitwiseAnd, c, b.uconst(0xFFFFu)), b.uconst(16));
            d = b.ibin(Op_BitwiseOr, lo, hi);
            break;
        }
        case 0x35: d = b.umul_hi(a, c); break;               // s_mul_hi_u32 (high 32 bits; no SCC)
        case 0x36: d = b.smul_hi(a, c); break;               // s_mul_hi_i32 (high 32 bits; no SCC).
                                                             // gfx10 SOP2 opcode is 0x36 (llvm-mc:
                                                             // 0x9b000201>>23&0x7f); 0x37 was WRONG
                                                             // (invalid encoding), so the handler was
                                                             // dead and s_mul_hi_i32 got rejected. #462
        case 0x27: {                                         // s_bfe_u32: offset=src1[4:0], width=src1[22:16]
            uint32_t off = b.ibin(Op_BitwiseAnd, c, b.uconst(0x1f));
            uint32_t width = b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, c, b.uconst(16)), b.uconst(0x7f));
            d = b.bfe_u(a, off, width); scc_nz(d); break;
        }
        default: ok = false;
    }
    // Every modeled SOP2 operation changes the scalar bits (including cselect unless both
    // inputs happen to be identical). A previous descriptor tag on the destination is stale;
    // retaining it can bind an unrelated later SRSRC to the old buffer.
    if (ok) {
        rs.sreg_srt.erase(in.dst.value);
        if (in.opcode == kSop2OpcodeBfeU64)
            rs.sreg_srt.erase(in.dst.value + 1);
    }
    return true;
}

} // namespace prosper::gpu::ir
