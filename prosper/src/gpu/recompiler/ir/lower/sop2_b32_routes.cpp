// sop2_b32_routes.cpp — SOP2 one-word routes that touch EXEC_LO, VCC halves or Wave32 masks, plus s_pack_*.
// Relocated verbatim from emit_alu's SOP2 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop2Lowering::route_exec_lo_logical_data() {
    if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
        sop2_is_b32_logical(in.opcode) &&
        scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1])) {
        // A physical EXEC_LO write always changes the wave mask, even when both inputs are
        // ordinary scalar DATA rather than Bool-domain masks. GTA V writes 0x80 into
        // VCC_LO as scalar scratch, then ORs it with literal 0x100 to select lanes 7 and 8.
        // Both complete dwords are available uniformly, so compute the architectural u32,
        // publish exact SCC, and select this invocation's Wave32 bit without a subgroup op.
        const ScalarOpIds logical = emit_data_op(val(in.src[0]), val(in.src[1]));
        const uint32_t result = logical.d;
        if (!ok) return true;
        rs.scc = logical.scc;
        const uint32_t lane = b.ibin(
            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
        rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        rs.exec_narrowed = true;
        rs.sreg.erase(126);
        rs.sreg_srt.erase(126);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_exec_lo_bfm_data() {
    if (b.is_compute && b.wave_size == 32 && in.dst.value == 126 &&
        in.opcode == kSop2OpcodeBfmB32 &&
        scalar_data_operand(in.src[0]) && scalar_data_operand(in.src[1])) {
        // BFM writes an ordinary 32-bit result, but physical EXEC_LO still consumes that
        // dword as the complete Wave32 mask. GTA V uses width=16, offset=16 to enable the
        // upper half of a traversal wave. Both operands are uniform scalar DATA, so select
        // the resulting architectural bit for this invocation; S_BFM_B32 leaves SCC alone.
        const uint32_t result = emit_data_op(
            val(in.src[0]), val(in.src[1])).d;
        if (!ok) return true;
        const uint32_t lane = b.ibin(
            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
        rs.exec = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        rs.exec_narrowed = true;
        rs.sreg.erase(126);
        rs.sreg_srt.erase(126);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_wave32_b32_mask_logical() {
    if (b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
        !gtav_wave32_vcchi_scalar_packet &&
        !mixed_wave32_mask_scalar_data &&
        sop2_is_b32_logical(in.opcode) &&
        (in.dst.value == 126 ||
         wave32_live_mask_operand(in.src[0]) ||
         wave32_live_mask_operand(in.src[1]))) {
        // Wave32 B32 logical operations are one-word wave-mask operations, parallel to the
        // B64 family immediately below. The live Astro material uses
        //   s_andn2_b32 s64, s64, vcc_hi
        // after an explicit VOPC write to VCC_HI. Resolve each operand in the per-lane Bool
        // domain; treating VCC_HI as scalar bits is unrepresentable in this model.
        auto mask = [&](const Operand& o) -> uint32_t {
            if (o.value == 126) return rs.exec;
            if ((o.value == 106 || o.value == 107) &&
                rs.sreg_bool_b32.contains(o.value)) {
                auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second;
                if (o.value == 106) return rs.vcc;
            }
            if (o.kind == OperandKind::SGPR && rs.sreg_bool_b32.contains(o.value)) {
                auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second;
            }
            if (o.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, o.value);
            if (o.kind == OperandKind::Literal)
                return inline_int_mask_bit(
                    b, static_cast<int32_t>(in.literal));
            return 0;
        };
        const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
        if (!m0 || !m1 || in.dst.value == 127) { ok = false; return true; }
        const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
        const uint32_t x = b.bsel(m0, n1, m1);
        const uint32_t r = in.opcode == kSop2OpcodeAndB32 ? b.land(m0, m1)
                         : in.opcode == kSop2OpcodeOrB32 ? b.lor(m0, m1)
                         : in.opcode == kSop2OpcodeXorB32 ? x
                         : in.opcode == kSop2OpcodeAndn2B32 ? b.land(m0, n1)
                         : in.opcode == kSop2OpcodeOrn2B32 ? b.lor(m0, n1)
                         : in.opcode == kSop2OpcodeNandB32 ? b.lor(n0, n1)
                         : in.opcode == kSop2OpcodeNorB32 ? b.land(n0, n1)
                         : b.logical_not(x);
        // SCC=(result!=0) is a guest-wave reduction. A fragment with proven Wave32 can
        // request one exact 32-lane Vulkan subgroup and vote here; compute retains the
        // existing poison unless its dispatcher supplies a synchronized reduction.
        rs.scc = b.is_fragment ? b.fragment_wave_any(r) : 0;
        if (in.dst.value == 126) {
            rs.exec = r;
            rs.exec_narrowed = true;
        } else {
            rs.sreg_bool[in.dst.value] = r;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            rs.sreg_bool_b32.insert(in.dst.value);
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            if (in.dst.value == 106) rs.vcc = r;
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_pack_halfwords() {
    if (in.opcode >= 0x32 && in.opcode <= 0x34) {
        // GFX10 scalar halfword pack family.  LL selects src0.lo/src1.lo, LH selects
        // src0.lo/src1.hi, and HH selects src0.hi/src1.hi.  These are pure scalar DATA
        // operations (no SCC write); NGG uses LL to pack two wave population counts that
        // temporarily live in VCC_LO/HI.
        const uint32_t a = val(in.src[0]), c = val(in.src[1]);
        if (!ok) return true;
        const uint32_t result = emit_data_op(a, c).d;   // ir/ops/scalar_alu.cpp pack_b32_b16
        rs.sreg[in.dst.value] = result;
        rs.sreg_srt.erase(in.dst.value);
        if (b.vcc_pack_scalar_pair_pcs.contains(in.pc)) {
            const auto high = rs.sreg.find(107);
            if (vcc_sibling_unavailable(b, rs, 107, high, ok)) return true;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
            const uint32_t word = b.sel(
                b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                high->second, result);
            const uint32_t bit = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, word,
                       b.ibin(Op_BitwiseAnd, lane, b.uconst(31))),
                b.uconst(1));
            rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
            rs.sreg_bool[106] = rs.vcc;
            rs.sreg_bool_narrowed[106] = true;
        }
        if (!b.is_compute && !b.is_fragment && in.dst.value == 106) {
            const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
            rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
            rs.sreg_bool_narrowed[106] = true;
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_vcc_half_b32_logical() {
    if ((in.dst.value == 106 || in.dst.value == 107) &&
        in.opcode >= 0x0e && in.opcode <= 0x1c && (in.opcode & 1u) == 0) {
        // 32-bit logical writes to VCC_LO/HI bridge scalar DATA masks into the architectural
        // lane mask. NGG prologs use this to merge a scalar bitset with a VOPC result before
        // a v_cndmask. We cannot materialize the complete VCC dword in a per-invocation
        // module, but its bit for this guest lane is exact. Preserve the untouched VCC half;
        // unlike a B64 mask write, a B32 write changes only LO or HI.
        auto has_scalar_data = [&](const Operand& source) {
            switch (source.kind) {
                case OperandKind::SGPR:
                case OperandKind::InlineInt:
                case OperandKind::InlineFloat:
                case OperandKind::Literal: return true;
                case OperandKind::Special:
                    if (source.value == 125) return true;       // SGPR_NULL
                    if (source.value == 253) return rs.scc != 0; // SCC as scalar 0/1
                    return source.value >= 106 && source.value <= 124 &&
                           rs.sreg.count(source.value) != 0;
                default: return false;
            }
        };
        const bool complete_scalar_pair = vcc_b32_scalar_pair_defined(b, rs, in);
        // Snapshot source authority before an in-place write replaces VCC or SCC. The
        // shared Special-word check omits physical words unavailable in this domain.
        const auto unavailable = [&](const Operand& source) {
            return (source.kind == OperandKind::SGPR ||
                    source.kind == OperandKind::Special) &&
                   rs.unavailable_saved_mask_words.contains(source.value);
        };
        const bool current_data_sources = b32_sources_defined(rs, in) &&
                                          !unavailable(in.src[0]) &&
                                          !unavailable(in.src[1]);
        const bool has_proven_scalar_sources = has_scalar_data(in.src[0]) &&
                                               has_scalar_data(in.src[1]) &&
                                               vcc_b32_scalar_write_proven(b, rs, in);
        if ((!b.is_fragment && !b.is_compute) || gtav_wave32_vcchi_scalar_packet ||
            has_proven_scalar_sources) {
            // The vertex shell is a complete one-lane virtual wave, so its scalar VCC
            // dwords are representable: LO starts as {bit0=vcc}, HI as zero, and later B32
            // operations may use either as ordinary scratch. In Wave32 compute, VCC_HI is
            // entirely outside the architectural mask and is therefore always scalar
            // scratch; an exact 32-lane subgroup also lets operand_bits materialize an
            // EXEC_LO/VCC_LO source through one complete ballot. Other fragment/compute
            // forms retain a full dword only when BOTH inputs already have scalar-data
            // representations. The captured Plucky Squire tonemap shader, for example,
            // computes `s_and_b32 vcc_lo, loop_index, 3` and immediately compares VCC_LO as
            // a uint. True VOPC/mask inputs have no scalar representation and continue into
            // the wave-mask path below.
            const ScalarOpIds logical = emit_data_op(val(in.src[0]), val(in.src[1]));
            const uint32_t result = logical.d;
            if (!ok) return true;
            rs.sreg[in.dst.value] = result;
            rs.sreg_srt.erase(in.dst.value);
            rs.scc = logical.scc;
            // Current vertex DATA need not recreate a retired wave predicate. Preserve
            // genuine scalar arithmetic; later mask consumers still need their own proof.
            if (b.is_vertex && !b.ngg_one_lane && !rs.vcc && !complete_scalar_pair &&
                current_data_sources) {
                drop_vcc_mask_view(rs);
                return true;
            }
            const bool writes_hi = in.dst.value == 107;
            // A whole-CFG proof can establish that this physical VCC word is scalar data
            // only: the untouched sibling dies before any mask-domain read. Do not ask for
            // a guest lane merely to manufacture a predicate that no later instruction can
            // observe. This also covers vertex scalar scratch after an ordinary DATA
            // write has retired the old predicate; no lane-mask value is manufactured.
            const bool scalar_low_only = b.vcc_b32_low_only_pcs.contains(in.pc);
            if (b.wave_size != 32 && scalar_low_only) {
                rs.vcc = 0;
                rs.sreg_bool.erase(106);
                rs.sreg_bool_narrowed.erase(106);
                rs.sreg_bool_b32.erase(106);
                rs.sreg_bool.erase(107);
                rs.sreg_bool_narrowed.erase(107);
                rs.sreg_bool_b32.erase(107);
                return true;
            }
            uint32_t lane = b.guest_lane_id();
            lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
            const uint32_t in_written_half = writes_hi
                ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
                : b.ucmp(Op_ULessThan, lane, b.uconst(32));
            const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
            const uint32_t result_bit = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd,
                       b.ibin(Op_ShiftRightLogical, result, bit), b.uconst(1)),
                b.uconst(0));
            if (b.wave_size == 32) {
                // A complete scalar write covers every architectural bit of VCC_LO in
                // Wave32, so it establishes a fresh predicate without needing the old VCC
                // lifetime. VCC_HI is ordinary scratch outside the 32-lane mask and cannot
                // affect the implicit VCC condition at all. This distinction matters for
                // generated kernels that load scalar data into LO, derive a flag in HI, and
                // compare both as uints before creating their next vector predicate.
                if (writes_hi) {
                    rs.sreg_bool.erase(107);
                    rs.sreg_bool_narrowed.erase(107);
                    rs.sreg_bool_b32.erase(107);
                } else {
                    rs.vcc = result_bit;
                    rs.sreg_bool[106] = result_bit;
                    rs.sreg_bool_narrowed[106] = true;
                }
                return true;
            }
            if (complete_scalar_pair) {
                const int sibling = writes_hi ? 106 : 107;
                auto other = rs.sreg.find(sibling);
                if (vcc_sibling_unavailable(b, rs, sibling, other, ok)) return true;
                const uint32_t other_bit = b.ucmp(
                    Op_INotEqual,
                    b.ibin(Op_BitwiseAnd,
                           b.ibin(Op_ShiftRightLogical, other->second, bit),
                           b.uconst(1)),
                    b.uconst(0));
                rs.vcc = b.bsel(in_written_half, result_bit, other_bit);
            } else if (scalar_low_only) {
                // The whole-CFG proof established that the untouched high word cannot be
                // observed before a complete pair replacement. This instruction therefore
                // starts a scalar-only VCC_LO lifetime; retaining the old Bool predicate
                // would give later dispatcher blocks a competing stale mask domain.
                rs.vcc = 0;
                rs.sreg_bool.erase(106);
                rs.sreg_bool_narrowed.erase(106);
                rs.sreg_bool_b32.erase(106);
            } else {
                if (!rs.vcc) { ok = false; return true; }
                rs.vcc = b.bsel(in_written_half, result_bit, rs.vcc);
            }
            if (!scalar_low_only) {
                rs.sreg_bool[in.dst.value] = rs.vcc;
                rs.sreg_bool_narrowed[in.dst.value] = true;
            }
            return true;
        }

        uint32_t lane = b.guest_lane_id();
        lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
        const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
        const bool writes_hi = in.dst.value == 107;
        const uint32_t in_written_half = writes_hi
            ? b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32))
            : b.ucmp(Op_ULessThan, lane, b.uconst(32));
        auto mask_bit = [&](const Operand& source) -> uint32_t {
            if (source.kind == OperandKind::Special &&
                (source.value == 106 || source.value == 107)) {
                // Reading the same VCC half maps directly to this invocation's predicate.
                // Cross-half bit movement needs another guest lane and remains unsupported.
                if (source.value != in.dst.value) { ok = false; return b.bfalse(); }
                return rs.vcc;
            }
            const uint32_t raw = val(source);
            return b.ucmp(Op_INotEqual,
                          b.ibin(Op_BitwiseAnd,
                                 b.ibin(Op_ShiftRightLogical, raw, bit), b.uconst(1)),
                          b.uconst(0));
        };
        const uint32_t m0 = mask_bit(in.src[0]);
        const uint32_t m1 = mask_bit(in.src[1]);
        if (!ok) return true;
        const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
        const uint32_t x = b.bsel(m0, n1, m1);
        const uint32_t result = in.opcode == 0x0e ? b.land(m0, m1)
                              : in.opcode == 0x10 ? b.lor(m0, m1)
                              : in.opcode == 0x12 ? x
                              : in.opcode == 0x14 ? b.land(m0, n1)
                              : in.opcode == 0x16 ? b.lor(m0, n1)
                              : in.opcode == 0x18 ? b.lor(n0, n1)
                              : in.opcode == 0x1a ? b.land(n0, n1)
                              : b.logical_not(x);
        if (!rs.vcc) { ok = false; return true; }
        rs.vcc = b.bsel(in_written_half, result, rs.vcc);
        rs.sreg_bool[in.dst.value] = rs.vcc;
        rs.sreg_bool_narrowed[in.dst.value] = true;
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        rs.scc = 0; // SCC=(complete written dword != 0) is a guest-wave reduction
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
