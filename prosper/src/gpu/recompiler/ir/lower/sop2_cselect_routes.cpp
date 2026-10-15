// sop2_cselect_routes.cpp — SOP2 select routes: s_cselect_b32 into VCC/EXEC halves and the s_cselect_b64 mask and pair forms.
// Relocated verbatim from emit_alu's SOP2 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop2Lowering::route_wave64_vcc_lo_cselect() {
    if ((b.is_compute || (b.is_fragment && !b.vcc_b32_low_only_pcs.contains(in.pc) &&
                          (is_wave64_vcc_lo_scalar_cselect(in) ||
                           is_wave64_vcc_lo_cselect_reading_vcc_hi(in)))) &&
        b.wave_size == 64 && in.opcode == 0x0a && in.dst.kind == OperandKind::SGPR &&
        in.dst.value == 106) {
        // A B32 write leaves Wave64 VCC_HI untouched, so the resulting mixed physical pair
        // can remain in the architectural mask domain only when the dispatcher proves that
        // sibling is scalar data at this exact PC. Otherwise admit only a whole-stream
        // low-only proof and poison VCC until its later complete replacement.
        const bool path_local_scalar_pair =
            rs.scalar_presence_has_no_placeholders && rs.sreg.contains(107);
        const bool complete_scalar_pair =
            b.is_fragment
                ? vcc_b32_scalar_pair_defined(b, rs, in)
                : path_local_scalar_pair || b.vcc_b32_scalar_pair_pcs.contains(in.pc);
        if ((!is_wave64_vcc_lo_scalar_cselect(in) &&
             !(complete_scalar_pair && rs.sreg.contains(107) &&
               is_wave64_vcc_lo_cselect_reading_vcc_hi(in))) ||
            (!complete_scalar_pair && !b.vcc_b32_low_only_pcs.contains(in.pc)) || !rs.scc ||
            (b.is_fragment && !scalar_condition_source_defined(b, rs))) {
            ok = false; return true;
        }
        const uint32_t selected = b.sel(
            rs.scc, val(in.src[0]), val(in.src[1]));
        if (!ok) return true;
        rs.sreg[106] = selected;
        rs.sreg_srt.erase(106);
        rs.sreg_bool.erase(106);
        rs.sreg_bool_narrowed.erase(106);
        rs.sreg_bool_b32.erase(106);
        if (complete_scalar_pair) {
            auto high = rs.sreg.find(107);
            if (vcc_sibling_unavailable(b, rs, 107, high, ok)) return true;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
            const uint32_t word = b.sel(
                b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                high->second, selected);
            const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
            rs.vcc = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd,
                       b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                b.uconst(0));
            rs.sreg_bool[106] = rs.vcc;
            rs.sreg_bool_narrowed[106] = true;
        } else {
            rs.vcc = 0;
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_b32_mask_cselect() {
    if (b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
        in.opcode == 0x0a &&
        (in.dst.value == 126 ||
         in.src[0].value == 126 || in.src[1].value == 126 ||
         (((in.src[0].kind == OperandKind::SGPR ||
            in.src[0].kind == OperandKind::Special) &&
           rs.sreg_bool_b32.contains(in.src[0].value))) ||
         (((in.src[1].kind == OperandKind::SGPR ||
            in.src[1].kind == OperandKind::Special) &&
           rs.sreg_bool_b32.contains(in.src[1].value))))) {
        // Wave32 s_cselect_b32 can select complete one-word wave masks. Astro's material
        // setup uses `s_cselect_b32 vcc_hi, exec_lo, 0` after a scalar mode comparison,
        // then consumes that VCC_HI mask with VOP3 cndmask instructions. Keep the select in
        // the Bool domain; routing either operand through scalar bits would lose lane state.
        auto mask = [&](const Operand& source) -> uint32_t {
            if (source.value == 126) return rs.exec;
            if ((source.value == 106 || source.value == 107) &&
                rs.sreg_bool_b32.contains(source.value)) {
                auto it = rs.sreg_bool.find(source.value);
                if (it != rs.sreg_bool.end()) return it->second;
                if (source.value == 106) return rs.vcc;
            }
            if (source.kind == OperandKind::SGPR &&
                rs.sreg_bool_b32.contains(source.value)) {
                auto it = rs.sreg_bool.find(source.value);
                if (it != rs.sreg_bool.end()) return it->second;
            }
            if (source.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, source.value);
            return 0;
        };
        const uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
        if (!m0 || !m1 || !rs.scc || in.dst.value == 127) {
            ok = false; return true;
        }
        const uint32_t result = b.bsel(rs.scc, m0, m1);
        if (in.dst.value == 126) {
            rs.exec = result;
            rs.exec_narrowed = true;
        } else {
            rs.sreg_bool[in.dst.value] = result;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            rs.sreg_bool_b32.insert(in.dst.value);
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            if (in.dst.value == 106) rs.vcc = result;
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_cselect_b32_vcc_lo_bridge() {
    if (b.is_compute && is_scalar_cselect_b32_to_vcc_lo(in)) {
        // The mask-domain cselect above remains first: this bridge is only for the distinct
        // scalar-data lifetime seen in GTA V. Preserve the full selected dword in `sreg`,
        // publish its bit for this guest lane as VCC, and leave SCC unchanged as required by
        // S_CSELECT_B32. Wave64 cannot represent both dwords this way and stays rejected.
        if (!allows_compute_scalar_vcc_bridge(b)) { ok = false; return true; }
        if (!rs.scc) { ok = false; return true; }
        const uint32_t selected = b.sel(rs.scc, val(in.src[0]), val(in.src[1]));
        if (!ok) return true;
        rs.sreg[106] = selected;
        rs.sreg_srt.erase(106);
        const uint32_t lane = b.ibin(
            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
        const uint32_t bit = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, selected, lane), b.uconst(1));
        rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        rs.sreg_bool[106] = rs.vcc;
        rs.sreg_bool_narrowed[106] = true;
        rs.sreg_bool_b32.insert(106);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_cselect_b64() {
    if (in.opcode == 0x0b) {   // s_cselect_b64: SCC ? src0 : src1 (mask or scalar-pair domain)
        // Operands/dest are wave masks (EXEC/VCC/saved/inline), NOT uint bits — resolve like the
        // SOP1 mask ops and select in the bool domain. Keep this path first so established
        // mask lifetimes retain byte-identical behavior even when Function-backed scalar
        // placeholders coexist in a CFG dispatcher. (s_cselect_b32 stays in the uint path.)
        auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
        auto mask = [&](const Operand& o) -> uint32_t {
            if ((o.value == 106 || o.value == 107) && rs.vcc) return rs.vcc;
            if (o.value == 126 || o.value == 127) return rs.exec;
            if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second; }
            if (o.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, o.value);
            return 0;   // not a recognizable mask
        };
        uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
        if (m0 && m1) {
            if (!rs.scc) { ok = false; return true; }   // SCC poisoned by a mask op
            uint32_t r = b.bsel(rs.scc, m0, m1);
            if (is_exec(in.dst)) { rs.exec = r; rs.exec_narrowed = true; }
            else if (in.dst.value == 106 || in.dst.value == 107) { rs.vcc = r; rs.sreg_bool_narrowed[in.dst.value] = true;
                                                                   mask_write_clobbers_pair(rs, in.dst.value); }
            else { rs.sreg_bool[in.dst.value] = r; rs.sreg_bool_narrowed[in.dst.value] = true;   // conservative flag
                   mask_write_clobbers_pair(rs, in.dst.value); }
            return true;
        }

        // Ordinary scalar-data pairs are selected word-for-word. Unlike EXEC, VCC has a
        // dual role: its two physical scalar dwords remain readable as data, while vector
        // instructions consume the bit for this invocation. Materialize both views from
        // the same selected pair so native and portable compute paths agree exactly.
        struct ScalarPair {
            uint32_t lo = 0, hi = 0;
            bool have_lo = false, have_hi = false;
        };
        auto scalar_pair = [&](const Operand& source) {
            ScalarPair pair;
            if (source.kind == OperandKind::SGPR ||
                (source.kind == OperandKind::Special &&
                 source.value >= 106 && source.value < 124)) {
                pair.have_lo = native_logical_data_word(rs, source.value, pair.lo);
                pair.have_hi = native_logical_data_word(rs, source.value + 1, pair.hi);
            } else if (source.kind == OperandKind::InlineInt) {
                pair.lo = b.uconst(static_cast<uint32_t>(source.value));
                pair.hi = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
                pair.have_lo = pair.have_hi = true;
            } else if (source.kind == OperandKind::Literal) {
                pair.lo = b.uconst(in.literal);
                pair.hi = b.uconst(0);
                pair.have_lo = pair.have_hi = true;
            }
            return pair;
        };
        if (is_exec(in.dst) || !rs.scc) { ok = false; return true; }
        const ScalarPair p0 = scalar_pair(in.src[0]);
        const ScalarPair p1 = scalar_pair(in.src[1]);
        const bool complete = p0.have_lo && p0.have_hi && p1.have_lo && p1.have_hi;
        const bool low_only = p0.have_lo && p1.have_lo &&
            b.is_compute && b.wave_size == 64 &&
            b.cselect_b64_low_only_pcs.contains(in.pc);
        if (!complete && !low_only) { ok = false; return true; }

        const uint32_t selected_lo = b.sel(rs.scc, p0.lo, p1.lo);
        const uint32_t selected_hi = complete
            ? b.sel(rs.scc, p0.hi, p1.hi) : b.uconst(0);
        rs.sreg[in.dst.value] = selected_lo;
        if (complete) rs.sreg[in.dst.value + 1] = selected_hi;
        else rs.sreg.erase(in.dst.value + 1);
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value + 1);

        // A scalar pair write ends every overlapping saved-mask/B32 lifetime. The complete
        // VCC form immediately installs its newly-derived predicate below; the GTA low-only
        // form deliberately leaves VCC unavailable because the CFG proof says no implicit
        // mask consumer can observe it.
        auto erase_mask_alias = [&](int base) {
            rs.sreg_bool.erase(base);
            rs.sreg_bool_narrowed.erase(base);
            rs.sreg_bool_b32.erase(base);
        };
        if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
        erase_mask_alias(in.dst.value);
        erase_mask_alias(in.dst.value + 1);
        if (in.dst.value == 106) {
            if (complete) {
                uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
                const uint32_t high = b.ucmp(
                    Op_UGreaterThanEqual, lane, b.uconst(32));
                const uint32_t word = b.sel(high, selected_hi, selected_lo);
                const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
                rs.vcc = b.ucmp(
                    Op_INotEqual,
                    b.ibin(Op_BitwiseAnd,
                           b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                    b.uconst(0));
                rs.sreg_bool[106] = rs.vcc;
                rs.sreg_bool_narrowed[106] = true;
            } else {
                rs.vcc = 0;
            }
        }
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
