// sop1_wqm_scan_routes.cpp — SOP1 routes: odd-pair s_cmov_b64, s_wqm numeric forms, s_quadmask_b64, s_ff1, EXEC-half moves.
// Relocated verbatim from emit_alu's SOP1 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop1Lowering::route_cmov_b64_odd_pair() {
    if (in.opcode == 0x0a &&
        ((in.dst.value & 1) ||
         ((in.src[0].kind == OperandKind::SGPR ||
           in.src[0].kind == OperandKind::Special) && (in.src[0].value & 1)))) {
        log_recompile_diagnostic(b.diagnostic, "wqm-unaligned-pair", "sop1",
                                 "pc=%u WQM_B64 requires aligned scalar pairs", in.pc);
        ok = false;
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_wqm_numeric_special_destination() {
    if (wqm_has_intrinsic_numeric_source(in) && !wqm_has_numeric_destination(in) &&
        !(in.src[0].kind == OperandKind::InlineInt &&
          (in.src[0].value == 0 || in.src[0].value == -1))) {
        log_recompile_diagnostic(b.diagnostic, "wqm-numeric-special-destination", "sop1",
                                 "pc=%u numeric WQM destination s%d is not supported",
                                 in.pc, in.dst.value);
        ok = false;
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_wqm_numeric() {
    if (wqm_has_numeric_destination(in) &&
        (wqm_has_intrinsic_numeric_source(in) ||
         wqm_has_scalar_numeric_source(rs, in))) {
        // Capture the complete source before a self/overlapping write ends its aliases.
        const uint32_t lo = val(in.src[0]);
        uint32_t hi = 0;
        if (in.opcode == 0x0a) {
            if (in.src[0].kind == OperandKind::SGPR) {
                Operand high = in.src[0];
                ++high.value;
                hi = val(high);
            } else {
                hi = b.uconst(in.src[0].kind == OperandKind::InlineInt &&
                              in.src[0].value < 0 ? UINT32_MAX : 0u);
            }
        }
        if (!ok) return true;
        const uint32_t widened_lo = numeric_wqm_word(b, lo);
        const uint32_t widened_hi = hi ? numeric_wqm_word(b, hi) : 0;
        invalidate_wqm_numeric_mask_aliases(rs, in.dst.value, hi ? 2 : 1);
        rs.sreg[in.dst.value] = widened_lo;
        if (hi) rs.sreg[in.dst.value + 1] = widened_hi;
        rs.scc = b.ucmp(Op_INotEqual,
            hi ? b.ibin(Op_BitwiseOr, widened_lo, widened_hi) : widened_lo,
            b.uconst(0));
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_quadmask_b64() {
    // s_quadmask_b64 D, S (#2412): D[i] = S[4i]|S[4i+1]|S[4i+2]|S[4i+3] for i in 0..15, upper
    // bits zero. Each output bit ORs a QUAD of lanes, so a per-invocation bool cannot form it.
    // `s_quadmask_b64 vcc, vcc` (0xbeea2d6a, confirmed with llvm-mc) is the first reject in 20
    // of GTA V's 59 failing shaders.
    //
    // A ballot gives the source mask exactly; the quadmask is then integer arithmetic over it.
    // BOTH representations prosper keeps for a mask register are produced -- the scalar DATA
    // value in sreg and the per-lane bool -- so it cannot matter which a later instruction
    // consumes; writing one and leaving the other stale would be silent wrong rendering.
    //
    // Wave64 only, and each stage supplies its exact-wave contract its own way: compute carries
    // native_subgroup_size, the fragment stage DECLARES fragment_required_subgroup_size. An
    // earlier revision gated on native_subgroup_size alone and therefore never fired for the
    // graphics shaders that need it -- dead code that ctest could not see, caught only by
    // measuring the failing-shader count. Vertex has neither mechanism and still rejects.
    if (in.opcode == 0x2d && b.wave_size == 64 &&
        (b.is_fragment || (b.native_subgroup_size == b.wave_size))) {
        uint32_t src_mask = 0;
        if (in.src[0].value == 106 || in.src[0].value == 107) src_mask = rs.vcc;
        else if (in.src[0].value == 126 || in.src[0].value == 127) src_mask = rs.exec;
        else if (in.src[0].kind == OperandKind::SGPR) {
            auto it = rs.sreg_bool.find(in.src[0].value);
            if (it != rs.sreg_bool.end()) src_mask = it->second;
        }
        const uint32_t lo = src_mask ? (b.is_fragment
                ? b.fragment_wave_ballot_half(src_mask, 0)
                : b.native_wave_ballot_half(src_mask, 0)) : 0;
        const uint32_t hi = src_mask ? (b.is_fragment
                ? b.fragment_wave_ballot_half(src_mask, 1)
                : b.native_wave_ballot_half(src_mask, 1)) : 0;
        if (lo && hi) {
            uint32_t value = b.uconst(0);
            for (uint32_t i = 0; i < 16; ++i) {
                const uint32_t word = i < 8 ? lo : hi;
                const uint32_t shift = (i < 8 ? i : i - 8) * 4u;
                const uint32_t nibble = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, word, b.uconst(shift)), b.uconst(0xFu));
                value = b.ibin(Op_BitwiseOr, value,
                               b.sel(b.ucmp(Op_INotEqual, nibble, b.uconst(0)),
                                     b.uconst(1u << i), b.uconst(0)));
            }
            const int dst = in.dst.value;
            rs.sreg[dst] = value;
            rs.sreg[dst + 1] = b.uconst(0);
            rs.sreg_srt.erase(dst); rs.sreg_srt.erase(dst + 1);
            const uint32_t lane = b.ibin(Op_BitwiseAnd, b.subgroup_local_id(),
                                         b.uconst(b.wave_size - 1));
            const uint32_t my_bit = b.ibin(
                Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, value, lane), b.uconst(1u));
            const uint32_t as_bool = b.land(
                b.ucmp(Op_ULessThan, lane, b.uconst(16u)),
                b.ucmp(Op_INotEqual, my_bit, b.uconst(0)));
            if (dst == 126 || dst == 127) { rs.exec = as_bool; rs.exec_narrowed = true; }
            else if (dst == 106 || dst == 107) rs.vcc = as_bool;
            else rs.sreg_bool[dst] = as_bool;
            rs.scc = b.ucmp(Op_INotEqual, value, b.uconst(0));
            return true;
        }
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_ff1_b64_native64() {
    if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 &&
        in.opcode == kSop1OpcodeFf1I32B64) { // s_ff1_i32_b64
        // RDNA2 scans the complete 64-bit source from its least-significant bit and
        // returns the first set index, or 0xffffffff for an empty mask.  The captured GTA
        // compute sites feed EXEC, VCC, or a VOPC-saved mask pair into this scalar result.
        // A required native Wave64 subgroup makes native_wave_first_active architectural;
        // a 32-wide or unknown subgroup would silently lose bits 32..63.
        const auto data_word_present = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        const int source = in.src[0].value;
        uint32_t mask = 0;
        // EXEC is architectural mask state, even when structured-loop or dispatcher state
        // bookkeeping also leaves synthetic scalar placeholders for physical s126/s127.
        // Unlike VCC and saved SGPR pairs, those words cannot carry competing scalar data.
        if (in.src[0].kind == OperandKind::Special && source == 126) {
            mask = rs.exec;
        } else {
            const bool proven_saved_mask =
                b.structured_wave64_mask_reduction_pcs.contains(in.pc);
            const bool competing_data = !proven_saved_mask &&
                (data_word_present(source) || data_word_present(source + 1));
            if (!competing_data && source == 106) {
                mask = rs.vcc;
            } else if (!competing_data && in.src[0].kind == OperandKind::SGPR &&
                       !rs.sreg_bool_b32.contains(source)) {
                const auto saved = rs.sreg_bool.find(source);
                if (saved != rs.sreg_bool.end()) mask = saved->second;
            }
        }
        if (!mask || in.dst.value == 126 || in.dst.value == 127) {
            ok = false;
            return true;
        }

        const uint32_t result = b.native_wave_first_active(mask);
        rs.sreg[in.dst.value] = result;
        rs.sreg_srt.erase(in.dst.value);

        // S_FF1 writes ordinary 32-bit scalar DATA.  End every complete-mask lifetime
        // overlapping that physical dword, including a pair rooted one register earlier;
        // otherwise a later implicit predicate consumer could observe the pre-S_FF1 mask.
        const auto erase_mask_alias = [&](int base) {
            rs.sreg_bool.erase(base);
            rs.sreg_bool_narrowed.erase(base);
            rs.sreg_bool_b32.erase(base);
        };
        erase_mask_alias(in.dst.value);
        if (in.dst.value > 0) erase_mask_alias(in.dst.value - 1);
        if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
        // S_FF1 does not modify SCC.
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_ff1_b32_data() {
    if (in.opcode == 0x13 && in.dst.value >= 0 && in.dst.value <= 105) {
        // AMD RDNA2 70648, 12.3 p113: FF1_B32 scans one scalar word from the LSB,
        // returns all ones for zero, and leaves SCC alone. Numeric data needs no vote.
        // Keep the existing mask path below: a Bool or a CFG placeholder is not a word.
        const Operand& source = in.src[0];
        uint32_t word = 0;
        if (source.kind == OperandKind::InlineInt ||
            source.kind == OperandKind::InlineFloat || source.kind == OperandKind::Literal) {
            word = val(source);
        } else if (source.kind == OperandKind::SGPR && source.value >= 0 &&
                   source.value <= 105 && !rs.sreg_bool.contains(source.value) &&
                   !(source.value > 0 && rs.sreg_bool.contains(source.value - 1) &&
                     !rs.sreg_bool_b32.contains(source.value - 1)) &&
                   !rs.sreg_wave64_mask_half.contains(source.value) &&
                   !rs.sreg_srt.contains(source.value) && !entry_m0_live(rs, source.value)) {
            if (rs.scalar_presence_has_no_placeholders) {
                const auto current = rs.sreg.find(source.value);
                if (current != rs.sreg.end()) word = current->second;
            }
            if (!word) {
                const auto input = rs.sreg_input.find(source.value);
                if (input != rs.sreg_input.end()) word = input->second;
            }
        }
        if (word) {
            // Capture first, including a self-source, then end every overlapping complete
            // mask lifetime. A B32 mask in the preceding register is an independent word.
            const uint32_t result = b.find_ilsb(word);
            // #3606: scanning a lane-local scalar cannot make it wave-uniform. Keep the
            // known source provenance when FindILsb and its bitcast create a fresh id.
            if (rs.lane_local_scalars.contains(word)) rs.lane_local_scalars.insert(result);
            const int dst = in.dst.value;
            const auto erase_mask = [&](int base) {
                rs.sreg_bool.erase(base);
                rs.sreg_bool_narrowed.erase(base);
                rs.sreg_bool_b32.erase(base);
            };
            erase_mask(dst);
            if (dst > 0 && !rs.sreg_bool_b32.contains(dst - 1)) erase_mask(dst - 1);
            rs.sreg[dst] = result;
            rs.sreg_srt.erase(dst);
            return true;
        }
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_ff1_b32_wave32() {
    if (b.is_compute && b.wave_size == 32 && b.native_subgroup_size == 32 &&
        in.opcode == 0x13) { // s_ff1_i32_b32
        // RDNA2 returns the first set bit from the low end, or 0xffffffff for an empty
        // source. In Wave32, VCC_LO/EXEC_LO and one-word saved masks are complete scalar
        // masks. The live Astro traversal kernel uses this to select the first hit lane.
        // A required 32-wide Vulkan subgroup makes the reduction architectural rather than
        // an implementation-width approximation.
        const auto data_value_present = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        uint32_t mask = 0;
        if (in.src[0].value == 126 && !data_value_present(126)) {
            mask = rs.exec;
        } else if (in.src[0].value == 106 && !data_value_present(106)) {
            mask = rs.vcc;
        } else if ((in.src[0].kind == OperandKind::SGPR ||
                    in.src[0].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[0].value) &&
                   !data_value_present(in.src[0].value)) {
            auto saved = rs.sreg_bool.find(in.src[0].value);
            if (saved != rs.sreg_bool.end()) mask = saved->second;
        }
        if (!mask || in.dst.value == 126 || in.dst.value == 127) {
            ok = false;
            return true;
        }

        const uint32_t result = b.native_wave_first_active(mask);
        rs.sreg[in.dst.value] = result;
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_bool.erase(in.dst.value);
        rs.sreg_bool_narrowed.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value);
        if (in.dst.value == 106) {
            // VCC_LO is also a physical scalar dword. Preserve its architectural mask view
            // for any implicit VCC consumer while publishing the same bits as scalar data.
            const uint32_t lane = b.subgroup_local_id();
            const uint32_t bit = b.ibin(
                Op_BitwiseAnd,
                b.ibin(Op_ShiftRightLogical, result, lane), b.uconst(1));
            rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
        }
        // S_FF1 does not modify SCC.
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_exec_half_from_vcc_half() {
    if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
        in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
        (in.src[0].value == 106 || in.src[0].value == 107) &&
        (in.dst.value - 126) == (in.src[0].value - 106)) {
        // Wave64 can update one EXEC dword without touching the other. Preserve that
        // distinction per invocation for the common VCC_LO->EXEC_LO / VCC_HI->EXEC_HI
        // copy. A cross-half copy would need another guest lane's predicate and remains
        // unsupported; the matching-half form maps directly to this lane's VCC bit.
        const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
        const uint32_t in_half = in.dst.value == 126
            ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
            : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
        rs.exec = b.bsel(in_half, rs.vcc, rs.exec);
        rs.exec_narrowed = true;
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_exec_half_immediate() {
    if ((b.is_compute || b.is_fragment) && b.wave_size == 64 &&
        in.opcode == 0x03 && (in.dst.value == 126 || in.dst.value == 127) &&
        in.src[0].kind == OperandKind::InlineInt) {
        // The companion immediate form restores or clears one EXEC dword. Select the
        // addressed 32-lane half and preserve the other half exactly; this covers the
        // compiler's `s_mov_b32 exec_lo, -1` reconvergence after a low-half mask.
        const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
        const uint32_t in_half = in.dst.value == 126
            ? b.ucmp(Op_ULessThan, lane, b.uconst(32))
            : b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
        const uint32_t value = in.dst.value == 126
            ? inline_int_mask_bit(b, in.src[0].value)
            : inline_int_mask_bit_hi(b, in.src[0].value);
        rs.exec = b.bsel(in_half, value, rs.exec);
        rs.exec_narrowed = true;
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
