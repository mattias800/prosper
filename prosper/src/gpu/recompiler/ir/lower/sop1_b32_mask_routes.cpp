// sop1_b32_mask_routes.cpp — SOP1 Wave32 one-word mask routes: s_mov/saveexec/wqm/not_b32 kept in the Bool domain.
// Relocated verbatim from emit_alu's SOP1 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop1Lowering::route_mov_b32_mask() {
    // Wave32 shaders use the low 32-bit halves of EXEC/VCC for the same save/copy/restore
    // idioms that wave64 shaders express with s_mov_b64.  A wave mask is one bool in this
    // per-invocation model, so preserve that bool domain when either source is an
    // unambiguous low-half mask or EXEC_LO is the destination.  VCC_LO remains available as
    // ordinary scalar scratch; when that scalar dword is copied back to EXEC_LO, select this
    // invocation's bit from the complete Wave32 value.  High-half moves remain unsupported:
    // without the guest wave mode they may name another lane rather than this invocation's bit.
    if (b.allow_b32_masks && in.opcode == 0x03) {   // s_mov_b32 (mask-domain form)
        const auto data_value_present = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        const bool src_exec_lo = in.src[0].value == 126;
        const bool src_vcc_lo = in.src[0].value == 106 &&
            !data_value_present(106);
        const auto saved = (in.src[0].kind == OperandKind::SGPR ||
                            in.src[0].kind == OperandKind::Special)
            ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
        const bool src_saved_mask = saved != rs.sreg_bool.end() &&
            rs.sreg_bool_b32.contains(in.src[0].value) &&
            !data_value_present(in.src[0].value);
        const bool dst_exec_lo = in.dst.value == 126;
        uint32_t scalar_mask_data = 0;
        bool src_scalar_data = false;
        if (dst_exec_lo &&
            (in.src[0].kind == OperandKind::SGPR ||
             (in.src[0].kind == OperandKind::Special &&
              in.src[0].value >= 106 && in.src[0].value <= 124))) {
            auto current = rs.sreg.find(in.src[0].value);
            if (current != rs.sreg.end()) {
                scalar_mask_data = current->second;
                src_scalar_data = true;
            } else {
                auto input = rs.sreg_input.find(in.src[0].value);
                if (input != rs.sreg_input.end()) {
                    scalar_mask_data = input->second;
                    src_scalar_data = true;
                }
            }
        }
        const bool mask_move = src_exec_lo || src_vcc_lo || src_saved_mask ||
            src_scalar_data ||
            (dst_exec_lo && in.src[0].kind == OperandKind::InlineInt);
        if (mask_move) {
            uint32_t mask = 0;
            bool narrowed = true;
            if (src_exec_lo) {
                mask = rs.exec;
                narrowed = rs.exec_narrowed;
            } else if (src_vcc_lo) {
                mask = rs.vcc;
                auto state = rs.sreg_bool_narrowed.find(106);
                narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
            } else if (src_saved_mask) {
                mask = saved->second;
                auto state = rs.sreg_bool_narrowed.find(in.src[0].value);
                narrowed = state == rs.sreg_bool_narrowed.end() || state->second;
            } else if (src_scalar_data) {
                const uint32_t lane = b.ibin(
                    Op_BitwiseAnd, b.guest_lane_id(), b.uconst(31));
                const uint32_t bit = b.ibin(
                    Op_BitwiseAnd,
                    b.ibin(Op_ShiftRightLogical, scalar_mask_data, lane),
                    b.uconst(1));
                mask = b.ucmp(Op_INotEqual, bit, b.uconst(0));
            } else if (in.src[0].kind == OperandKind::InlineInt) {
                // Every inline B32 dword is a valid Wave32 EXEC mask, not only 0/-1.
                // Select this invocation's architectural bit exactly; GTA V uses 1 here
                // to isolate lane zero before a scalar load.
                mask = inline_int_mask_bit(b, in.src[0].value);
                narrowed = in.src[0].value != -1;
            }
            if (!mask || in.dst.value == 127) {
                ok = false;
                return true;
            }

            if (dst_exec_lo) {
                rs.exec = mask;
                rs.exec_narrowed = narrowed;
            } else if (in.dst.value == 106) {
                rs.vcc = mask;
                rs.sreg_bool[106] = mask;
                rs.sreg_bool_narrowed[106] = narrowed;
                rs.sreg_bool_b32.insert(106);
            } else {
                rs.sreg_bool[in.dst.value] = mask;
                rs.sreg_bool_narrowed[in.dst.value] = narrowed;
                rs.sreg_bool_b32.insert(in.dst.value);
            }
            // A B32 move overwrites only the addressed physical word.  Remove stale scalar
            // data/descriptor provenance for that word while leaving its neighbor intact.
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            return true;
        }
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_mov_b32_exec_lo_full() {
    // `s_mov_b32 exec_lo, -1` while EXEC is still the full unnarrowed wave mask, at ANY
    // wave width and in any stage.
    //
    // Everything above needs to know the wave width, because "write all-ones to the low
    // EXEC dword" means the whole mask in Wave32 and only half of it in Wave64 -- and the
    // GRAPHICS wave size is not plumbed into the recompiler, so the vertex shell fails
    // closed (rdna2_to_spirv.cpp: `b.allow_b32_masks = b.ngg_one_lane`, an exception for
    // one byte-exact captured wrapper rather than a property of any stage).
    //
    // This case does not need it. `exec_narrowed == false` is prosper's invariant for "every
    // lane is on" (the field's own definition, and what the s_wqm_b32 path below and the
    // fragment export path in rdna2_to_spirv.cpp both already rely on). Setting the low
    // dword to all-ones from that state leaves every lane on under EITHER width: in Wave32
    // it rewrites the whole mask with what it already held, and in Wave64 it rewrites the
    // low half with what it already held and does not touch the high half. So the exact
    // lowering is a no-op, derived rather than approximated -- no lane mapping, no wave
    // width, no peer lane.
    //
    // It is the compiler's standard program prologue, which is why it gates whole titles
    // rather than odd shaders: Yakuza Kiwami (PPSA31334) emits `bfa00003` (s_branch) then
    // `befe03c1` (this instruction) at pc=3 in 19 of the 20 distinct programs a boot
    // reaches, and every draw was skipped for it.
    //
    // Deliberately NOT extended to the narrowed case. Restoring all-lanes from a NARROWED
    // EXEC is the Wave32 reconvergence idiom, and reading it that way in a Wave64 program
    // would silently re-enable the wrong lanes -- that one does need the wave size, and
    // stays fail-closed. CONFIDENCE: HIGH.
    if (in.opcode == kSop1OpcodeMovB32 && in.dst.value == 126 && !rs.exec_narrowed &&
        in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
        rs.exec = b.btrue();
        rs.exec_narrowed = false;
        rs.sreg.erase(126);
        rs.sreg_srt.erase(126);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_saveexec_b32() {
    if (b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
        sop1_opcode_is_emitted_saveexec_b32(in.opcode)) {
        // s_and_saveexec_b32 / s_orn2_saveexec_b32 / s_andn1_saveexec_b32
        // Save the previous EXEC_LO into one physical SGPR, then narrow EXEC_LO by the
        // one-word source mask. Astro uses VCC_HI as the saved-mask destination and restores
        // it later with s_mov_b32. The ANDN1 form selects the complementary old-EXEC arm.
        // Both halves are independent complete values in Wave32.
        auto source_mask = [&]() -> uint32_t {
            if (in.src[0].value == 126) return rs.exec;
            if ((in.src[0].kind == OperandKind::SGPR ||
                 in.src[0].kind == OperandKind::Special) &&
                rs.sreg_bool_b32.contains(in.src[0].value)) {
                auto it = rs.sreg_bool.find(in.src[0].value);
                if (it != rs.sreg_bool.end()) return it->second;
                if (in.src[0].value == 106) return rs.vcc;
            }
            if (in.src[0].kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, in.src[0].value);
            return 0;
        };
        const uint32_t mask = source_mask();
        if (!mask || in.dst.value == 126 || in.dst.value == 127) {
            ok = false; return true;
        }
        const uint32_t old_exec = rs.exec;
        rs.sreg_bool[in.dst.value] = old_exec;
        rs.sreg_bool_narrowed[in.dst.value] = rs.exec_narrowed;
        rs.sreg_bool_b32.insert(in.dst.value);
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        if (in.dst.value == 106) rs.vcc = old_exec;
        rs.exec = in.opcode == kSop1OpcodeAndSaveexecB32 ? b.land(old_exec, mask)
                : in.opcode == kSop1OpcodeOrn2SaveexecB32
                    ? b.lor(mask, b.logical_not(old_exec))
                                    : b.land(old_exec, b.logical_not(mask));
        rs.exec_narrowed = true;
        rs.scc = b.is_fragment ? b.fragment_wave_any(rs.exec) : 0;
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_wqm_b32() {
    if (b.allow_b32_masks && in.opcode == 0x09) {   // s_wqm_b32
        // Wave32 uses the low half of EXEC for the same whole-quad-mode idiom as
        // s_wqm_b64.  In the per-invocation fragment model helper lanes are implicit, so
        // widening is an identity on the tracked bool.  Keep this in the mask domain and
        // reject either high half; treating EXEC_LO as scalar data drops Astro's material
        // fragments before their first sample.
        const auto data_value_present = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        const bool src_exec_lo = in.src[0].value == 126;
        const bool src_vcc_lo = in.src[0].value == 106 &&
            !data_value_present(106);
        const auto saved = in.src[0].kind == OperandKind::SGPR
            ? rs.sreg_bool.find(in.src[0].value) : rs.sreg_bool.end();
        const bool src_saved_mask = saved != rs.sreg_bool.end() &&
            !data_value_present(in.src[0].value);
        uint32_t mask = 0;
        if (src_exec_lo) mask = rs.exec;
        else if (src_vcc_lo) mask = rs.vcc;
        else if (src_saved_mask) mask = saved->second;
        else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1)
            mask = b.btrue();
        else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == 0)
            mask = b.bfalse();
        if (!mask || in.dst.value == 127 || in.dst.value == 107) {
            ok = false;
            return true;
        }

        // ISA SCC is a reduction over the whole resulting wave mask.  That value cannot be
        // recovered from one invocation, so poison it exactly like s_wqm_b64 does.
        rs.scc = 0;
        if (in.dst.value == 126) {
            if (src_exec_lo) { /* exec <- wqm(exec): tracked identity */ }
            else if (in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1) {
                rs.exec = b.btrue();
                rs.exec_narrowed = false;
            } else {
                rs.exec = mask;
                rs.exec_narrowed = true;
            }
        } else if (in.dst.value == 106) {
            rs.vcc = mask;
            rs.sreg_bool[106] = mask;
            rs.sreg_bool_narrowed[106] = true;
            rs.sreg_bool_b32.insert(106);
        } else {
            rs.sreg_bool[in.dst.value] = mask;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            rs.sreg_bool_b32.insert(in.dst.value);
        }
        rs.sreg.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop1Lowering::route_not_b32_mask() {
    if (b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32)) &&
        in.opcode == 0x07) {   // s_not_b32 (mask-domain form)
        // Wave32 shader compilers also invert a saved one-word mask in an ordinary SGPR
        // before combining it back into VCC. Astro's world-map kernel does exactly this:
        //   s_mov_b32 s20, exec_lo
        //   ...
        //   s_not_b32 vcc_lo, s20
        //   s_nor_b32 vcc_lo, vcc_lo, vcc_hi
        // The old special case below only recognized `s_not_b32 vcc_lo,vcc_lo`; routing the
        // saved-SGPR form through scalar DATA either rejected (there are no uint bits for a
        // per-lane mask) or left a stale VCC lifetime at a dispatcher boundary. Accept only
        // an unambiguous complete Wave32 mask source with no competing scalar-data value.
        auto data_value_present = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        uint32_t source_mask = 0;
        if (in.src[0].value == 126) {
            source_mask = rs.exec;
        } else if ((in.src[0].kind == OperandKind::SGPR ||
                    in.src[0].kind == OperandKind::Special) &&
                   rs.sreg_bool_b32.contains(in.src[0].value) &&
                   !data_value_present(in.src[0].value)) {
            auto found = rs.sreg_bool.find(in.src[0].value);
            if (found != rs.sreg_bool.end()) source_mask = found->second;
            else if (in.src[0].value == 106) source_mask = rs.vcc;
        }
        if (source_mask && in.dst.value != 127) {
            const uint32_t result = b.logical_not(source_mask);
            if (in.dst.value == 126) {
                rs.exec = result;
                rs.exec_narrowed = true;
            } else {
                rs.sreg_bool[in.dst.value] = result;
                // Complementing an arbitrary mask may activate any previously-clear lane;
                // it is not a proof that the result is the full wave.
                rs.sreg_bool_narrowed[in.dst.value] = true;
                rs.sreg_bool_b32.insert(in.dst.value);
                if (in.dst.value == 106) rs.vcc = result;
            }
            rs.sreg.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            rs.scc = 0; // SCC=(complete written dword != 0) needs a guest-wave reduction.
            return true;
        }
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
