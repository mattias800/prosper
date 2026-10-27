// vop3_cross_lane.cpp — emit_alu's VOP3 cross-lane cases, moved verbatim by
// tools/refactor/extract_case_routes.py. Their order in lower() is the old `if` chain's order.
// v_readlane_b32's native-subgroup value goes through ops/vector_cross_lane.cpp; every scalar
// write, lane-slot, mask-half and retirement step around it is the moved text, unchanged.
#include "gpu/recompiler/ir/lower/vop3_cross_lane_lowering_internal.hpp"
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"
#include "gpu/recompiler/ir/ops/vector_permlane.hpp"

#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop3CrossLaneLowering::route_writelane() {
    // SCALAR-SPILL lane slots (#273): v_writelane_b32 (0x361) / v_readlane_b32 (0x360) with a
    // COMPILE-TIME lane index — the pack-scalars-into-a-VGPR's-lanes idiom (DOLL's big post PS
    // spills 19 s_buffer_load results into v36 and reads them back). Per-invocation each
    // (vgpr, lane) is a named wave-uniform scalar. An exact native compute wave also supports
    // a dynamic writelane selector by updating the one matching subgroup invocation in an
    // ordinary VGPR lifetime. The portable NGG vertex shell additionally projects an ordinary
    // VGPR read from any physical lane onto its one represented live lane; this is the same
    // collapsed-wave contract used for MBCNT and Function LDS. Other compute/fragment stages
    // use a native subgroup shuffle for ordinary VGPR reads and therefore support both scalar
    // and inline lane selectors. Neither op is EXEC-predicated on hardware, so no
    // predicate_write.
    // VERIFIED(round-trip llvm-mc gfx1030: 0xd761 v_writelane_b32 / 0xd760 v_readlane_b32).
    if (in.opcode == 0x361) {                                 // v_writelane_b32 vDST, sSRC, lane
        if (b.is_compute && (b.wave_size == 32 || b.wave_size == 64) &&
            b.native_subgroup_size == b.wave_size &&
            in.src[1].kind != OperandKind::InlineInt) {
            // RDNA2 masks the scalar selector to the active wave width and replaces VDST
            // only in that physical lane. Under the enforced exact-width native-subgroup
            // contract, SubgroupLocalInvocationId is that architectural lane. The scalar
            // source and selector are uniform, so this needs no shuffle or barrier. Astro's
            // Wave32 and GTA V's Wave64 traversal kernels reach this form after readlane or
            // a mask scan publishes the chosen hit as scalar data.
            if (in.src[1].kind == OperandKind::VGPR ||
                rs.vgpr_lane_slots.count(in.dst.value) ||
                rs.vgpr_lane_mask_slots.count(in.dst.value)) {
                ok = false; return true;
            }
            const uint32_t value = val(in.src[0]);
            const uint32_t selector = val(in.src[1]);
            if (!ok) return true;
            rs.vreg[in.dst.value] = emit_writelane_b32(
                b, value, selector, b.wave_size, [&] { return vreg_old(b, rs, in.dst.value); });
            rs.invalidated_vgpr_lane_slots.erase(in.dst.value);
            rs.vgpr_lane_slots.erase(in.dst.value);
            rs.vgpr_lane_mask_slots.erase(in.dst.value);
            return true;
        }
        if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 || in.src[1].value > 63) {
            ok = false; return true;
        }
        const int lane = in.src[1].value;
        rs.invalidated_vgpr_lane_slots.erase(in.dst.value);
        uint32_t mask_value = 0;
        bool mask_source = false;
        const bool proven_half_alias =
            b.wave64_mask_writelane_alias_pcs.contains(in.pc);
        if (proven_half_alias) {
            const auto half = rs.sreg_wave64_mask_half.find(in.src[0].value);
            if (half == rs.sreg_wave64_mask_half.end()) {
                ok = false; return true;
            }
            mask_value = half->second;
            mask_source = true;
        } else if (in.src[0].kind == OperandKind::SGPR && in.src[0].value > 0 &&
                   b.wave64_mask_high_half_writelane_pcs.contains(in.pc) &&
                   !saved_mask_spill_uses_data(rs, in.src[0].value)) {
            // The HIGH dword of a MUST saved-mask pair with no retained physical word
            // (SpillSlotDomain::MaskHi, proven by the dispatcher). Its value is the upper
            // ballot word of the pair's Bool, so the slot stores that Bool; a reload the
            // dispatcher proves is this half (wave64_mask_readlane_half_for_pc) takes the
            // upper ballot word and, beside the low half, rebuilds the pair. Without the
            // Bool there is nothing exact to store: refuse rather than spill a placeholder.
            const int root = in.src[0].value - 1;
            const auto saved = rs.sreg_bool.find(root);
            if (saved == rs.sreg_bool.end() || rs.sreg_bool_b32.contains(root)) {
                ok = false;
                return true;
            }
            mask_value = saved->second;
            mask_source = true;
        } else if (in.src[0].value == 126 || in.src[0].value == 127) {
            mask_value = rs.exec; mask_source = true;
        } else if ((in.src[0].value == 106 || in.src[0].value == 107) &&
                   !rs.sreg.count(in.src[0].value)) {
            if (b.is_compute && b.wave_size == 64 &&
                b.native_subgroup_size == 64) {
                // A physical VCC word has no runtime mask/data tag. The Wave64 half proof
                // deliberately seeds only EXEC; until VCC is tied to the mask-domain MUST
                // analysis, persisting this spill could resurrect a false Bool placeholder.
                ok = false; return true;
            }
            mask_value = rs.vcc; mask_source = true;
        } else if (in.src[0].kind == OperandKind::SGPR &&
                   !saved_mask_spill_uses_data(rs, in.src[0].value) &&
                   !(b.is_fragment && b.wave_size == 64 &&
                     (rs.sreg_bool.contains(in.src[0].value) ||
                      (in.src[0].value > 0 &&
                       rs.sreg_bool.contains(in.src[0].value - 1))))) {
            // A Wave64 fragment V_WRITELANE consumes one physical scalar word, not the
            // complete Bool alias rooted at the pair's low SGPR. val() must consume a
            // retained physical word below. Portable stages retain the mask-slot model.
            auto saved = rs.sreg_bool.find(in.src[0].value);
            if (saved != rs.sreg_bool.end()) { mask_value = saved->second; mask_source = true; }
        }
        rs.vreg.erase(in.dst.value);                            // spill-array lifetime
        if (mask_source) {
            rs.vgpr_lane_mask_slots[in.dst.value][lane] = mask_value;
            auto data = rs.vgpr_lane_slots.find(in.dst.value);
            if (data != rs.vgpr_lane_slots.end()) {
                data->second.erase(lane);
                if (data->second.empty()) rs.vgpr_lane_slots.erase(data);
            }
        } else {
            rs.vgpr_lane_slots[in.dst.value][lane] = val(in.src[0]);
            auto masks = rs.vgpr_lane_mask_slots.find(in.dst.value);
            if (masks != rs.vgpr_lane_mask_slots.end()) {
                masks->second.erase(lane);
                if (masks->second.empty()) rs.vgpr_lane_mask_slots.erase(masks);
            }
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop3CrossLaneLowering::route_readlane() {
    if (in.opcode == 0x360) {                                 // v_readlane_b32 sDST, vSRC, lane
        if (refuse_arbitrary_placeholder_lane_read(b, rs, in, ok)) return true;
        // V_READLANE is an architectural scalar write.  Drop an older dedicated half
        // before selecting this read's source domain; only the exact CFG-proven mask-slot
        // path below may publish a replacement.  record_scalar_write consequently knows
        // that map presence after emission describes this instruction, not stale state.
        expire_wave64_mask_half(rs, in.dst.value);
        if (b.is_fragment && b.wave_size == 32 &&
            in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 32) {
            ok = false; return true;
        }
        auto vit = rs.vgpr_lane_slots.find(in.src[0].value);
        auto mit = rs.vgpr_lane_mask_slots.find(in.src[0].value);
        const bool vertex_shell = !b.is_compute && !b.is_fragment;
        if (!vertex_shell && vit == rs.vgpr_lane_slots.end() &&
            mit == rs.vgpr_lane_mask_slots.end()) {
            auto source = rs.vreg.find(in.src[0].value);
            if (source == rs.vreg.end()) { ok = false; return true; }
            const uint32_t selector = val(in.src[1]);
            if (!ok) return true;
            uint32_t result = 0;
            if (allow_wave && b.is_compute &&
                b.native_subgroup_size != b.wave_size) {
                // A workgroup-uniform site can model the guest wave exactly through
                // scratch even when a Wave64 guest does not fit in the host subgroup.
                result = b.guest_wave_readlane(source->second, selector);
            } else {
                // The subgroup-size contract is the lowering's; the value is the definition's.
                if (b.wave_size == 64) b.mark_subgroup_min64();
                else b.mark_subgroup_min32();
                result = emit_readlane_b32(b, source->second, selector, b.wave_size);
            }
            rs.sreg[in.dst.value] = result;
            if (b.wave_size == 64)
                retire_scalar_data_word_mask(rs, in.dst.value);
            else {
                rs.sreg_bool.erase(in.dst.value);
                rs.sreg_bool_narrowed.erase(in.dst.value);
            }
            rs.sreg_srt.erase(in.dst.value);
            if (in.dst.value == 106 && b.wave_size != 64) {
                const uint32_t bit = b.ibin(Op_BitwiseAnd, result, b.uconst(1));
                rs.vcc = b.ucmp(Op_INotEqual, bit, b.uconst(0));
                rs.sreg_bool_narrowed[106] = true;
            }
            return true;
        }
        const bool static_lane = in.src[1].kind == OperandKind::InlineInt &&
                                 in.src[1].value >= 0 && in.src[1].value <= 63;
        const bool dynamic_absent_ngg_peer = b.ngg_one_lane &&
            in.src[1].kind == OperandKind::Special &&
            in.src[1].value >= 106 && in.src[1].value <= 124;
        if (!static_lane && !dynamic_absent_ngg_peer) {
            ok = false; return true;
        }
        const int lane = static_lane ? in.src[1].value : -1;
        if (mit != rs.vgpr_lane_mask_slots.end()) {
            auto sit = mit->second.find(lane);
            if (sit != mit->second.end()) {
                const auto proven_half =
                    b.wave64_mask_readlane_half_for_pc.find(in.pc);
                if (native_mask_half_readlane_has_physical_launch(b, in) &&
                    proven_half != b.wave64_mask_readlane_half_for_pc.end()) {
                    // This is one scalar dword plus one Bool spelling of the SAME physical
                    // half, not a complete saved B64 predicate. End every overlapping full
                    // mask alias and retain the Bool only in the dedicated re-spill domain.
                    auto erase_full_mask = [&](int base) {
                        rs.sreg_bool.erase(base);
                        rs.sreg_bool_narrowed.erase(base);
                        rs.sreg_bool_b32.erase(base);
                    };
                    if (in.dst.value > 0) erase_full_mask(in.dst.value - 1);
                    erase_full_mask(in.dst.value);
                    if (in.dst.value == 106 || in.dst.value == 107) rs.vcc = 0;
                    rs.sreg_wave64_mask_half[in.dst.value] = sit->second;
                    rs.sreg_wave64_mask_half_index[in.dst.value] = proven_half->second;
                    // Override the scalar placeholder: a dual-domain spill lane's uint
                    // Function variable is zero on mask-only paths.
                    rs.sreg[in.dst.value] = b.native_wave_ballot_half(
                        sit->second, proven_half->second);
                    // Two adjacent scalar words become one complete B64 predicate only in
                    // architectural LO/HI order. Combine their per-lane Bool spellings;
                    // they need not originate from the same historical EXEC value.
                    const int base = in.dst.value & ~1;
                    const auto low = rs.sreg_wave64_mask_half.find(base);
                    const auto high = rs.sreg_wave64_mask_half.find(base + 1);
                    const auto low_index =
                        rs.sreg_wave64_mask_half_index.find(base);
                    const auto high_index =
                        rs.sreg_wave64_mask_half_index.find(base + 1);
                    if (low != rs.sreg_wave64_mask_half.end() &&
                        high != rs.sreg_wave64_mask_half.end() &&
                        low_index != rs.sreg_wave64_mask_half_index.end() &&
                        high_index != rs.sreg_wave64_mask_half_index.end() &&
                        low_index->second == 0 && high_index->second == 1) {
                        const uint32_t lane = b.ibin(
                            Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
                        rs.sreg_bool[base] = b.bsel(
                            b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                            high->second, low->second);
                        rs.sreg_bool_narrowed[base] = true;
                    }
                } else {
                    if (b.is_compute && uses_native_saved_mask_words(b)) {
                        refuse_unavailable_saved_mask_word(b, in, &ok);
                        return true;   // one unproved dword cannot supply a complete B64 alias
                    }
                    rs.sreg_wave64_mask_half.erase(in.dst.value);
                    rs.sreg_wave64_mask_half_index.erase(in.dst.value);
                    rs.sreg_bool[in.dst.value] = sit->second;
                    rs.sreg_bool_narrowed[in.dst.value] = true;
                }
                if (proven_half == b.wave64_mask_readlane_half_for_pc.end() &&
                    vit != rs.vgpr_lane_slots.end()) {
                    auto data = vit->second.find(lane);
                    if (data != vit->second.end()) rs.sreg[in.dst.value] = data->second;
                    else rs.sreg.erase(in.dst.value);
                } else if (proven_half ==
                           b.wave64_mask_readlane_half_for_pc.end()) {
                    rs.sreg.erase(in.dst.value);
                }
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
        }
        if (vit == rs.vgpr_lane_slots.end()) {
            if (rs.invalidated_vgpr_lane_slots.count(in.src[0].value)) {
                ok = false; return true;
            }
            // Fragment shaders execute in an enforced 64-lane Vulkan subgroup, so a static
            // hardware v_readlane maps exactly to a subgroup shuffle. Unlike ordinary VALU,
            // v_readlane ignores EXEC; do not mask the source or predicate the scalar write.
            if (b.is_fragment && static_lane) {
                rs.sreg[in.dst.value] = b.subgroup_shuffle(
                    val(in.src[0]), b.uconst(static_cast<uint32_t>(lane)));
                if (b.wave_size == 64)
                    retire_scalar_data_word_mask(rs, in.dst.value);
                else {
                    rs.sreg_bool.erase(in.dst.value);
                    rs.sreg_bool_narrowed.erase(in.dst.value);
                }
                rs.sreg_srt.erase(in.dst.value);
                return true;
            }
            // NGG's final wave-packing tail reads peer lanes (Astro Bot uses lanes 63 and 3)
            // from an ordinary VGPR. A Vulkan vertex invocation models guest lane zero: a
            // lane-zero read returns this invocation's source, while every absent peer reads
            // zero. The dynamic form remains exact by selecting on its scalar lane index.
            if (!b.ngg_one_lane) { ok = false; return true; }
            const uint32_t source = val(in.src[0]);
            uint32_t reads_self = 0;
            if (static_lane) {
                reads_self = lane == 0 ? b.btrue() : b.bfalse();
            } else {
                auto lane_value = rs.sreg.find(in.src[1].value);
                if (lane_value == rs.sreg.end()) { ok = false; return true; }
                reads_self = b.ucmp(Op_IEqual, lane_value->second, b.uconst(0));
            }
            rs.sreg[in.dst.value] = b.sel(reads_self, source, b.uconst(0));
            rs.sreg_bool.erase(in.dst.value);
            rs.sreg_bool_narrowed.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value);
            return true;
        }
        auto sit = vit->second.find(lane);
        if (sit == vit->second.end()) { ok = false; return true; }          // slot never written
        rs.sreg[in.dst.value] = sit->second;   // dst field is the SGPR number (like readfirstlane)
        if (!vertex_shell && b.wave_size == 64)
            retire_scalar_data_word_mask(rs, in.dst.value);
        else {
            rs.sreg_bool.erase(in.dst.value);
            rs.sreg_bool_narrowed.erase(in.dst.value);
        }
        rs.sreg_srt.erase(in.dst.value);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop3CrossLaneLowering::route_permlane16() {
    if (in.opcode == 0x377 || in.opcode == 0x378) {            // v_permlane{x}16_b32
        // AMD RDNA ISA 12.12: SRC1:SRC2 is a packed table of sixteen four-bit lane
        // selectors. PERMLANE16 gathers within the current 16-lane row; PERMLANEX16 gathers
        // from the paired row (0<->1, 2<->3). These are untyped operations; every ordinary
        // VOP3 float modifier is reserved. The two overloaded OPSEL bits were retained by
        // the decoder as FI and BOUND_CTRL.
        const uint32_t reserved_opsel = (in.words[0] >> 13) & 3u;
        if ((!b.is_compute && !b.is_fragment) || reserved_opsel || in.clamp || in.omod ||
            in.src_abs[0] || in.src_abs[1] || in.src_abs[2] ||
            in.src_neg[0] || in.src_neg[1] || in.src_neg[2] ||
            in.src[0].kind != OperandKind::VGPR ||
            in.src[1].kind == OperandKind::VGPR || in.src[2].kind == OperandKind::VGPR) {
            ok = false; return true;
        }
        if (in.opcode == 0x378) b.mark_subgroup_min32();
        else b.mark_subgroup_min16();
        const uint32_t result = emit_permlane_b32(
            b, val(in.src[0]), val(in.src[1]), val(in.src[2]), in.opcode == 0x378,
            in.permlane_fetch_inactive, in.permlane_bound_ctrl, [&] { return rs.exec; },
            [&] { return vreg_old(b, rs, in.dst.value); });
        const uint32_t old_d = vreg_old(b, rs, in.dst.value);
        vreg[in.dst.value] = result;
        predicate_write(b, rs, in.dst.value, old_d);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Vop3CrossLaneLowering::lower() {
    if (auto routed = route_writelane()) return *routed;
    if (auto routed = route_readlane()) return *routed;
    if (auto routed = route_permlane16()) return *routed;
    return std::nullopt;
}

std::optional<bool> lower_vop3_cross_lane(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                          bool& ok, bool allow_wave) {
    return Vop3CrossLaneLowering({b, rs, in, ok, allow_wave}).lower();
}

}   // namespace prosper::gpu::ir
