// vop2_stages.cpp — VOP2 source and destination modifier stages: SDWA source selects / float abs-neg, SDWA destination, OMOD/CLAMP.
// Relocated verbatim from emit_alu's VOP2 case by tools/refactor/extract_case_routes.py (ADR 0012).
#include "gpu/recompiler/ir/lower/vector_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Vop2Lowering::stage_source_modifiers() {
    if (integer_sdwa) {
        // SEXT sign-extends the selected field instead of zero-extending it (#2013's
        // `v_add_nc_u32_sdwa v4, 8, sext(v5) src1_sel:WORD_0`). The decoder only sets the
        // flag alongside a real sub-dword select, so a DWORD source never reaches bfe_s.
        auto select = [&](uint32_t raw, uint8_t sel, bool sext) {
            if (sel > 5u) return raw;
            const uint32_t offset = sel <= 3u ? 8u * sel : 16u * (sel - 4u);
            const uint32_t width = sel <= 3u ? 8u : 16u;
            return sext ? b.bfe_s(raw, b.uconst(offset), b.uconst(width))
                        : b.bfe_u(raw, b.uconst(offset), b.uconst(width));
        };
        a = select(a, in.sdwa_src0_sel, in.sdwa_src0_sext);
        c = select(c, in.sdwa_src1_sel, in.sdwa_src1_sext);
    } else if (!packed_f16) {
        if (in.src_abs[0]) a = b.fext1(Glsl_FAbs, a);
        if (in.src_neg[0]) a = b.fneg(a);
        if (in.src_abs[1]) c = b.fext1(Glsl_FAbs, c);
        if (in.src_neg[1]) c = b.fneg(c);
    }
    return std::nullopt;
}

std::optional<bool> Vop2Lowering::stage_sdwa_destination(uint32_t& d) {
    // SDWA sub-dword DESTINATION for the integer VOP2 ops (#2013). The operation itself ran
    // on the (already sub-dword-selected) sources above and produced a full dword; DST_SEL
    // says which byte/word of the register receives its low bits, and DST_UNUSED says what
    // happens to the rest — UNUSED_PAD (0) zero-fills it, UNUSED_PRESERVE (2) keeps the old
    // register bits. (Sonic Racing: CrossWorlds' `v_and_b32_sdwa v2, 6, v0 dst_sel:WORD_0
    // dst_unused:UNUSED_PRESERVE`; the decoder admits only PAD/PRESERVE, so UNUSED_SEXT
    // still rejects.) Without this the whole result landed in bits [31:0], silently
    // clobbering the half the guest asked to preserve.
    if (ok && integer_sdwa && in.sdwa_dst_sel != 6) {
        const uint32_t width  = in.sdwa_dst_sel <= 3 ? 8u : 16u;
        const uint32_t offset = in.sdwa_dst_sel <= 3 ? 8u * in.sdwa_dst_sel
                                                     : 16u * (in.sdwa_dst_sel - 4u);
        const uint32_t field_mask = ((1u << width) - 1u) << offset;
        const uint32_t placed = b.ibin(
            Op_ShiftLeftLogical,
            b.ibin(Op_BitwiseAnd, d, b.uconst((1u << width) - 1u)), b.uconst(offset));
        d = in.sdwa_dst_unused == 2u
            ? b.ibin(Op_BitwiseOr,
                     b.ibin(Op_BitwiseAnd, old_d, b.uconst(~field_mask)), placed)
            : placed;
    }
    return std::nullopt;
}

std::optional<bool> Vop2Lowering::stage_output_modifiers(uint32_t& d) {
    // SDWA output modifier: OMOD scale (×2/×4/×0.5) then CLAMP saturate, on FLOAT-result opcodes
    // only (int ops never carry omod). Mirrors the VOP3 fresult path; a no-op when omod/clamp unset.
    if (ok && (in.omod || in.clamp) && !packed_f16) switch (in.opcode) {
        case 0x03: case 0x04: case 0x05: case 0x08: case 0x0F: case 0x10:
        case 0x1F: case 0x2B: case 0x20: case 0x2C: case 0x21: case 0x2D:
            if      (in.omod == 1) d = b.fbin(Op_FMul, d, b.uconst(fbits(2.0f)));
            else if (in.omod == 2) d = b.fbin(Op_FMul, d, b.uconst(fbits(4.0f)));
            else if (in.omod == 3) d = b.fbin(Op_FMul, d, b.uconst(fbits(0.5f)));
            if (in.clamp) d = b.clamp01(d);
            break;
        // A non-float-result opcode carrying a modifier (e.g. an INTEGER SDWA op with CLAMP =
        // integer saturation) is not modeled by the float-domain omod/clamp above, so applying
        // nothing would SILENTLY drop the saturation and emit a valid-but-wrong shader. Reject
        // loudly instead — the same fail-visibly-over-miscompile discipline as the forward-if
        // clamp (#129/#174). The guard means this only fires for a modifier-carrying op.
        default: ok = false; break;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
