// Original scalar CFG case groups; native path exits are retained.
#include <cstdint>
#include "gpu/recompiler/rdna2_cfg_scalar_merge_routes.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <optional>

namespace prosper::gpu::detail {
bool CfgScalarMergeRoutes::alu_reads() {
    // Conditional moves, SOPK immediate operations, and SOP1 bitsets may read their
    // encoded destination before replacing it. Their decoded source list does not
    // contain that old value, so consult the shared implicit-read inventory first.
    const uint32_t implicit_read_width = scalar_implicit_destination_read_width(in);
    if (implicit_read_width && R >= in.dst.value &&
        R < in.dst.value + static_cast<int>(implicit_read_width) &&
        !(data_read_ok && implicit_read_width == 1))
        return block(in, "implicit-dst-rmw");
    for (int k = 0; k < in.n_src; k++) {
        if (in.src[k].kind != OperandKind::SGPR && in.src[k].kind != OperandKind::Special) continue;
        // Operand does not carry B32/B64 width. Use the shared opcode inventory so a
        // physical low-half read (notably GTA's CSELECT_B32 -> V_ADD3 chain) cannot
        // falsely keep the sibling live, while unlisted VOP3 sources remain pairs.
        const uint32_t words = scalar_alu_source_words(in, k);
        if (words == UINT32_MAX) continue;
        if (R >= in.src[k].value && R < in.src[k].value + static_cast<int>(words)) {
            // A one-dword source is a scalar-data read. Anything wider reads the pair,
            // which is exactly the set of lane-mask consumers.
            if (data_read_ok && words == 1) continue;
            return block(in, words == 1 ? "source-dword" : "source-pair");
        }
    }
    return true;
}

std::optional<bool> CfgScalarMergeRoutes::alu() {
    do {
        switch (in.fmt) {
            case Rdna2Format::SOP1:
            case Rdna2Format::SOP2:
            case Rdna2Format::SOPC:
            case Rdna2Format::VOP1:
            case Rdna2Format::VOP2:
            case Rdna2Format::VOP3:
            case Rdna2Format::VOPC:
            case Rdna2Format::
                DS:   // DS operands are VGPR/M0 only; it cannot read an ordinary SGPR/VCC
            case Rdna2Format::EXP: {   // EXP data sources are all VGPRs — it can never read an SGPR
                if (!alu_reads()) return false;
                // Check all sources before the kill: carry-in can read the old destination.
                const bool carry_low_only =
                    !mask_widths.carry_pair &&
                    (vop3b_fresh_carry_output(in) ||
                     (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 && in.opcode <= 0x12a));
                if (in.sdst.kind == OperandKind::SGPR &&
                    (in.sdst.value == R || (!carry_low_only && in.sdst.value + 1 == R)))
                    continue;
                // A non-X VOPC e32 compare writes the whole VCC pair — kills both halves. V_CMPX
                // writes EXEC ONLY on RDNA2 (ISA 12.9: "EXEC[threadId] = ..."; no VCC/SDST dest),
                // so it must NOT count as a VCC kill — miscounting it let safe_execz linearize a
                // block whose vcc scratch-write hardware would have skipped.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.value == 106 || in.dst.value == 107)) {
                    const bool pair = in.dst.kind == OperandKind::SGPR
                                          ? mask_widths.explicit_vcc_pair
                                          : mask_widths.implicit_vcc_pair;
                    if (pair ? (R == 106 || R == 107) : R == in.dst.value) continue;
                }
                // CONFIDENCE: HIGH. RDNA2 ISA 3.9/6.8: a non-CMPX explicit compare
                // replaces its complete low mask word, writing zero for EXEC-off lanes.
                // SDWAB has no partial destination selection. Check sources first: a
                // compare may read its own old scalar destination. Keep HIGH conservative.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
                    R == in.dst.value)
                    continue;
                if (vopc_sgpr_pair_kills && in.fmt == Rdna2Format::VOPC &&
                    !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::SGPR &&
                    in.dst.value >= 0 && in.dst.value <= 104 &&
                    (R == in.dst.value || R == in.dst.value + 1))
                    continue;
                // A scalar redefinition kills every word covered by its opcode's write width. Keep
                // this after the explicit/implicit read checks above: B64 read-modify-write forms
                // must observe the old pair before replacing it. EXEC/M0 (124/126/127) still cannot
                // be proven dead (implicit reads everywhere), and cmpx writes EXEC rather than its
                // decoded SGPR destination.
                const uint32_t dst_width = scalar_write_width(in);
                if (dst_width && in.dst.kind == OperandKind::SGPR && R <= 107 &&
                    R >= in.dst.value && R < in.dst.value + static_cast<int>(dst_width) &&
                    !(in.fmt == Rdna2Format::VOPC && vopc_is_cmpx(in.opcode)))
                    continue;
                break;
            }
            default: break;
        }
        return std::nullopt;
    } while (false);
    return true;
}

}   // namespace prosper::gpu::detail
