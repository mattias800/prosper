#include "gpu/recompiler/rdna2_dpp_row_shr.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"

namespace prosper::gpu {

bool is_inplace_vmax_u32_dpp_row_shr(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == 0x14 && in.has_dpp && !in.has_modifier &&
           !in.has_sdwa && !in.dpp_bound_ctrl && in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
           in.dpp_row_mask == 0xfu && in.dpp_bank_mask == 0xfu && in.n_src == 2 && !in.src_neg[0] &&
           !in.src_neg[1] && !in.src_abs[0] && !in.src_abs[1] && !in.clamp && !in.omod &&
           in.dst.kind == OperandKind::VGPR && in.src[0].kind == OperandKind::VGPR &&
           in.src[1].kind == OperandKind::VGPR && in.dst.value == in.src[0].value &&
           in.dst.value == in.src[1].value;
}

bool emit_compute_inplace_dpp_row_shr(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                      uint32_t source, uint32_t old_destination, bool wave_uniform,
                                      uint32_t& result) {
    if (!b.is_compute) return false;
    // Retain the established ADD ladder and its generated graph. Other ADD strides belong to
    // the existing event-isolated CFG service rather than the ordinary instruction emitter.
    const bool add =
        is_inplace_vadd_nc_u32_dpp_row_shr(in) && (in.dpp_ctrl == 0x111u || in.dpp_ctrl == 0x112u ||
                                                   in.dpp_ctrl == 0x114u || in.dpp_ctrl == 0x118u);
    const bool maximum = wave_uniform && is_inplace_vmax_u32_dpp_row_shr(in);
    if (!add && !maximum) return false;

    // DPP16 ROW_SHR selects lane-N inside its 16-lane row. BC0 keeps VDST at a row edge.
    // FI0 supplies zero for an inactive source. For this in-place unsigned maximum,
    // max(own, 0) == own, so preserving VDST there is exact as well. Destination EXEC is a
    // separate write condition applied by the caller, including when its peer is active.
    uint32_t valid_source = 0;
    const uint32_t shifted =
        b.subgroup_row_shr(source, rs.exec, in.dpp_ctrl - 0x110u, &valid_source);
    const uint32_t combined =
        add ? b.ibin(Op_IAdd, source, shifted) : b.uext2(Glsl_UMax, source, shifted);
    result = b.sel(valid_source, combined, old_destination);
    return true;
}

}  // namespace prosper::gpu
