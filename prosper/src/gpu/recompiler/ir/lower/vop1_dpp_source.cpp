// Relocated verbatim with extract_case_routes.py; compute source values use IR; reserved paths stay here.
#include "gpu/recompiler/ir/lower/vop1_dpp_source_lowering.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_row_xor.hpp"
#include "gpu/recompiler/ir/lower/vector_dpp_source_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
namespace prosper::gpu::ir {
namespace {
struct SourceContext {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;
    uint32_t& a;
    uint32_t& dpp_active;
};
struct Vop1DppSourceLowering : SourceContext {
    std::optional<bool> stage_dpp_source();
};
std::optional<bool> Vop1DppSourceLowering::stage_dpp_source() {
    // DPP16 quad_perm on src0 (#273): fragment shaders reconstruct the selected quad lane
    // from derivatives. Compute shaders use the exact subgroup quad-swap operation for the
    // three XOR permutations emitted by Astro Bot's blur kernels: horizontal (0xb1), vertical
    // (0x4e), and diagonal (0x1b). Quad boundaries are architectural and remain exact even
    // when the host subgroup width differs from the guest wave width.
    if (in.has_dpp) {
        const bool row_shr = in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11Fu;
        uint32_t row_xor = 0;
        const bool row_ror8 = dpp_row_xor_ctrl(in.dpp_ctrl, &row_xor);
        if (row_ror8) {
            // Direct shuffle is valid only when one native subgroup is exactly one guest
            // wave. Portable/default-subgroup compute is routed through synchronized CFG
            // scratch below. FI=0 makes an EXEC-inactive rotated source read as zero; a
            // ROW_ROR source is always in-range, so BOUND_CTRL does not decide this case.
            if (!b.is_compute || dpp_row_ror8_op(in) != DppRowRor8Op::MovB32 ||
                !b.native_subgroup_size) {
                ok = false;
                return true;
            }
            b.mark_subgroup_min16();
            a = emit_dpp_row_xor_source(b, a, rs.exec, row_xor);
        } else if (row_shr) {
            // The portable NGG vertex shell represents the one live guest lane as lane 0.
            // Every non-zero row-right shift therefore addresses a lane before the start of
            // its row; BOUND_CTRL=1 supplies the architectural zero.  An unbounded access
            // retains the prior destination value and cannot be represented here, so reject.
            if (b.is_fragment || (!b.is_compute && !in.dpp_bound_ctrl) ||
                (b.is_compute && in.opcode != 0x01)) {
                ok = false;
                return true;
            }
            if (b.is_compute) {
                b.mark_subgroup_min16();
                const auto source =
                    emit_dpp_row_shr_source(b, a, rs.exec, in.dpp_ctrl - 0x110u, in.dpp_bound_ctrl);
                dpp_active = source.in_bounds;
                a = source.word;
            } else {
                a = b.uconst(0);
            }
        } else {
            if (b.is_fragment) {
                if (in.opcode != 0x01) {
                    ok = false;
                    return true;
                }
                a = b.dpp_quad(a, in.dpp_ctrl);
            } else if (b.is_compute) {
                // DPP transforms SRC0 before the VOP1 operation. v_mov and the numeric
                // v_cvt_u32_f32 used by UE light-grid kernels are lane-pure afterward.
                if (in.opcode != 0x01 && in.opcode != 0x07) {
                    ok = false;
                    return true;
                }
                a = b.subgroup_quad_permute(a, in.dpp_ctrl, rs.exec);
            } else {
                ok = false;
                return true;
            }
        }
    }
    return std::nullopt;
}

}   // namespace
std::optional<bool> lower_vop1_dpp_source(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                          bool& ok, uint32_t& a, uint32_t& dpp_active) {
    Vop1DppSourceLowering stage{{b, rs, in, ok, a, dpp_active}};
    return stage.stage_dpp_source();
}
}   // namespace prosper::gpu::ir
