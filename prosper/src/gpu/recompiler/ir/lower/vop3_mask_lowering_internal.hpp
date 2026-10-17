// vop3_mask_lowering_internal.hpp — the class the VOP3 mask-reading and mask-writing routes are
// members of. Its members carry emit_alu's local names, so the moved text compiles unchanged.
#pragma once

#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_mask_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <optional>

namespace prosper::gpu::ir {

class Vop3MaskLowering : LoweringContext {
public:
    Vop3MaskLowering(const LoweringContext& context, const Vop3MaskSource* source = nullptr)
        : LoweringContext(context), vreg(rs.vreg), source_(source) {}

    std::optional<bool> route_carry_out();
    std::optional<bool> route_carry_in_out();
    std::optional<bool> route_cndmask();

private:
    decltype(RegState::vreg)& vreg;
    const Vop3MaskSource* source_;
    uint32_t fv(int k) const { return (*source_)(k); }
    void write_vop3b_carry_output(uint32_t carry_mask);
};

}   // namespace prosper::gpu::ir
