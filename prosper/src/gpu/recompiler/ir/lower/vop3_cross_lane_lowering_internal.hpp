// vop3_cross_lane_lowering_internal.hpp — the class the VOP3 cross-lane routes are members of. Its
// members carry emit_alu's local names, so the moved text compiles unchanged.
#pragma once

#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/ir/lower/vop3_cross_lane_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <optional>

namespace prosper::gpu::ir {

class Vop3CrossLaneLowering : LoweringContext {
public:
    explicit Vop3CrossLaneLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}

    std::optional<bool> lower();

private:
    std::optional<bool> route_writelane();
    std::optional<bool> route_readlane();
    std::optional<bool> route_permlane16();

    decltype(RegState::vreg)& vreg;
};

}   // namespace prosper::gpu::ir
