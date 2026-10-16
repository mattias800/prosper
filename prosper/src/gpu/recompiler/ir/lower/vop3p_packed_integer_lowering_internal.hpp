// Preserve the original case-scope alias without inserting a destination in the constructor.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <optional>
namespace prosper::gpu::ir {
class Vop3pPackedIntegerLowering : LoweringContext {
public:
    explicit Vop3pPackedIntegerLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    std::optional<bool> route_integer();
};
}   // namespace prosper::gpu::ir
