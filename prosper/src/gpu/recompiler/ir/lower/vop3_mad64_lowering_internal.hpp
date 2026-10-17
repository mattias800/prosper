// Preserve the original scope names without inserting either destination word.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <optional>
namespace prosper::gpu::ir {
class Vop3Mad64Lowering : LoweringContext {
public:
    explicit Vop3Mad64Lowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    std::optional<bool> route_value();

private:
    decltype(RegState::vreg)& vreg;
};
}   // namespace prosper::gpu::ir
