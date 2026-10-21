// Remaining VOP1 routes borrow the original operand callback and register map.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>

namespace prosper::gpu::ir {
struct Vop1CaseRoutes : LoweringContext {
    std::unordered_map<int, uint32_t>& vreg;
    std::function<uint32_t(const Operand&)> val;
    bool lower();
    std::optional<bool> early_routes(uint32_t& a);
    std::optional<bool> sdwa_routes(uint32_t& a, uint32_t& d, uint32_t& old_d);
    std::optional<bool> residual_routes(uint32_t& d, const bool& integer_handled,
                                        const bool& float_handled, const bool& conversion_handled);
};
}  // namespace prosper::gpu::ir
