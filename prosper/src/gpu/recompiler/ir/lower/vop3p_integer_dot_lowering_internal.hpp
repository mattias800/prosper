#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <optional>
#include <cstdint>
namespace prosper::gpu::ir {
class Vop3pIntegerDotLowering : LoweringContext {
public:
    explicit Vop3pIntegerDotLowering(const LoweringContext& context) : LoweringContext(context) {}
    bool lower();

private:
    std::optional<bool> route_dot();
    uint32_t compute();
};
}   // namespace prosper::gpu::ir
