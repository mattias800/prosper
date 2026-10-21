// FLAT case routes borrow the original operand reader and proof-bearing state.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <functional>
#include <optional>
namespace prosper::gpu {
struct FlatAccessInfo;
struct ShaderResourceTable;
} // namespace prosper::gpu
namespace prosper::gpu::ir {
struct FlatCaseRoutes : LoweringContext {
    const FlatAccessInfo& access;
    const ShaderResourceTable* rt;
    std::function<uint32_t(const Operand&)> val;
    std::optional<bool> relocated();
    std::optional<bool> packed();
    std::optional<bool> window();
    bool scratch();
    bool lower();
};
} // namespace prosper::gpu::ir
