// Remaining VOP3 routes borrow the original state and modifier callbacks.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace prosper::gpu::ir {
struct Vop3CaseRoutes : LoweringContext {
    std::unordered_map<int, uint32_t>& vreg;
    uint32_t& old_d;
    std::function<uint32_t(const Operand&)> val;
    std::function<uint32_t(int)> fv;
    std::function<uint32_t(uint32_t)> fresult;
    void lower();
    void cube_and_integer();
    void mbcnt_values();
    void mbcnt_lane_id();
    void mbcnt_rejection();
    void float_tail();
};
}  // namespace prosper::gpu::ir
