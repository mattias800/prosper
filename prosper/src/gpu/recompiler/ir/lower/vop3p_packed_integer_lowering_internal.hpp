// Preserve the original case-scope alias without inserting a destination in the constructor.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
struct TernaryIntegerDef;
class Vop3pPackedIntegerLowering : LoweringContext {
public:
    explicit Vop3pPackedIntegerLowering(const LoweringContext& context)
        : LoweringContext(context), vreg(rs.vreg) {}
    bool lower();

private:
    decltype(RegState::vreg)& vreg;
    uint32_t u16src(int, bool);
    uint32_t operation(bool, const TernaryIntegerDef&);
};
}   // namespace prosper::gpu::ir
