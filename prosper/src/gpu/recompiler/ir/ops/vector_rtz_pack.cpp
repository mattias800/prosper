// AMD70648 CVT_PKRTZ: reuse the existing VOP2 graph after preserving VOP3 source-call order.
#include "gpu/recompiler/ir/ops/vector_rtz_pack.hpp"
#include "gpu/recompiler/ir/ops/vector_half.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value pack_rtz(Builder& ir, const TernaryInputs& x) {
    return build_half_rtz(ir, x.s0(), x.s1());
}
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x12f, pack_rtz}};
}   // namespace
std::span<const TernaryIntegerDef> vector_rtz_pack_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_rtz_pack_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
