// AMD 70648 12.12 V_PERM_B32: byte selection, sign replication and constant fills.
// Match the existing ascending byte loop and expression nesting for stable instruction IDs.
#include "gpu/recompiler/ir/ops/vector_permute.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
Value permute(Builder& ir, const TernaryInputs& x) {
    const Value high = x.s0(), low = x.s1(), sel_dword = x.s2();
    Value result = ir.c32(0);
    for (uint32_t byte = 0; byte < 4u; ++byte) {
        const Value selector = ir.bfe_u(sel_dword, ir.c32(byte * 8u), ir.c32(8));
        const Value sign_index =
            ir.iadd(ir.imul(ir.and_(selector, ir.c32(3)), ir.c32(2)), ir.c32(1));
        const Value index = ir.select(ir.ult(selector, ir.c32(8)), selector, sign_index);
        const Value word = ir.select(ir.ult(index, ir.c32(4)), low, high);
        const Value shift = ir.imul(ir.and_(index, ir.c32(3)), ir.c32(8));
        const Value val_byte = ir.bfe_u(word, shift, ir.c32(8));
        const Value sign = ir.select(ir.ugt(val_byte, ir.c32(127)), ir.c32(255), ir.c32(0));
        const Value selected = ir.select(ir.ult(selector, ir.c32(8)), val_byte, sign);
        const Value fill = ir.select(ir.ieq(selector, ir.c32(12)), ir.c32(0), ir.c32(255));
        const Value out_byte = ir.select(ir.ult(selector, ir.c32(12)), selected, fill);
        result = ir.or_(result, ir.shl(out_byte, ir.c32(byte * 8u)));
    }
    return result;
}
constexpr TernaryIntegerDef kDefs[] = {{Rdna2Format::VOP3, 0x344, permute}};
}   // namespace
std::span<const TernaryIntegerDef> vector_permute_defs() {
    return kDefs;
}
const TernaryIntegerDef* find_vector_permute_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
