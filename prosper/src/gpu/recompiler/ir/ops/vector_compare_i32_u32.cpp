// I32/U32 comparison values only: AMD RDNA2 ISA 70648, section 12.9, Table 61.
// The existing lowering prepares sources; cmpx and e64 share these ordinary cmp definitions.
// No definition reads EXEC, writes a mask, or changes source/proof bookkeeping.
#include "gpu/recompiler/ir/ops/vector_compare.hpp"

namespace prosper::gpu::ir {
namespace {
using In = VectorCompareIn;
Value cmp_slt(Builder& ir, const In& x) {
    return ir.slt(x.s0, x.s1);
}
Value cmp_eq(Builder& ir, const In& x) {
    return ir.ieq(x.s0, x.s1);
}
Value cmp_sle(Builder& ir, const In& x) {
    return ir.sle(x.s0, x.s1);
}
Value cmp_sgt(Builder& ir, const In& x) {
    return ir.sgt(x.s0, x.s1);
}
Value cmp_ne(Builder& ir, const In& x) {
    return ir.ine(x.s0, x.s1);
}
Value cmp_sge(Builder& ir, const In& x) {
    return ir.sge(x.s0, x.s1);
}
Value cmp_ult(Builder& ir, const In& x) {
    return ir.ult(x.s0, x.s1);
}
Value cmp_ule(Builder& ir, const In& x) {
    return ir.ule(x.s0, x.s1);
}
Value cmp_ugt(Builder& ir, const In& x) {
    return ir.ugt(x.s0, x.s1);
}
Value cmp_uge(Builder& ir, const In& x) {
    return ir.uge(x.s0, x.s1);
}
constexpr VectorCompareDef kDefs[] = {
    {0x81, cmp_slt}, {0x82, cmp_eq},  {0x83, cmp_sle}, {0x84, cmp_sgt},
    {0x85, cmp_ne},  {0x86, cmp_sge}, {0xc1, cmp_ult}, {0xc2, cmp_eq},
    {0xc3, cmp_ule}, {0xc4, cmp_ugt}, {0xc5, cmp_ne},  {0xc6, cmp_uge},
};
} // namespace
std::span<const VectorCompareDef> vector_compare_i32_u32_defs() {
    return kDefs;
}
} // namespace prosper::gpu::ir
