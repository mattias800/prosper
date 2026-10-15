// scalar_compare.cpp — definitions of the SOPC compares and the SOPK s_cmpk_* forms.
//
// Each writes only SCC (ISA 70648 12.2, 12.4). EQ/LG are sign-agnostic.
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"

namespace prosper::gpu::ir {

namespace {

using F = Rdna2Format;
using O = ScalarOperands;
using In = ScalarOpIn;
using Out = ScalarOpOut;

template <Value (Builder::*Compare)(Value, Value)>
Out compare(Builder& ir, const In& x) {
    return {{}, (ir.*Compare)(x.s0, x.s1)};
}

// s_bitcmp0/1_b32: SCC = bit S1[4:0] of S0, negated for bitcmp0. (llvm-mc gfx1010: 0xbf0d8014 is
// s_bitcmp1_b32 s20, 0; #273.)
template <bool ONE>
Out bitcmp_b32(Builder& ir, const In& x) {
    const Value amount = ir.and_(x.s1, ir.c32(31));
    const Value bit = ir.and_(ir.lshr(x.s0, amount), ir.c32(1));
    const Value set = ir.ine(bit, ir.c32(0));
    return {{}, ONE ? set : ir.bselect(set, ir.cfalse(), ir.ctrue())};
}

constexpr ScalarOpDef kDefs[] = {
    {F::SOPC, 0x00, O::Src0Src1, compare<&Builder::ieq>, Fold::Yes},   // s_cmp_eq_i32
    {F::SOPC, 0x01, O::Src0Src1, compare<&Builder::ine>, Fold::Yes},   // s_cmp_lg_i32
    {F::SOPC, 0x02, O::Src0Src1, compare<&Builder::sgt>, Fold::Yes},
    {F::SOPC, 0x03, O::Src0Src1, compare<&Builder::sge>, Fold::Yes},
    {F::SOPC, 0x04, O::Src0Src1, compare<&Builder::slt>, Fold::Yes},
    {F::SOPC, 0x05, O::Src0Src1, compare<&Builder::sle>, Fold::Yes},
    {F::SOPC, 0x06, O::Src0Src1, compare<&Builder::ieq>, Fold::Yes},   // s_cmp_eq_u32
    {F::SOPC, 0x07, O::Src0Src1, compare<&Builder::ine>, Fold::Yes},   // s_cmp_lg_u32
    {F::SOPC, 0x08, O::Src0Src1, compare<&Builder::ugt>, Fold::Yes},
    {F::SOPC, 0x09, O::Src0Src1, compare<&Builder::uge>, Fold::Yes},
    {F::SOPC, 0x0a, O::Src0Src1, compare<&Builder::ult>, Fold::Yes},
    {F::SOPC, 0x0b, O::Src0Src1, compare<&Builder::ule>, Fold::Yes},
    {F::SOPC, 0x0c, O::Src0Src1, bitcmp_b32<false>},
    {F::SOPC, 0x0d, O::Src0Src1, bitcmp_b32<true>},
    {F::SOPK, 0x03, O::DstSimmSext, compare<&Builder::ieq>},   // s_cmpk_eq_i32
    {F::SOPK, 0x04, O::DstSimmSext, compare<&Builder::ine>},
    {F::SOPK, 0x05, O::DstSimmSext, compare<&Builder::sgt>},
    {F::SOPK, 0x06, O::DstSimmSext, compare<&Builder::sge>},
    {F::SOPK, 0x07, O::DstSimmSext, compare<&Builder::slt>},
    {F::SOPK, 0x08, O::DstSimmSext, compare<&Builder::sle>},
    {F::SOPK, 0x09, O::DstSimmZext, compare<&Builder::ieq>},   // s_cmpk_eq_u32
    {F::SOPK, 0x0a, O::DstSimmZext, compare<&Builder::ine>},
    {F::SOPK, 0x0b, O::DstSimmZext, compare<&Builder::ugt>},
    {F::SOPK, 0x0c, O::DstSimmZext, compare<&Builder::uge>},
    {F::SOPK, 0x0d, O::DstSimmZext, compare<&Builder::ult>},
    {F::SOPK, 0x0e, O::DstSimmZext, compare<&Builder::ule>},
};

} // namespace

std::span<const ScalarOpDef> scalar_compare_defs() { return kDefs; }

} // namespace prosper::gpu::ir
