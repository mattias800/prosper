// AMD 70648 §§12.7/12.12: e64 adds 0x100 to the VOP2 opcode. Arithmetic is shared;
// modifiers, source authority and predicated destination publication belong to lowering.
#include "gpu/recompiler/ir/ops/vector_binary_integer.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
using F = Rdna2Format;
constexpr VectorOpDef kDefs[] = {
    {F::VOP3, 0x10b, build_vector_mul_u32_u24},   {F::VOP3, 0x111, build_vector_min_i32},
    {F::VOP3, 0x112, build_vector_max_i32},       {F::VOP3, 0x113, build_vector_min_u32},
    {F::VOP3, 0x114, build_vector_max_u32},       {F::VOP3, 0x116, build_vector_lshrrev_b32},
    {F::VOP3, 0x118, build_vector_ashrrev_i32},   {F::VOP3, 0x11a, build_vector_lshlrev_b32},
    {F::VOP3, 0x11b, build_vector_and_b32},       {F::VOP3, 0x11c, build_vector_or_b32},
    {F::VOP3, 0x11d, build_vector_xor_b32},       {F::VOP3, 0x11e, build_vector_xnor_b32},
    {F::VOP3, 0x125, build_vector_add_nc_u32},    {F::VOP3, 0x126, build_vector_sub_nc_u32},
    {F::VOP3, 0x127, build_vector_subrev_nc_u32},
};
} // namespace
std::span<const VectorOpDef> vector_binary_integer_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_binary_integer_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
