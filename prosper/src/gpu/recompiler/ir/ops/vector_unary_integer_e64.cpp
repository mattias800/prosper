// AMD70648 12.8/12.12: the e64 aliases reuse the exact existing VOP1 word operations.
#include "gpu/recompiler/ir/ops/vector_unary_integer_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_alu.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <span>
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_integer_unary.hpp"
namespace prosper::gpu::ir {
namespace {
constexpr VectorOpDef kDefs[] = {
    {Rdna2Format::VOP3, 0x1b7, build_integer_unary_not_b32},
    {Rdna2Format::VOP3, 0x1b8, build_integer_unary_bfrev_b32},
    {Rdna2Format::VOP3, 0x1b9, build_integer_unary_ffbh_u32},
    {Rdna2Format::VOP3, 0x1ba, build_integer_unary_ffbl_b32},
};
}
std::span<const VectorOpDef> vector_unary_integer_e64_defs() {
    return kDefs;
}
const VectorOpDef* find_vector_unary_integer_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
} // namespace prosper::gpu::ir
