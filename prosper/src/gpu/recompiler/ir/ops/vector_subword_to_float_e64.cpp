// AMD70648 12.8/12.8.1: unsigned bytes and signed low nibbles convert exactly to F32, independent of FP modes.
#include "gpu/recompiler/ir/ops/vector_subword_to_float_e64.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <span>
namespace prosper::gpu::ir {
namespace {
constexpr ConversionDef kDefs[] = {
    {Rdna2Format::VOP3, 0x18e, build_conversion_off_f32_i4},
    {Rdna2Format::VOP3, 0x191, build_conversion_ubyte0_f32},
    {Rdna2Format::VOP3, 0x192, build_conversion_ubyte1_f32},
    {Rdna2Format::VOP3, 0x193, build_conversion_ubyte2_f32},
    {Rdna2Format::VOP3, 0x194, build_conversion_ubyte3_f32},
};
}   // namespace
std::span<const ConversionDef> vector_subword_to_float_e64_defs() {
    return kDefs;
}
const ConversionDef* find_vector_subword_to_float_e64_op(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
}   // namespace prosper::gpu::ir
