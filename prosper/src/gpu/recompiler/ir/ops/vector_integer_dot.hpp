// Integer dot definitions use already-materialized 32-bit source words.
#pragma once
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
namespace prosper::gpu {
struct SpirvCompute;
}
namespace prosper::gpu::ir {
struct IntegerDotInputs {
    Value s0, s1, s2;
    uint8_t low = 0, high = 7;
};
struct IntegerDotDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder&, const IntegerDotInputs&);
};
std::span<const IntegerDotDef> vector_integer_dot_defs();
const IntegerDotDef* find_vector_integer_dot_op(Rdna2Format fmt, uint32_t opcode);
Known eval_integer_dot_op(const IntegerDotDef&, const std::array<Known, 3>&, uint8_t low = 0,
                          uint8_t high = 7);
uint32_t emit_integer_dot_op(SpirvCompute&, const IntegerDotDef&, const std::array<uint32_t, 3>&,
                             uint8_t low = 0, uint8_t high = 7);
}   // namespace prosper::gpu::ir
