// Numeric 64-bit shifts. Guest count masking is a separate Block so scalar lowering retains its
// existing failed-source boundary, and result unpacking is a separate Block for old_hi ordering.
#pragma once
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include <array>
#include <functional>
#include <span>
namespace prosper::gpu::ir {
struct Integer64ShiftDef {
    Rdna2Format fmt;
    uint32_t opcode;
    Value (*build)(Builder&, const TernaryInputs&); // low word, high word, masked count
};
std::span<const Integer64ShiftDef> integer64_shift_defs();
const Integer64ShiftDef* find_integer64_shift(Rdna2Format fmt, uint32_t opcode);
Known64 eval_integer64_shift(const Integer64ShiftDef& def, Known low, Known high, Known count);
uint32_t emit_integer64_shift_count(SpirvCompute& b, const std::function<uint32_t()>& read_count);
uint32_t emit_integer64_shift_result(SpirvCompute& b, const Integer64ShiftDef& def,
                                     const TernarySourceReader& read_source);
std::array<uint32_t, 2> emit_integer64_shift_words(SpirvCompute& b, uint32_t result);
} // namespace prosper::gpu::ir
