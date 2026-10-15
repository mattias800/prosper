// ir.hpp — the recompiler's SSA intermediate representation (ADR 0012).
//
// A Block is a straight-line list of instructions; each instruction defines exactly one Value, and
// its arguments name Values defined earlier in the same Block. Values are typed (32-bit word or
// boolean). The IR is private to gpu/recompiler.
//
// Today a Block is built for ONE guest instruction and then either EVALUATED on known inputs
// (ir/eval.hpp, used by constant folding) or EMITTED as SPIR-V (ir/emit/spirv_emitter.hpp), in
// instruction order. Emission order is the order the builder appended instructions, which is what
// keeps the recompiler's SPIR-V byte-identical to the pre-IR emitter: a lowering appends in the
// order the old code called SpirvCompute.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::gpu::ir {

enum class Type : uint8_t { U32, Bool };

// A reference to the instruction that defines it, by position in its Block.
struct Value {
    static constexpr uint32_t kNone = UINT32_MAX;
    uint32_t index = kNone;
    bool valid() const { return index != kNone; }
    friend bool operator==(Value, Value) = default;
};

// Every opcode's meaning is written down twice: its evaluation in ir/eval.cpp and the SPIR-V
// ir/emit/spirv_emitter.cpp writes for it. test_ir_scalar_alu_execution runs the emitted SPIR-V
// against the evaluator over many inputs, and requires every opcode to be exercised, so the two
// cannot drift apart silently.
enum class Opcode : uint8_t {
    Input,       // a value supplied from outside the Block; imm = input slot
    ConstU32,    // imm
    ConstTrue,
    ConstFalse,
    IAdd,
    ISub,
    IMul,
    UMulHi,
    SMulHi,   // high 32 bits of the 64-bit unsigned / signed product
    And,
    Or,
    Xor,
    Not,
    BitReverse,
    Shl,
    LShr,
    AShr,  // a shift count of 32 or more is undefined (SPIR-V): it evaluates to unknown
    SMin,
    SMax,
    UMin,
    UMax,
    BfeU,   // (base, offset, count) with SpirvCompute::bfe_u's clamps: min(offset, 32), then
    // min(count, 32 - offset); zero when the clamped count is zero
    Select,   // (bool cond, u32 a, u32 b)
    BoolSelect,   // (bool cond, bool a, bool b)
    IEq,
    INe,
    ULt,
    ULe,
    UGt,
    UGe,
    SLt,
    SLe,
    SGt,
    SGe,   // u32 x u32 -> bool
    F32Add,
    F32Sub,
    F32Mul,
    F32NMin,
    F32NMax,   // raw binary32 bits -> raw bits
};

struct OpcodeInfo {
    const char* name;
    uint8_t arity;
    Type result;
};
const OpcodeInfo& opcode_info(Opcode op);

struct Inst {
    Opcode op = Opcode::ConstU32;
    Type type = Type::U32;
    uint32_t imm = 0;
    std::array<Value, 3> args{};
};

class Block {
public:
    // Appends `inst` and returns the Value it defines. Arguments must already be defined here and
    // of the type the opcode expects; a violation is a lowering bug and aborts with a message.
    Value append(const Inst& inst);
    const Inst& at(Value v) const { return insts_.at(v.index); }
    const std::vector<Inst>& insts() const { return insts_; }
    size_t size() const { return insts_.size(); }
    uint32_t input_count() const { return input_count_; }

private:
    std::vector<Inst> insts_;
    uint32_t input_count_ = 0;
};

} // namespace prosper::gpu::ir
