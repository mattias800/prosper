// ir.hpp — the recompiler's SSA intermediate representation (ADR 0012).
//
// A Block is a straight-line list of instructions; each instruction defines exactly one Value, and
// its arguments name Values defined earlier in the same Block. Values are typed (32-bit word,
// boolean, or 64-bit word). The IR is private to gpu/recompiler.
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

// U32 is a raw 32-bit word and U64 a raw 64-bit word: neither says integer or float, the opcode
// does. U64 is SPIR-V's 64-bit unsigned integer (SpirvCompute::t_u64, the Int64 capability the
// pre-IR emitter already declared for SGPR-pair and VGPR-pair arithmetic). A guest register pair
// enters as two U32 words and is packed (PackU64); results leave through U64Lo/U64Hi.
enum class Type : uint8_t { U32, Bool, U64 };

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
    F32AddNoContraction,   // separate binary32 rounding; cannot contract with a product
    F32MulNoContraction,
    F32Fma,   // GLSL Fma intent; evaluation is a finite RTE reference, not a Vulkan accuracy promise
    BoolAnd,
    BoolOr,
    BoolNot,
    F32Clamp01,
    PackHalfLo,
    FfbhU32,   // leading zero count; all ones for a zero word
    FindUMsbU32,   // GLSL.std.450 FindUMsb: highest set-bit index, or all ones for zero
    BitCount,   // population count of one numeric word, without wave-mask state
    F32Fract,
    F32Trunc,
    F32Ceil,
    F32RoundEven,
    F32Floor,
    F32Exp2,
    F32Log2,
    F32Rcp,
    F32InverseSqrt,
    F32Sqrt,
    F32Sin,
    F32Cos,
    // Numeric conversions (raw binary32 words); saturating float-to-int matches SpirvCompute.
    I32ToF32,
    U32ToF32,
    F32ToU32Sat,
    F32ToI32Sat,
    BfeS,
    UnpackHalf,   // imm = component 0 or 1, raw binary16 -> binary32
    F32Ge,
    F32Abs,   // absolute value of one raw binary32 word
    F32Lt,   // ordered less-than over raw binary32 words
    F32UnordNe,   // unordered-not-equal over raw binary32 words
    F32Eq,   // ordered equality over binary32 words, returning Bool
    F32Ldexp,   // binary32 word scaled by a signed integer exponent
    PackNorm16,   // raw binary32 to normalized low half; imm0 unsigned, imm1 signed
    NormalizeField,   // integer field / positive f32 divisor; imm31 signed/clamp, imm[30:0] divisor
    // The rest of the binary32 relations over raw words, returning Bool (with F32Lt, F32Eq, F32Ge
    // and F32UnordNe above). Ordered relations are false when either word is a NaN; unordered
    // ones are true.
    F32Le,
    F32Gt,
    F32Lg,   // ordered not-equal
    F32UnordLt,
    F32UnordEq,
    F32UnordLe,
    F32UnordGt,
    F32UnordGe,
    // 64-bit words (HOW_TO_ADD_A_FAMILY.md, "64-bit values"). There are no signed 64-bit relations:
    // signed order is the unsigned order of the word with its sign bit flipped
    // (Builder::pack_u64_signed_order), which is how the pre-IR emitter wrote it.
    PackU64,   // (u32 lo, u32 hi) -> u64 = hi << 32 | lo
    U64Lo,   // u64 -> its low u32 word
    U64Hi,   // u64 -> its high u32 word
    U64Eq,
    U64Ne,
    U64Lt,
    U64Le,
    U64Gt,
    U64Ge,   // u64 x u64 -> bool, unsigned order
    U64Shl,
    U64LShr,
    U64AShr,   // (u64 raw word, u32 count) -> u64; unknown for count >= 64
    U64Bfe,   // (U64 base, U32 offset <= 63, U32 count), clamped at the word end
    // Binary64 relations over raw U64 words, returning Bool (HOW_TO_ADD_A_FAMILY.md, "F64 values").
    // There is no F64 Type: an F64 opcode reads its U64 word as an IEEE binary64 value, the way the
    // F32 opcodes read U32 words, and the emitter bitcasts to OpTypeFloat 64. The same twelve
    // relations as F32: ordered ones are false on a NaN, unordered ones true; F64Lg is ordered
    // not-equal. Emitting one needs float64_admitted() (float64_support.hpp).
    F64Lt,
    F64Eq,
    F64Le,
    F64Gt,
    F64Lg,
    F64Ge,
    F64UnordLt,
    F64UnordEq,
    F64UnordLe,
    F64UnordGt,
    F64UnordNe,
    F64UnordGe,
    // Cross-lane values (HOW_TO_ADD_A_FAMILY.md, "Cross-lane operations"). Their result depends on
    // other invocations of the host subgroup, so `evaluate` (one invocation; the constant folder)
    // never knows one. eval_lanes.hpp evaluates a Block over an explicit vector of invocations.
    LaneId,   // () -> u32: this invocation's index in the host subgroup
    Shuffle,   // (u32 value, u32 lane) -> u32: `value` of invocation `lane`; undefined (unknown)
    // when `lane` is not an invocation of the subgroup
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
