// ir.cpp — opcode facts and Block construction for the recompiler IR.
#include <iterator>
#include "gpu/recompiler/ir/ir.hpp"

#include <cstdio>
#include <cstdlib>

namespace prosper::gpu::ir {

namespace {

constexpr OpcodeInfo kInfo[] = {
    {"input", 0, Type::U32},   // the real type is the Inst's own
    {"const.u32", 0, Type::U32},  {"const.true", 0, Type::Bool},  {"const.false", 0, Type::Bool},
    {"iadd", 2, Type::U32},       {"isub", 2, Type::U32},         {"imul", 2, Type::U32},
    {"umulhi", 2, Type::U32},     {"smulhi", 2, Type::U32},       {"and", 2, Type::U32},
    {"or", 2, Type::U32},         {"xor", 2, Type::U32},          {"not", 1, Type::U32},
    {"bitreverse", 1, Type::U32}, {"shl", 2, Type::U32},          {"lshr", 2, Type::U32},
    {"ashr", 2, Type::U32},       {"smin", 2, Type::U32},         {"smax", 2, Type::U32},
    {"umin", 2, Type::U32},       {"umax", 2, Type::U32},         {"bfe.u", 3, Type::U32},
    {"select", 3, Type::U32},     {"select.bool", 3, Type::Bool}, {"ieq", 2, Type::Bool},
    {"ine", 2, Type::Bool},       {"ult", 2, Type::Bool},         {"ule", 2, Type::Bool},
    {"ugt", 2, Type::Bool},       {"uge", 2, Type::Bool},         {"slt", 2, Type::Bool},
    {"sle", 2, Type::Bool},       {"sgt", 2, Type::Bool},         {"sge", 2, Type::Bool},
    {"f32.add", 2, Type::U32},    {"f32.sub", 2, Type::U32},      {"f32.mul", 2, Type::U32},
    {"f32.nmin", 2, Type::U32},   {"f32.nmax", 2, Type::U32},
};
static_assert(std::size(kInfo) == static_cast<size_t>(Opcode::F32NMax) + 1);

// The type argument `slot` of `op` must have.
Type argument_type(Opcode op, size_t slot) {
    if ((op == Opcode::Select || op == Opcode::BoolSelect) && slot == 0) return Type::Bool;
    if (op == Opcode::BoolSelect) return Type::Bool;
    return Type::U32;
}

// A malformed Block is a lowering bug, never a property of the guest program, so it stops the
// process loudly instead of producing a module that silently computes something else. No exception:
// recompilation runs inside guest calls (ADR 0022).
[[noreturn]] void malformed(const char* what, const OpcodeInfo& info) {
    std::fprintf(stderr, "[ir] malformed block: %s (%s)\n", what, info.name);
    std::abort();
}

} // namespace

const OpcodeInfo& opcode_info(Opcode op) {
    return kInfo[static_cast<size_t>(op)];
}

Value Block::append(const Inst& inst) {
    const OpcodeInfo& info = opcode_info(inst.op);
    if (inst.op != Opcode::Input && inst.type != info.result)
        malformed("wrong result type", info);
    for (size_t slot = 0; slot < inst.args.size(); ++slot) {
        const Value arg = inst.args[slot];
        if (slot >= info.arity) {
            if (arg.valid()) malformed("extra argument", info);
            continue;
        }
        if (!arg.valid() || arg.index >= insts_.size())
            malformed("undefined argument", info);
        if (insts_[arg.index].type != argument_type(inst.op, slot))
            malformed("argument type mismatch", info);
    }
    if (inst.op == Opcode::Input) {
        if (inst.imm != input_count_) malformed("inputs must be numbered in order", info);
        ++input_count_;
    }
    insts_.push_back(inst);
    return Value{static_cast<uint32_t>(insts_.size() - 1)};
}

} // namespace prosper::gpu::ir
