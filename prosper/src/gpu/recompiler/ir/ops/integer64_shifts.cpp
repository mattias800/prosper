// AMD70648 12.12/13.2: numeric raw-word shifts with the guest count modulo 64.
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <functional>
#include "gpu/recompiler/ir/ops/integer64_shifts.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include <vector>
namespace prosper::gpu::ir {
namespace {
Value left(Builder& ir, const TernaryInputs& x) {
    return ir.u64shl(ir.pack_u64(x.s0(), x.s1()), x.s2());
}
Value logical_right(Builder& ir, const TernaryInputs& x) {
    return ir.u64lshr(ir.pack_u64(x.s0(), x.s1()), x.s2());
}
Value arithmetic_right(Builder& ir, const TernaryInputs& x) {
    return ir.u64ashr(ir.pack_u64(x.s0(), x.s1()), x.s2());
}
constexpr Integer64ShiftDef kDefs[] = {
    {Rdna2Format::VOP3, kVop3OpcodeLshlrevB64, left},
    {Rdna2Format::VOP3, kVop3OpcodeLshrrevB64, logical_right},
    {Rdna2Format::VOP3, 0x301, arithmetic_right},
    {Rdna2Format::SOP2, 0x1f, left},
    {Rdna2Format::SOP2, 0x21, logical_right},
    {Rdna2Format::SOP2, 0x23, arithmetic_right},
};
struct Built {
    Block block;
    std::vector<uint32_t> sources;
    Value result;
};
Built build(const Integer64ShiftDef& def) {
    Built out;
    Builder ir(out.block);
    out.result = def.build(ir, TernaryInputs(ir, out.sources));
    return out;
}
Block count_block() {
    Block block;
    Builder ir(block);
    ir.and_(ir.input(Type::U32), ir.c32(63));
    return block;
}
} // namespace
std::span<const Integer64ShiftDef> integer64_shift_defs() {
    return kDefs;
}
const Integer64ShiftDef* find_integer64_shift(Rdna2Format fmt, uint32_t opcode) {
    for (const auto& def : kDefs)
        if (def.fmt == fmt && def.opcode == opcode) return &def;
    return nullptr;
}
Known64 eval_integer64_shift(const Integer64ShiftDef& def, Known low, Known high, Known count) {
    const Block mask = count_block();
    const std::array<Known, 1> raw_count = {count};
    const auto masked = evaluate(mask, raw_count).back();
    const std::array<Known64, 3> sources = {low, high, masked};
    const Built out = build(def);
    std::vector<Known64> inputs;
    inputs.reserve(out.sources.size());
    for (const uint32_t slot : out.sources) inputs.push_back(sources[slot]);
    return evaluate_wide(out.block, inputs)[out.result.index];
}
uint32_t emit_integer64_shift_count(SpirvCompute& b, const std::function<uint32_t()>& read_count) {
    const Block block = count_block();
    return emit_spirv_resolving_inputs(b, block, [&](uint32_t) { return read_count(); }).back();
}
uint32_t emit_integer64_shift_result(SpirvCompute& b, const Integer64ShiftDef& def,
                                     const TernarySourceReader& read_source) {
    const Built out = build(def);
    return emit_spirv_resolving_inputs(b, out.block, [&](uint32_t slot) {
        return read_source(out.sources[slot]);
    })[out.result.index];
}
std::array<uint32_t, 2> emit_integer64_shift_words(SpirvCompute& b, uint32_t result) {
    Block block;
    Builder ir(block);
    const Value word = ir.input(Type::U64);
    const Value lo = ir.u64lo(word), hi = ir.u64hi(word);
    const std::array<uint32_t, 1> input = {result};
    const auto ids = emit_spirv(b, block, input);
    return {ids[lo.index], ids[hi.index]};
}
} // namespace prosper::gpu::ir
