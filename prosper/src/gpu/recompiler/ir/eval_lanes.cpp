// eval_lanes.cpp — per-subgroup evaluation of a Block (eval_lanes.hpp).
#include "gpu/recompiler/ir/eval_lanes.hpp"

namespace prosper::gpu::ir {

namespace {

// One instruction of one invocation, once every invocation's earlier values are known.
Known evaluate_lane(const Inst& inst, const std::vector<std::vector<Known>>& values, size_t lane,
                    std::span<const InvocationInputs> inputs) {
    const auto arg = [&](size_t slot) -> Known { return values[lane][inst.args[slot].index]; };
    switch (inst.op) {
        case Opcode::Input:
            if (inst.type == Type::U64 || inst.imm >= inputs[lane].size()) return std::nullopt;
            return inputs[lane][inst.imm];
        case Opcode::LaneId: return static_cast<uint32_t>(lane);
        case Opcode::Shuffle: {
            const Known source = arg(1);
            if (!source || *source >= values.size()) return std::nullopt;
            return values[*source][inst.args[0].index];
        }
        default: break;
    }
    if (inst.type == Type::U64) return std::nullopt;
    uint32_t words[3] = {0, 0, 0};
    for (uint8_t slot = 0; slot < opcode_info(inst.op).arity; ++slot) {
        const Known a = arg(slot);
        if (!a) return std::nullopt;   // includes every U64 argument, which is never known here
        words[slot] = *a;
    }
    return evaluate_op(inst.op, inst.imm, words[0], words[1], words[2]);
}

} // namespace

std::vector<std::vector<Known>> evaluate_lanes(const Block& block,
                                               std::span<const InvocationInputs> inputs) {
    std::vector<std::vector<Known>> values(inputs.size(), std::vector<Known>(block.size()));
    // Instruction-major, so a Shuffle reads its source invocation's value after every invocation
    // has computed it.
    for (size_t i = 0; i < block.size(); ++i)
        for (size_t lane = 0; lane < inputs.size(); ++lane)
            values[lane][i] = evaluate_lane(block.insts()[i], values, lane, inputs);
    return values;
}

}   // namespace prosper::gpu::ir
