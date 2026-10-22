// Workgroup-uniform branch proof orchestration and original closure ownership.
#include "gpu/recompiler/rdna2_cfg_uniform_branch.hpp"
#include "gpu/recompiler/rdna2_cfg_uniform_branch_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {

bool branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                 UniformBranchCondition condition) {
    size_t branch_index = ins.size();
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].pc == branch_pc) {
            branch_index = i;
            break;
        }
    if (branch_index == ins.size() || branch_index == 0) return false;
    if (!cfg_uniform_exec_full(condition, ins, branch_index)) return false;

    auto is_branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
               in.opcode != 0x03;
    };
    std::set<uint32_t> targets;
    for (const auto& in : ins)
        if (is_branch(in)) targets.insert(branch_target(in));

    // The straight-line region ending at the branch. Every definition consumed from inside it has a
    // unique reaching definition, which is what lets the backwards slice below be a textual walk.
    size_t block_start = branch_index;
    cfg_uniform_block_start(block_start, targets, ins, std::cref(is_branch));
    auto writes_reg = [](const Rdna2Inst& in, int reg, uint32_t words) {
        return cfg_uniform_writes_reg(in, reg, words);
    };

    auto last_writer = [&](int reg, uint32_t words, size_t before) -> size_t {
        for (size_t i = before; i-- > block_start;)
            if (writes_reg(ins[i], reg, words)) return i;
        return ins.size();
    };

    // A register with no definition inside the region is accepted only when it provably still holds
    // its wave-entry value at the branch. Entry SGPRs are launch data, hence workgroup-uniform.
    std::map<int, bool> entry_value_cache;
    auto still_entry_value = [&](int reg) {
        return cfg_uniform_entry_value(entry_value_cache, ins, std::cref(writes_reg), branch_index,
                                       reg);
    };

    std::function<bool(int, size_t, uint32_t)> uniform_scalar;
    std::function<bool(size_t, uint32_t)> uniform_scc;
    std::function<bool(int, size_t, uint32_t)> uniform_mask;
    auto uniform_operand = [&](const Operand& op, uint32_t words, size_t before,
                               uint32_t depth) -> bool {
        return cfg_uniform_operand(uniform_scalar, op, words, before, depth);
    };

    std::function<bool(int, size_t, uint32_t)> entry_uniform_scalar;
    entry_uniform_scalar = [&](int reg, size_t before, uint32_t depth) -> bool {
        return cfg_uniform_entry_scalar(ins, std::cref(writes_reg), entry_uniform_scalar, reg,
                                        before, depth);
    };
    auto sole_launch_derived_definition = [&](int reg) -> bool {
        return cfg_uniform_sole_launch(ins, std::cref(writes_reg), targets, entry_uniform_scalar,
                                       reg);
    };
    uniform_scalar = [&](int reg, size_t before, uint32_t depth) -> bool {
        return cfg_uniform_scalar(std::cref(last_writer), ins, std::cref(still_entry_value),
                                  std::cref(sole_launch_derived_definition),
                                  std::cref(uniform_operand), reg, before, depth);
    };
    uniform_scc = [&](size_t before, uint32_t depth) -> bool {
        return cfg_uniform_scc(block_start, ins, std::cref(uniform_operand), before, depth);
    };
    uniform_mask = [&](int reg, size_t before, uint32_t depth) -> bool {
        return cfg_uniform_mask(std::cref(last_writer), ins, uniform_scc, uniform_mask, reg, before,
                                depth);
    };

    // Obligation (2): the consumed condition is workgroup-uniform. For a VCC branch that means the
    // architectural mask is a select over workgroup-uniform SCCs; for a scalar branch it is the SCC
    // itself, which `uniform_scc` already proves for the VCC case's inner terms.
    return condition == UniformBranchCondition::Vcc ? uniform_mask(106, branch_index, 0)
                                                    : uniform_scc(branch_index, 0);
}

bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    return branch_is_workgroup_uniform(ins, branch_pc, UniformBranchCondition::Vcc);
}

bool scc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc) {
    return branch_is_workgroup_uniform(ins, branch_pc, UniformBranchCondition::Scc);
}

}   // namespace prosper::gpu
