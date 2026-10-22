// Private bodies delegated to by the original uniformity-proof closures.
#pragma once
#include "gpu/recompiler/rdna2_cfg_uniform_branch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {
bool cfg_uniform_writes_reg(const Rdna2Inst& in, int& reg, uint32_t& words);
bool cfg_uniform_entry_value(std::map<int, bool>& entry_value_cache,
                             const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
                             size_t& branch_index, int& reg);
bool cfg_uniform_operand(std::function<bool(int, size_t, uint32_t)>& uniform_scalar,
                         const Operand& op, uint32_t& words, size_t& before, uint32_t& depth);
bool cfg_uniform_entry_scalar(
    const std::vector<Rdna2Inst>& ins,
    const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, int& reg, size_t& before,
    uint32_t& depth);
bool cfg_uniform_entry_scalar_operand(
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, size_t& w, uint32_t& depth,
    const Operand& op, uint32_t& words);
bool cfg_uniform_entry_scalar_address(
    const uint32_t& words, std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar,
    size_t& w, uint32_t& depth, const Operand& op);
bool cfg_uniform_sole_launch(const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&, int, uint32_t)>& writes_reg,
                             std::set<uint32_t>& targets,
                             std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar,
                             int& reg);
bool cfg_uniform_sole_launch_operand(
    std::function<bool(int, size_t, uint32_t)>& entry_uniform_scalar, size_t& only,
    const Operand& op, uint32_t& count);
bool cfg_uniform_scalar(
    const std::function<size_t(int, uint32_t, size_t)>& last_writer,
    const std::vector<Rdna2Inst>& ins, const std::function<bool(int)>& still_entry_value,
    const std::function<bool(int)>& sole_launch_derived_definition,
    const std::function<bool(const Operand&, uint32_t, size_t, uint32_t)>& uniform_operand,
    int& reg, size_t& before, uint32_t& depth);
bool cfg_uniform_scc(
    size_t& block_start, const std::vector<Rdna2Inst>& ins,
    const std::function<bool(const Operand&, uint32_t, size_t, uint32_t)>& uniform_operand,
    size_t& before, uint32_t& depth);
bool cfg_uniform_mask(const std::function<size_t(int, uint32_t, size_t)>& last_writer,
                      const std::vector<Rdna2Inst>& ins,
                      std::function<bool(size_t, uint32_t)>& uniform_scc,
                      std::function<bool(int, size_t, uint32_t)>& uniform_mask, int& reg,
                      size_t& before, uint32_t& depth);
void cfg_uniform_block_start(size_t& block_start, std::set<uint32_t>& targets,
                             const std::vector<Rdna2Inst>& ins,
                             const std::function<bool(const Rdna2Inst&)>& is_branch);
bool cfg_uniform_exec_full(UniformBranchCondition& condition, const std::vector<Rdna2Inst>& ins,
                           size_t& branch_index);
}  // namespace prosper::gpu
