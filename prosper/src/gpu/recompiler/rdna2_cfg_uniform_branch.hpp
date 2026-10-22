// Workgroup-uniform branch proofs; private to the recompiler.
#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;

// Which condition register the branch consumes. The two share every obligation and every step of
// the backwards slice; only the final question differs, so they share one body rather than two
// copies that can drift -- this file has already paid for a duplicated opcode table once.
enum class UniformBranchCondition { Vcc, Scc };

bool branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc,
                                 UniformBranchCondition condition);
bool vcc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
bool scc_branch_is_workgroup_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
}  // namespace prosper::gpu
