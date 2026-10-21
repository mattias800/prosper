// CFG branch helpers; private to the recompiler.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <unordered_set>
#include <vector>
namespace prosper::gpu {
bool sopp_is_noop(const Rdna2Inst& in);
std::unordered_set<uint32_t> waterfall_branches(const std::vector<Rdna2Inst>& ins);
uint32_t branch_target(const Rdna2Inst& in);
uint32_t scc_do_while_condition_pc(const std::vector<Rdna2Inst>& ins, uint32_t header_pc,
                                   uint32_t backedge_pc);
} // namespace prosper::gpu
