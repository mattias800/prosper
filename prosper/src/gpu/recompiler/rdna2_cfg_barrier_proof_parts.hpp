// Bodies delegated to by the original terminal-guard proof closures.
#pragma once
#include "gpu/recompiler/rdna2_cfg_barrier_phases.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <functional>
#include <vector>

namespace prosper::gpu {
bool cfg_terminal_uniform_operand(std::function<bool(int, uint32_t, uint32_t)>& uniform_reg_at,
                                  const Rdna2Inst& writer, uint32_t& depth, const Operand& operand,
                                  uint32_t& words);
bool cfg_terminal_uniform_register(const std::vector<Rdna2Inst>& ins,
                                   std::function<bool(int, uint32_t, uint32_t)>& uniform_reg_at,
                                   int& reg, uint32_t& use_pc, uint32_t& depth);
void cfg_barrier_scalar_prefix(bool& scalar_prefix, BarrierPhasedCompute& result,
                               const std::vector<Rdna2Inst>& ins);
}  // namespace prosper::gpu
