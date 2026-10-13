// sopp_cfg.hpp -- direct-branch CFG helpers over decoded RDNA2 SOPP instructions, shared by the
// executor's control-flow proofs.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

bool sopp_is_branch(const Rdna2Inst& in);
bool sopp_is_unconditional_branch(const Rdna2Inst& in);
int64_t sopp_branch_target(const Rdna2Inst& in);
bool has_indirect_control_flow(const std::vector<Rdna2Inst>& instructions);

} // namespace prosper::gpu
