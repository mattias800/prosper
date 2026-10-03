#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <set>

namespace prosper::gpu {
int shader_max_vgpr(const std::vector<Rdna2Inst>&);
void loop_written_regs(const std::vector<Rdna2Inst>&, uint32_t lo, uint32_t hi,
                       std::set<int>& vgprs, std::set<int>& sgprs);
} // namespace prosper::gpu
