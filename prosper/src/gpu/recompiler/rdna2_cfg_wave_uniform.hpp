// Wave-uniform VCC exit proof; distinct from workgroup-uniform branch admission.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
bool cannot_write_vcc(const Rdna2Inst& in);
bool vcc_exit_is_wave_uniform(const std::vector<Rdna2Inst>& ins, uint32_t branch_pc);
}  // namespace prosper::gpu
