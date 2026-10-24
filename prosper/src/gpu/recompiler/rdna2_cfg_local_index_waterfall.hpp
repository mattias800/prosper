#pragma once

#include <cstdint>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;

// A closed, register-only index waterfall already linearized by waterfall_branches.
// Each lane's one body execution uses its own index. The temporary scalar index
// must be dead afterward; gfx10 CMPX preserves VCC throughout these cohorts.
// Returns the exact backward branch PC, or UINT32_MAX without that proof.
uint32_t local_index_waterfall_latch(const std::vector<Rdna2Inst>& ins, uint32_t read_pc);
// Locate the owning READFIRSTLANE for that read, its M0 copy, or a body MOVRELS.
// The complete packet proof is required at each capture point.
uint32_t local_index_waterfall_read_pc(const std::vector<Rdna2Inst>& ins, const Rdna2Inst& in);
} // namespace prosper::gpu
