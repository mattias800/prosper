#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
// Code-only finite-trip obligation for loops a newly authenticated fragment
// continuation would expose after linearizing a wave-empty survivor branch.
// This grants no branch classification, cached copy proof or scalar DATA.
bool fragment_dispatch_interior_loops_are_finite(const std::vector<Rdna2Inst>& selected,
                                                 uint32_t early_out_pc, uint32_t gate_pc);
} // namespace prosper::gpu
