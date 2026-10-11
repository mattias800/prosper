#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>

namespace prosper::gpu {
struct RegState;

// Ordinary READLANE needs the common portable dispatcher; WRITELANE spill reloads retain their
// existing typed-slot route. This inventory does not itself authorize a dispatcher or phase split.
bool has_ordinary_cfg_readlane(const std::vector<Rdna2Inst>& instructions);

// Only for the otherwise-refused READLANE route. The existing phase proof remains mandatory,
// and opaque-M0/relative-register state is excluded until its cross-region lifetime is modeled.
bool can_retry_readlane_barrier_phases(const std::vector<Rdna2Inst>& instructions,
                                       const RegState& initial);
} // namespace prosper::gpu
