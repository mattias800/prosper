// Bounded M0 retirement after an already-closed register-index waterfall packet.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <vector>
namespace prosper::gpu {
// Proves no M0 observation before an independent unconditional overwrite.
// Supplies no source value and grants no control-flow route permission.
bool deferred_waterfall_m0_retirement(const std::vector<Rdna2Inst>& ins, std::size_t first,
                                      int index);
} // namespace prosper::gpu
