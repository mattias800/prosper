#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <set>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
bool compute_dpp_quad_or(const Rdna2Inst& instruction);
// CONFIDENCE: HIGH. Per-site, original-entry proof. Acyclic predecessors give each EXEC writer a fresh
// generation; at most eight generations are represented. No Function variable seeds guest DATA.
std::set<uint32_t> proven_compute_quad_or_sources(SpirvCompute& builder, const RegState& entry,
                                                  const std::vector<Rdna2Inst>& instructions,
                                                  const uint32_t* code, size_t dwords,
                                                  uint32_t initial_active);
}   // namespace prosper::gpu
