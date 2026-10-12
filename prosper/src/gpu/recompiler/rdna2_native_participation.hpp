#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

namespace prosper::gpu {
// Original complete single-wave entry identity. This is compiler state, never runtime guest data.
std::vector<uint64_t> native_participation_entry(const SpirvCompute&, const RegState*);
// Sites at which every host worker reaches the same dynamic producer and its predicate consumes
// genuine current guest words. Routing equality alone never grants physical DATA authority.
std::set<uint32_t> native_mask_participation(const SpirvCompute&, const RegState*,
                                             const std::vector<Rdna2Inst>&,
                                             std::set<uint32_t>* numeric_vcc_moves = nullptr);
bool native_mask_predicate_available(const RegState&, const Rdna2Inst&);
bool native_mask_participation_site(const SpirvCompute&, const Rdna2Inst&);
} // namespace prosper::gpu
