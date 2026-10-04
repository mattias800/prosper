#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
// RDNA2 12.13.1 quad mode only. This instruction moves register words, not LDS/GDS bytes.
// Group32, rotate and FFT remain independent gaps; this is not a general DS admission predicate.
const char* packet_quad_swizzle_gap(const Rdna2Inst&);
uint32_t packet_quad_swizzle_source_lane(SpirvCompute&, const Rdna2Inst&);
// Completion is separate from the uniform implementation's internal rendezvous. All reachable
// pending results must survive until a genuine LGKMCNT0, not VMCNT0/EXPCNT0 or a maximum.
const char* packet_quad_swizzle_completion_gap(const std::vector<Rdna2Inst>&, uint32_t& pc);
} // namespace prosper::gpu
