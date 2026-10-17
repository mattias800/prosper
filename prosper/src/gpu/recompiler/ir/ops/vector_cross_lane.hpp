// vector_cross_lane.hpp — the VALU operations whose value depends on another lane's data
// (HOW_TO_ADD_A_FAMILY.md, "Cross-lane operations").
//
// A definition here is the per-invocation VALUE only, written with the cross-lane opcodes LaneId
// and Shuffle. Everything around it stays in the lowering: which scalar or vector register
// receives it, the subgroup-size contract (mark_subgroup_min*), the scratch route for a guest wave
// wider than the host subgroup, lane-slot spills, mask-domain retirement and the wave-uniform and
// lane-local facts.
#pragma once

#include "gpu/recompiler/ir/builder.hpp"

#include <cstdint>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

// v_readlane_b32 (VOP3 0x360): D = S0 of guest lane S1 % W, ignoring EXEC (AMD 70648,
// V_READLANE_B32: "Lane is S1 % (32 if wave32, 64 if wave64). Ignores exec mask."). The guest
// wave is the aligned W-invocation slice of the host subgroup holding this invocation, so the
// source invocation is (LaneId & ~(W - 1)) | (S1 & (W - 1)); a host subgroup wider than the wave
// (a Wave32 guest in a 64-invocation subgroup) reads within its own slice. `wave_size` is 32 or 64.
Value readlane_b32(Builder& ir, Value source, Value selector, uint32_t wave_size);

// The SPIR-V of readlane_b32 for bound ids, in the pre-IR emitter's order.
uint32_t emit_readlane_b32(SpirvCompute& b, uint32_t source, uint32_t selector, uint32_t wave_size);

}   // namespace prosper::gpu::ir
