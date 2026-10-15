// spirv_emitter.hpp — emit an IR Block as SPIR-V through SpirvCompute.
//
// Instructions are emitted in Block order, each through the same SpirvCompute call the pre-IR
// emitter used for that operation, so a Block built in the old call order produces the old bytes.
#pragma once

#include "gpu/recompiler/ir/ir.hpp"

#include <span>
#include <vector>

namespace prosper::gpu {
struct SpirvCompute;
}

namespace prosper::gpu::ir {

// Returns the SPIR-V id of every Value of `block`. `inputs[i]` is the id bound to input slot i.
std::vector<uint32_t> emit_spirv(SpirvCompute& b, const Block& block, std::span<const uint32_t> inputs);

} // namespace prosper::gpu::ir
