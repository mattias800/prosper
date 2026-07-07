// spirv_builder.hpp — minimal SPIR-V module emitter, the code-generation foundation for the
// RDNA2 -> SPIR-V shader recompiler. The recompiler will translate decoded RDNA2 instructions
// (rdna2_decode.hpp) into SPIR-V by driving a builder like this. As a first, self-verifying step we
// emit a compute shader by hand and prove — via the execution-differential harness — that our
// emitter produces valid, correct, runnable SPIR-V.
#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Emit a compute shader (local_size_x = 64, SPIR-V 1.3) that computes, for storage buffers
// `a` at (set 0, binding 0) and `b` at (set 0, binding 1):  b[gid.x] = a[gid.x] * scale + bias.
// Returns the SPIR-V module as 32-bit words. Deterministic; no I/O.
std::vector<uint32_t> build_compute_scale_bias(float scale, float bias);

} // namespace prosper::gpu
