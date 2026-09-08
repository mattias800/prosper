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

// Emit a compute shader (local_size_x = 256, SPIR-V 1.3) that compares arrays of
// four uint32_t words at storage-buffer bindings 0 and 1. A differing vector
// atomically sets binding 2 word 0 to one and updates that vector in binding 1.
// The number of vectors to compare is push-constant word 0. The caller clears the
// flag before dispatch. This is exact: the atomic only reduces collision-free
// component equality, while binding 1 becomes the next exact baseline.
std::vector<uint32_t> build_compute_compare_uvec4();

// Fused 2D tile27 RGBA16F detiling and exact half_to_unorm8 quantization.
// Binding 0: width, face height, padded face stride in uints, total texel count,
// then 16 packed equation words and independent padded faces. Binding 1: one
// packed RGBA8 uint per output texel. Local size 128; no optional numeric features.
// The caller must validate nonzero dimensions, count = width*height*face_count,
// stride covering each complete padded 128x64/64KiB block footprint, and all
// source/output word indices fitting both their buffer ranges and uint32_t.
// Input and output must not overlap; input remains immutable until execution
// completes. Only excess invocations are checked by the shader itself.
std::vector<uint32_t> build_compute_detile_rgba16f();

// Reconstruct packed R10G10B10A2 UNORM from canonical RGBA8 bytes in place.
// Binding 0: storage buffer of uint32 texels; push constant: texel count.
std::vector<uint32_t> build_compute_rgba8_to_packed10();

// Exact row-major words (binding 0) -> 2D 64 KiB or standard 3D guest tiles (binding 1).
// Push words: width, height, log2(words/texel), log2(block width), log2(block height),
// blocks/row, then 16 packed x/y (or x/y/z) equation masks. Volume words 22..25:
// depth, log2(block depth), block rows/plane, log2(words/block). Dispatch the full
// padded rectangle/volume; out-of-image texels write zero without reading the source.
std::vector<uint32_t> build_compute_retile_words(bool volume = false);

} // namespace prosper::gpu
