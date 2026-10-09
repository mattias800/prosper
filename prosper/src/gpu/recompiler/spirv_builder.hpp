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

// Fused 2D tile27 RG16F/RGBA16F detiling with exact integer conversion.
// Binding 0: width, face height, padded face stride in uints, total texel count,
// then 16 packed equation words and independent padded faces. Binding 1: one
// packed RGBA8 uint or four raw float-bit uints per output texel, selected below.
// Local size 128; no optional numeric features.
// The caller must validate nonzero dimensions, count = width*height*face_count,
// stride covering each complete padded 64KiB block (128x128 for RG, 128x64 for RGBA),
// and all source/output word indices fitting both their buffer ranges and uint32_t.
// Input and output must not overlap; input remains immutable until execution
// completes. Only excess invocations are checked by the shader itself.
// Only two or four source components are supported; other values return empty.
enum class Float16DetileOutput { Rgba8, RawUvec4 };
// RawUvec4 emits four float-bit dwords per texel, preserving every binary16 bit
// through the storage-image ABI (including exceptional values). Missing B/A are 0/1.
std::vector<uint32_t> build_compute_detile_float16(
    uint32_t components = 4, Float16DetileOutput output = Float16DetileOutput::Rgba8);

// Quantise D32 float depth to the retained depth-cube bridge's replicated RGBA8 in place.
// Binding 0: storage buffer of uint32 words (float bits in, packed texels out); push constant:
// texel count. Each texel becomes q | q<<8 | q<<16 | 0xff000000 with
// q = trunc(clamp(d, 0, 1) * 255 + 0.5), with both float32 operations rounded to nearest-even
// by integer arithmetic. This matches depth_cube_quantize.hpp for finite depth on devices whose
// native float rounding mode differs from the CPU's. NaN deterministically maps to zero; the CPU
// scalar cast from NaN has no defined result. Local size 128.
std::vector<uint32_t> build_compute_depth_to_rgba8();

// Reconstruct packed R10G10B10A2 UNORM from canonical RGBA8 bytes in place.
// Binding 0: storage buffer of uint32 texels; push constant: texel count.
std::vector<uint32_t> build_compute_rgba8_to_packed10();

// The reverse, exactly as the CPU unpack rounds it (unorm10_snapshot.hpp): channel = (v*255+511)/1023,
// alpha = a*85. Word i of the first half is converted into word i+count, so the packed result in the
// first half stays intact for guest writeback. Binding 0: storage buffer of 2*count uint32 words;
// push constant: count. Integer arithmetic, so the result does not depend on a driver's
// float-to-UNORM rounding (which RADV and lavapipe resolve differently, #4709). Local size 128.
std::vector<uint32_t> build_compute_packed10_to_rgba8_append();

// #3656: bound a device-produced vkCmdDispatchIndirect argument record before the dispatch reads it.
// Binding 0: storage buffer of at least seven uint32 words -- x, y, z group counts, an output flag,
// then the per-axis limits (maxComputeWorkGroupCount[0..2]) in words 4..6. If any count exceeds its
// own axis limit, all three become zero and word 3 is 1; otherwise word 3 is 0 and the counts are
// untouched. The push-constant block the shared in-place header declares is unused. Dispatch
// exactly one group; local size 128, only invocation 0 acts.
std::vector<uint32_t> build_compute_indirect_dispatch_validate();

// Exact row-major words (binding 0) -> 2D 64 KiB or standard 3D guest tiles (binding 1).
// Push words: width, height, log2(words/texel), log2(block width), log2(block height),
// blocks/row, then 16 packed x/y (or x/y/z) equation masks. Volume words 22..25:
// depth, log2(block depth), block rows/plane, log2(words/block). Dispatch the full
// padded rectangle/volume; out-of-image texels write zero without reading the source.
// PackedSubwordArray uses word 2 as log2(texels/word), grouping four bytes or two
// halfwords per horizontal word. Words 22..24 are layer count,
// linear words/layer and tiled words/layer. Each layer is an ordinary 2D tile plane.
enum class RetileShaderKind { Words2D, Volume3D, PackedSubwordArray };
enum class RetileImageSource { LinearBuffer, R32Uint, Rgba8Uint };
// Image variants also publish exact row-major words for existing result baselines.
// compare_result compares/adopts exact linear words at binding 3 and atomically sets
// the scalar flag at binding 4. Padding remains outside this linear comparison.
std::vector<uint32_t> build_compute_retile_words(RetileShaderKind kind = RetileShaderKind::Words2D,
    RetileImageSource source = RetileImageSource::LinearBuffer, bool image_arrayed = false,
    bool compare_result = false);

} // namespace prosper::gpu
