// rdna2_to_spirv.hpp — first real RDNA2 -> SPIR-V translation (recompiler stage 3).
//
// Scope for this stage: a straight-line float VALU kernel. We model VGPRs as SSA float values,
// translate a decoded RDNA2 instruction stream (rdna2_decode.hpp) op-by-op into SPIR-V float ops,
// and wrap it in a compute shader so it is verifiable by execution (test_rdna2_to_spirv runs it and
// asserts the numbers against the RDNA2 semantics — verification layer 4). Control flow, the EXEC
// mask, memory instructions, and integer/scalar ALU come in later stages; unsupported opcodes make
// recompile_valu return an empty vector so the caller can tell it wasn't handled.
//
// I/O convention (so the kernel is testable): the compute shader loads `num_inputs` consecutive
// floats per invocation from storage buffer 0 into v0..v(num_inputs-1)  (v[k] = a[gid*num_inputs+k]),
// runs the translated ALU ops, then stores v[out_vgpr] to storage buffer 1 (b[gid]).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace prosper::gpu {

struct ShaderResourceTable;   // resource-binding contract (shader_resources.hpp); optional to recompile_valu

// Translate a straight-line float-VALU RDNA2 stream to a compute-shader SPIR-V module.
// Returns {} if the stream contains an opcode/format this stage does not yet handle. An optional
// ShaderResourceTable routes SMEM constant-buffer loads to distinct bindings via descriptor provenance.
// `lds_bytes` is the shader's real per-workgroup LDS allocation (COMPUTE_PGM_RSRC2.LDS_SIZE): the
// emitted Workgroup array is sized to it, clamped to the RDNA2 64 KB max (#130). 0 = the 16 KB
// default (also the safe cap for the common maxComputeSharedMemorySize of 32 KB).
std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr,
                                     const ShaderResourceTable* rt = nullptr, uint32_t lds_bytes = 0);

// Recompile a pixel/fragment shader to a fragment SPIR-V module: run the VALU, and on EXP to an MRT
// target write vec4(src0..3) to the location-0 color output. Returns {} if unsupported / no export.
// An optional ShaderResourceTable enables memory ops (SMEM/MUBUF) with resolved bindings.
std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords,
                                         const ShaderResourceTable* rt = nullptr);

// Recompile a vertex shader to a vertex SPIR-V module: v0 = gl_VertexIndex, run the VALU, and on EXP
// to a POS target write vec4(src0..3) to gl_Position. Returns {} if unsupported / no position export.
// An optional ShaderResourceTable enables vertex fetch (buffer_load_format_*) + constant loads.
std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords,
                                       const ShaderResourceTable* rt = nullptr);

// How much of a shader the recompiler currently covers (per-instruction), without requiring the
// stream to be a complete vertex/fragment. `alu` = instructions emit_alu handles (VALU/scalar/
// control-flow); `exports` = EXP (handled by the stage recompilers); `unsupported` = not yet handled
// (memory ops / unknown), with the first such (format, opcode) recorded. A data-driven coverage metric.
struct RecompileCoverage {
    uint32_t total = 0, alu = 0, exports = 0, unsupported = 0;
    // Instructions the recompiler handles GIVEN the right context (a resource table for MIMG /
    // buffer_load_format, or a fragment stage for VINTRP) but that recompile_coverage — which runs
    // table-less on a compute shell — cannot exercise. Counting them apart from `unsupported` gives an
    // honest "recompilable in context" number; the table-less `alu`/`unsupported` split understates it.
    uint32_t table_dependent = 0;
    int      first_bad_fmt = -1;   // Rdna2Format of the first TRULY-unsupported instruction (-1 if none)
    uint32_t first_bad_op  = 0;
};
RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords);

} // namespace prosper::gpu
