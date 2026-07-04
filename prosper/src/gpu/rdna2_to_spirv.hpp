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

// Translate a straight-line float-VALU RDNA2 stream to a compute-shader SPIR-V module.
// Returns {} if the stream contains an opcode/format this stage does not yet handle.
std::vector<uint32_t> recompile_valu(const uint32_t* code, size_t dwords,
                                     uint32_t num_inputs, uint32_t out_vgpr);

// Recompile a pixel/fragment shader to a fragment SPIR-V module: run the VALU, and on EXP to an MRT
// target write vec4(src0..3) to the location-0 color output. Returns {} if unsupported / no export.
std::vector<uint32_t> recompile_fragment(const uint32_t* code, size_t dwords);

// Recompile a vertex shader to a vertex SPIR-V module: v0 = gl_VertexIndex, run the VALU, and on EXP
// to a POS target write vec4(src0..3) to gl_Position. Returns {} if unsupported / no position export.
std::vector<uint32_t> recompile_vertex(const uint32_t* code, size_t dwords);

// How much of a shader the recompiler currently covers (per-instruction), without requiring the
// stream to be a complete vertex/fragment. `alu` = instructions emit_alu handles (VALU/scalar/
// control-flow); `exports` = EXP (handled by the stage recompilers); `unsupported` = not yet handled
// (memory ops / unknown), with the first such (format, opcode) recorded. A data-driven coverage metric.
struct RecompileCoverage {
    uint32_t total = 0, alu = 0, exports = 0, unsupported = 0;
    int      first_bad_fmt = -1;   // Rdna2Format of the first unsupported instruction (-1 if none)
    uint32_t first_bad_op  = 0;
};
RecompileCoverage recompile_coverage(const uint32_t* code, size_t dwords);

} // namespace prosper::gpu
