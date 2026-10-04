#pragma once

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;

// Lowers RDNA2 S_MOVRELS_B32 (SOP1 opcode 0x2e): relative SGPR move indexed by M0.
bool emit_s_movrels_b32(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);

}  // namespace prosper::gpu
