// Explicit half-step constants keep every intermediate of the old RPI graph exact.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool lower_vop3_cvt_rpi_exact(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok);
} // namespace ir
} // namespace prosper::gpu
