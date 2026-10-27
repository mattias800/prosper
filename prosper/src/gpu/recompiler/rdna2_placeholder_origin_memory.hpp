// Observe successful scalar-load publication without granting any new load or register validity.
#pragma once
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
void note_placeholder_scalar_load(const SpirvCompute&, const RegState&, const Rdna2Inst&);
} // namespace prosper::gpu
