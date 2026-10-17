// Producing-stage policy and raw-half bookkeeping; exact arithmetic lives in ops/.
#pragma once
#include <array>
#include <cstdint>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool use_half_fma_finite(const SpirvCompute& b, const Rdna2Inst& in);
uint32_t emit_native_half_fma_finite(SpirvCompute& b, const Rdna2Inst& in,
                                     const std::array<uint32_t, 3>& raw, uint32_t fallback);
uint32_t emit_packed_half_fma_finite(SpirvCompute& b, const Rdna2Inst& in,
                                     const std::array<std::array<uint32_t, 3>, 2>& raw,
                                     uint32_t fallback);
uint32_t emit_packed_fmac_half_finite(SpirvCompute& b, uint32_t a, uint32_t c, uint32_t old_d,
                                      uint32_t fallback);
}   // namespace prosper::gpu::ir
