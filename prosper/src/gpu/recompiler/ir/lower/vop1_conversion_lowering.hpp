// Conversion routes keep their existing predicate calls; value lowering leaves final stages to emit_alu.
#pragma once
#include <cstdint>
#include <optional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
std::optional<bool> lower_vop1_conversion_routes(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                                 uint32_t a, uint32_t& d, uint32_t old_d, bool& ok);
bool lower_vop1_conversion_values(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, uint32_t a,
                                  uint32_t& d, uint32_t old_d, bool& ok);
}   // namespace prosper::gpu::ir
