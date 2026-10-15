// nullopt leaves the VOP1 pipeline in charge; a value owns the instruction, including refusal.
#pragma once
#include <cstdint>
#include <optional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
std::optional<bool> lower_vop1_unary_f16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                         uint32_t a, uint32_t& d, uint32_t old_d, bool& ok);
}
