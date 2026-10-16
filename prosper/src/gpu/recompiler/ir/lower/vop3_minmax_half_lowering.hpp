// The root owns OMOD/CLAMP and EXEC publication; dispatch borrows its result closure.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using MinmaxHalfResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_minmax_half_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                   bool allow_wave, uint32_t old_d,
                                   const MinmaxHalfResult& fresult);
template <class Modify>
bool lower_vop3_minmax_half(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            bool allow_wave, uint32_t old_d, Modify& fresult) {
    const MinmaxHalfResult modify = std::cref(fresult);
    return lower_vop3_minmax_half_erased(b, rs, in, ok, allow_wave, old_d, modify);
}
}   // namespace prosper::gpu::ir
