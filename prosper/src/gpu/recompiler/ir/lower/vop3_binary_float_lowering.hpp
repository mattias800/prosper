// Borrow the root's synchronous modifier closures, preserving CLAMP routing and predication.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using BinaryFloatSource = std::function<uint32_t(int)>;
using BinaryFloatResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_binary_float_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                    bool allow_wave, const BinaryFloatSource& fv,
                                    const BinaryFloatResult& fresult);
template <class Read, class Modify>
bool lower_vop3_binary_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                             bool allow_wave, Read& fv, Modify& fresult) {
    const BinaryFloatSource read = std::cref(fv);
    const BinaryFloatResult modify = std::cref(fresult);
    return lower_vop3_binary_float_erased(b, rs, in, ok, allow_wave, read, modify);
}
}   // namespace prosper::gpu::ir
