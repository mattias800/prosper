// Root-owned float modifier closures are borrowed only for this synchronous dispatch.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using LegacyMulSource = std::function<uint32_t(int)>;
using LegacyMulResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_legacy_mul_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                  bool allow_wave, const LegacyMulSource& fv,
                                  const LegacyMulResult& fresult);
template <class Read, class Modify>
bool lower_vop3_legacy_mul(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                           bool allow_wave, Read& fv, Modify& fresult) {
    const LegacyMulSource read = std::cref(fv);
    const LegacyMulResult modify = std::cref(fresult);
    return lower_vop3_legacy_mul_erased(b, rs, in, ok, allow_wave, read, modify);
}
}   // namespace prosper::gpu::ir
