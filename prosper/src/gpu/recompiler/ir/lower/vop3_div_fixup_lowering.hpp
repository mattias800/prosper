// Reference wrappers preserve the root's modifier closures without copying or allocating them.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using DivFixupSource = std::function<uint32_t(int)>;
using DivFixupResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_div_fixup_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                 bool allow_wave, const DivFixupSource& fv,
                                 const DivFixupResult& fresult);
template <class Read, class Modify>
bool lower_vop3_div_fixup(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, Read& fv, Modify& fresult) {
    const DivFixupSource read = std::cref(fv);
    const DivFixupResult modify = std::cref(fresult);
    return lower_vop3_div_fixup_erased(b, rs, in, ok, allow_wave, read, modify);
}
}   // namespace prosper::gpu::ir
