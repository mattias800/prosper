// The caller owns destination capture and final EXEC-predicated publication.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
bool lower_vop3_pack16(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                       bool allow_wave);
// The existing full-dword RTZ pack borrows the root float-source modifier closure.
using RtzPackSource = std::function<uint32_t(int)>;
bool lower_vop3_rtz_pack_erased(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                                bool allow_wave, const RtzPackSource& fv);
template <class Read>
bool lower_vop3_rtz_pack(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                         bool allow_wave, Read& fv) {
    const RtzPackSource read = std::cref(fv);
    return lower_vop3_rtz_pack_erased(b, rs, in, ok, allow_wave, read);
}
}
