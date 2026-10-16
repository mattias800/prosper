// Root-owned float modifiers stay at their existing call sites; the exponent uses raw word reads.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using Vop3LdexpSource = std::function<uint32_t(int)>;
using Vop3LdexpResult = std::function<uint32_t(uint32_t)>;
bool lower_vop3_ldexp(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok, bool allow_wave,
                      const Vop3LdexpSource& fv, const Vop3LdexpResult& fresult);
}   // namespace prosper::gpu::ir
