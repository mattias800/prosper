// Source float modifiers stay root-owned; packed output has no float-result callback.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using Vop3PackNormSource = std::function<uint32_t(int)>;
bool lower_vop3_pack_norm(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                          bool allow_wave, const Vop3PackNormSource& fv);
}   // namespace prosper::gpu::ir
