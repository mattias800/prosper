// VOP3P MIX source preparation and predicated result publication; no admission policy change.
#pragma once

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
bool lower_vop3p_mixed_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            bool allow_wave);
}   // namespace ir
}   // namespace prosper::gpu
