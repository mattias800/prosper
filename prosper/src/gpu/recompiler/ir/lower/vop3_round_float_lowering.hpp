// Numeric value lowering only; borrow the root's modifier callbacks synchronously.
#pragma once
#include "gpu/recompiler/ir/lower/vop3_binary_float_lowering.hpp"
namespace prosper::gpu::ir {
bool lower_vop3_round_float(SpirvCompute& b, RegState& rs, const Rdna2Inst& in, bool& ok,
                            const BinaryFloatSource& fv, const BinaryFloatResult& fresult);
}
