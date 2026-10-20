// Packed floating-point source expansion, result packing and existing EXEC publication.
#pragma once

#include <optional>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
namespace ir {
std::optional<bool> lower_vop3p_packed_values(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                              bool& ok, bool allow_wave);
}
}   // namespace prosper::gpu
