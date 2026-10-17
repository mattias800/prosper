// vop3_cross_lane_lowering.hpp — the VOP3 operations whose result depends on another lane's data:
// v_writelane_b32 (0x361), v_readlane_b32 (0x360) and v_permlane16/permlanex16_b32 (0x377/0x378).
// Returns nullopt when the instruction is none of them; otherwise what emit_alu returns.
#pragma once

#include <optional>

namespace prosper::gpu {
struct SpirvCompute;
struct RegState;
struct Rdna2Inst;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {

std::optional<bool> lower_vop3_cross_lane(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                          bool& ok, bool allow_wave);

}   // namespace prosper::gpu::ir
