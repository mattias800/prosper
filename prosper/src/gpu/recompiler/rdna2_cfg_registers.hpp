#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <set>

namespace prosper::gpu {
int shader_max_vgpr(const std::vector<Rdna2Inst>&);
template <typename Callback>
void for_each_possible_vector_write(const Rdna2Inst& in, int max_observable_vgpr,
                                    const Callback& callback) {
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32) {
        for (int reg = in.dst.value; reg <= max_observable_vgpr; ++reg) callback(reg);
        return;
    }
    for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word)
        callback(in.dst.value + static_cast<int>(word));
}
// Register STORAGE/effect inventory, not initialized guest VALUE or per-lane validity authority.
// Native VINTRP P2/MOV publish VDST; native P1 is a no-op. The owned packet also executes P1,
// so it must reload VDST for partial interpolation and inactive-word preservation.
void loop_written_regs(const std::vector<Rdna2Inst>&, uint32_t lo, uint32_t hi,
                       std::set<int>& vgprs, std::set<int>& sgprs, bool owned_packet = false);
// Scalar registers that MAY be overwritten in [lo, hi), mask-pair destinations included.
void loop_scalar_may_writes(const std::vector<Rdna2Inst>&, uint32_t lo, uint32_t hi,
                            std::set<int>& sgprs);
struct RegState;
// A loop body that may write these registers leaves no entry-time descriptor provenance or
// user-data alias standing for them at the header.
void invalidate_loop_descriptor_provenance(RegState& rs, const std::set<int>& sgprs);
} // namespace prosper::gpu
