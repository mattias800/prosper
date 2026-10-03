#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <set>

namespace prosper::gpu {
int shader_max_vgpr(const std::vector<Rdna2Inst>&);
// Register STORAGE/effect inventory, not initialized guest VALUE or per-lane validity authority.
// Native callers preserve their historical effect domain. The owned packet actually executes
// VINTRP, so it must reload VDST even for P1's inactive-word preservation and P2's OLD source.
void loop_written_regs(const std::vector<Rdna2Inst>&, uint32_t lo, uint32_t hi,
                       std::set<int>& vgprs, std::set<int>& sgprs, bool owned_packet = false);
} // namespace prosper::gpu
