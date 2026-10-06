// Which physical half of a Wave64 mask a V_WRITELANE spill slot, and the SGPR a V_READLANE
// reloads it into, holds -- the CFG MUST analysis behind wave64_mask_readlane_half_for_pc and
// wave64_mask_writelane_alias_pcs. Exact native Wave64 compute only.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

namespace prosper::gpu {

struct SpirvCompute;

// Calls its second argument with every VGPR the instruction may write.
using VectorWriteVisitor = std::function<void(const Rdna2Inst&, const std::function<void(int)>&)>;

// V_WRITELANE/V_READLANE scalar spills can carry one physical half of a Wave64 mask through a
// loop. The Bool slot alone loses whether it was LO or HI, while a dispatcher uint placeholder
// is not a validity tag. Track that identity as a CFG MUST fact and publish it only at exact
// native-Wave64 readlane PCs (into `b`). Joins retain equal facts; every ordinary overwrite kills
// them. The two vectors receive each block's entry SGPR->half facts and reachability.
void analyze_wave64_mask_half_aliases(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const std::vector<uint32_t>& starts,
    const std::vector<std::vector<uint32_t>>& successors,
    const VectorWriteVisitor& for_each_possible_vector_write,
    std::vector<std::map<int, uint32_t>>& wave64_mask_half_sreg_in,
    std::vector<bool>& wave64_mask_half_reachable);

}   // namespace prosper::gpu
