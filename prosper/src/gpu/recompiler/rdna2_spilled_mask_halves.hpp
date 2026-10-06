// Which saved Wave64 mask a V_WRITELANE spill slot, and the SGPR a V_READLANE reloads it into,
// holds one half of. A CFG MUST fact carried beside the CFG dispatcher's Wave64 mask analysis, so
// `s_mov_b64 dst, s[N:N+1]` over two reloaded halves of one mask can stay a mask.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {

// One 32-bit half of a saved B64 mask pair. `def_pc` names the INSTANCE: the pc of the mask write
// that defined the pair. A new write at that pc ends every older fact naming it, so on any one
// path all live facts with the same `def_pc` describe the same value.
struct SpilledMaskHalf {
    int pair = -1;          // low SGPR of the saved pair
    int half = -1;          // 0 = low word, 1 = high word
    uint32_t def_pc = 0;
    bool operator==(const SpilledMaskHalf&) const = default;
};

struct SpilledMaskHalves {
    std::map<int, uint32_t> pair_def_pc;   // live MUST-mask pair -> defining pc
    std::map<std::pair<int, int>, SpilledMaskHalf> slots;   // (vgpr, lane) -> spilled half
    std::map<int, SpilledMaskHalf> sregs;   // SGPR -> reloaded half
    bool operator==(const SpilledMaskHalves&) const = default;
};

// CFG join: keep only the facts every incoming path agrees on.
void meet_spilled_mask_halves(SpilledMaskHalves& into, const SpilledMaskHalves& incoming);

// True when `in` is `s_mov_b64 dst, s[N:N+1]` and sN / sN+1 hold the low / high half of the same
// saved-mask instance. Evaluated BEFORE `in`'s own transfer.
bool reassembles_spilled_mask_pair(const SpilledMaskHalves& state, const Rdna2Inst& in);

// The transfer for one instruction. `masks` is the MUST-mask set before `in` (a V_WRITELANE never
// changes it), `mask_slot_keys` the pairs whose spills the dispatcher persists as mask slots,
// `scalar_writes` the (base, width) SGPR ranges `in` writes, `vector_writes` the VGPRs it may
// write (V_WRITELANE and V_READLANE are modelled here instead), and `mask_write` the pair `in`
// defines as a mask, or -1.
void advance_spilled_mask_halves(SpilledMaskHalves& state, const Rdna2Inst& in,
                                 const std::set<int>& masks, const std::set<int>& mask_slot_keys,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write);

}   // namespace prosper::gpu
