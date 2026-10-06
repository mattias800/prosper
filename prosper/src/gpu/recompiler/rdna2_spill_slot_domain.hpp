// What a V_WRITELANE spill slot holds on every path reaching a point: scalar data, a wave mask, or
// something the CFG dispatcher cannot type. A CFG fact carried beside the dispatcher's Wave64 mask
// analysis, so a V_READLANE reload gets the domain emit_alu actually gave it (#4600).
//
// The dispatcher persists a slot in a uint (data) and/or a Bool (mask) Function variable. Neither
// carries a type tag, so a reload must not trust whichever variable happens to exist: on a path
// where the slot last held a mask, its uint variable is a zero placeholder, and the reverse.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu {

// The value a slot holds. MaskLo / MaskHi name which ballot word of the stored Bool the slot's
// architectural dword is; MaskAny is a mask whose word is not known. Other is a value the analysis
// cannot prove either way, which keeps the dispatcher's old untyped behaviour. Conflict is a mask
// on one path and not a mask on another.
enum class SpillSlotDomain : uint8_t { Data, MaskLo, MaskHi, MaskAny, Other, Conflict };

// (vgpr, lane) -> domain. An absent slot is unwritten on every reaching path, and an unwritten
// path does not constrain a join: the slot's value is undefined there.
using SpillSlotDomains = std::map<std::pair<int, int>, SpillSlotDomain>;

bool is_mask_domain(SpillSlotDomain domain);

// CFG join, per slot.
void meet_spill_slot_domains(SpillSlotDomains& into, const SpillSlotDomains& incoming);

// The Wave64 MUST analysis' sets, after the generic transfer of the instruction. A V_WRITELANE
// changes none of them, so for a write they are also the state before it.
struct SpillSlotMustSets {
    std::set<int>& masks;           // MUST B64 mask bases
    std::set<int>& ambiguous;       // a mask on some paths, data on others
    std::set<int>& scalar_words;    // MUST scalar-data words
    std::set<int>& readlane_words;  // words an untyped V_READLANE may have written last
};

struct SpillSlotContext {
    const std::set<int>& mask_keys;   // SGPRs the dispatcher persists in a Bool variable
    bool native_wave64;               // one exact 64-lane host subgroup per guest wave
    const std::unordered_map<uint32_t, uint32_t>& proven_half_readlane_pcs;
    const std::unordered_set<uint32_t>& proven_half_writelane_pcs;
};

// True when a V_WRITELANE's SGPR source is, on every path, ordinary scalar data: a MUST scalar
// word outside any mask or ambiguous pair, and not the result of an untyped V_READLANE.
bool spill_source_is_scalar_data(const Rdna2Inst& in, const std::set<int>& masks,
                                 const std::set<int>& ambiguous, const std::set<int>& scalar_words,
                                 const std::set<int>& readlane_words);

// The transfer for one compute instruction. A constant-lane V_WRITELANE records the domain of its
// source; any other write of a spill VGPR (`vector_writes`) forgets its slots. A constant-lane
// V_READLANE retypes its destination in `sets`: a data reload stays a scalar word, a mask reload
// becomes a mask and no scalar word, and a reload that is a mask on some paths becomes ambiguous,
// so its first read is refused. Adds each V_WRITELANE that stores a mask to `mask_writelane_pcs`
// when it is given.
void advance_spill_slot_domains(SpillSlotDomains& state, const Rdna2Inst& in,
                                const SpillSlotMustSets& sets, const SpillSlotContext& context,
                                const std::vector<int>& vector_writes,
                                std::unordered_set<uint32_t>* mask_writelane_pcs);

}   // namespace prosper::gpu
