// state_placement.hpp -- which state variables get a phi at each structured join
// (IR_CONTROL_FLOW_PLAN.md step 6).
//
// One pass over the region tree computes, per loop header and per if merge, the registers the
// structured emitter phis there: the VGPR and SGPR sets it used to derive in three places
// (`loop_written_regs` at loop headers, the per-arm write sets at if merges, and the V_WRITELANE
// slots of `LaneSlotLoopCarry`). The writes come from `rdna2_instruction_effects` and the shared
// width helpers; there is no opcode list here. VCC, SCC, EXEC and the saved lane masks are not
// in these sets: the emitter phis them from RegState, as before.
//
// Plain data: building it allocates no SPIR-V id. The emitter takes a set from it only when it
// equals the emitter's own set element for element (`take_placed_if_equal`), so emission is
// byte-identical by construction. Where the inventory differs from the legacy one, and why, is in
// AGENTS.md next to this file.
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The registers one join phis, by register number. std::set: the emitter emits its phis in this
// order, and that order is the byte-identity contract.
struct PhiSets {
    std::set<int> vgprs, sgprs;
    bool operator==(const PhiSets&) const = default;
};

// Constant-lane V_WRITELANE spill slots, as (vgpr, lane).
using LaneSlotSet = std::set<std::pair<int, int>>;

struct LoopPlacement {
    uint32_t header_pc = 0;
    PhiSets carried;   // written in [header, back edge): one header phi each
    PhiSets condition;   // written in [header, exit test): the check block's value leaves the loop
    LaneSlotSet lane_slots;   // written in [header, back edge): one header phi each
};

struct IfPlacement {
    uint32_t branch_pc = 0;
    PhiSets merge;   // written in either arm: a merge phi each where the edges differ
};

struct StatePlacement {
    std::vector<IfPlacement> ifs;   // ascending branch_pc
    std::vector<LoopPlacement> loops;   // ascending header_pc
    const IfPlacement* if_at(uint32_t branch_pc) const;
    const LoopPlacement* loop_at(uint32_t header_pc) const;
};

// The registers one instruction adds to a join's sets (the inventory the rules in AGENTS.md
// describe). `max_vgpr` bounds an M0-relative destination.
void add_phi_writes(const Rdna2Inst& in, int max_vgpr, PhiSets& out);
// The SGPRs from SDST up that one SOP2 result charges, from the instruction-effects table: one
// word; the pair for a 64-bit scalar-data result; nothing for a 64-bit shift into EXEC. The
// emitter's scan (loop_written_regs) charges SOP2 through this too, so the two cannot disagree.
uint32_t sop2_phi_words(const Rdna2Inst& in);

// Every IfThen, IfElse, Loop and CountedLoop reachable from the root. Empty when the root is
// Unstructured. `ins` is the instruction vector `cfg` and `tree` were built from.
StatePlacement place_state(const GuestCfg& cfg, const RegionTree& tree,
                           const std::vector<Rdna2Inst>& ins);
// Builds the CFG and the region tree under `policy` first.
StatePlacement place_state(const std::vector<Rdna2Inst>& ins, const RegionPolicy& policy);

// ---- The emitter's guard, and the census that observes it ----------------------------------

enum class PhiSite : uint8_t { IfMerge, LoopCarried, LoopCondition, LaneSlots };
const char* phi_site_name(PhiSite site);

// Replaces the emitter's own sets with the placement's only when they are equal element for
// element. A missing placement or any difference keeps the emitter's sets. Either way the
// comparison is reported to the active census, if any.
void take_placed_if_equal(const PhiSets* placed, PhiSite site, uint32_t pc, std::set<int>& vgprs,
                          std::set<int>& sgprs);
void take_placed_if_equal(const LaneSlotSet* placed, uint32_t pc, LaneSlotSet& slots);

struct PlacementSiteRecord {
    enum class Verdict : uint8_t { Equal, Differs, Absent };
    PhiSite site = PhiSite::IfMerge;
    uint32_t pc = 0;
    Verdict verdict = Verdict::Absent;
    std::string difference;   // Differs: "emitter-only=... placed-only=..."
};

// Observes every guarded join on this thread while it lives (the shadow census, the gtest).
// `withhold` makes the guard return before comparing, which is the emitter as it was before
// step 6: the test compiles a program both ways and requires identical modules.
class StatePlacementCensus {
public:
    explicit StatePlacementCensus(bool withhold = false);
    ~StatePlacementCensus();
    StatePlacementCensus(const StatePlacementCensus&) = delete;
    StatePlacementCensus& operator=(const StatePlacementCensus&) = delete;
    const std::vector<PlacementSiteRecord>& sites() const { return sites_; }
    bool withhold() const { return withhold_; }
    void record(PlacementSiteRecord site) { sites_.push_back(std::move(site)); }
    static StatePlacementCensus* active();

private:
    std::vector<PlacementSiteRecord> sites_;
    StatePlacementCensus* previous_ = nullptr;
    bool withhold_ = false;
};

}   // namespace prosper::gpu::ir::cfg
