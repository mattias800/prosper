#pragma once

// INTERNAL recompiler companion: constant-lane V_WRITELANE spill slots across the STRUCTURED
// emitter's joins -- loop headers, loop exits and if merges (rdna2_emit_cfg.cpp).
//
// `v_writelane_b32 vN, sX, <const lane>` parks a wave-uniform scalar in one lane of a VGPR, and
// `v_readlane_b32 sY, vN, <const lane>` reloads it. emit_alu models each (vgpr, lane) as a named
// SSA value in RegState::vgpr_lane_slots (data) or vgpr_lane_mask_slots (a spilled lane mask). The
// CFG dispatcher already persists those slots in Function variables. The structured emitter gave
// SGPRs, VGPRs, SCC, VCC, EXEC and saved masks a phi at every join -- but never the slots, so:
//
//   * a loop header read the PREHEADER value on every iteration. Kena's level-load pixel program
//     0x5007ad0000 spills its loop counter to v20 lane 40 (written at the latch, reloaded at the
//     header), so the recompiled exit test was `0 < count`: an infinite loop on the GPU and a
//     device loss (`ring gfx timeout`, an empty RADV vm_fault.log);
//   * a loop exit, or an if merge, used the SSA id the body or arm left, which does not dominate the
//     merge (invalid SPIR-V), and an if/else lost the then arm's writes outright.
//
// The rules mirror the scalar ones beside them. A slot written inside a loop gets a header phi;
// the merge takes the value the exit edge carried. An if merge phis a slot whose value differs; a
// slot unwritten on one edge takes a placeholder there, because an unwritten lane is undefined on
// that path (the same convention as the absent-SGPR zero and as SpillSlotDomains). What cannot be
// represented never reaches a reload silently:
//   * an if merge marks a slot written on only one edge as fabricated (#4725's mark, so a lane
//     projection refuses it), and leaves an EMPTY spill array -- which every later read refuses --
//     where one edge ended the array or the edges disagree on a slot's data/mask domain;
//   * a loop refuses a slot it reloads when the body leaves no value for it in the phi's domain:
//     the body ended its lifetime (an ordinary VGPR write) or left it in the other data/mask
//     domain. A slot the loop never reloads closes with its loop-invariant seed, which nothing
//     inside the loop reads, and leaves the loop as a missing lane of a kept (possibly EMPTY) spill
//     array, so a reload after the exit refuses too.
//
// CONFIDENCE: HIGH for loops (the header phi is the standard loop-carried form, and the merge value
// is the one that reaches the exit). HIGH for if merges with the placeholder caveat above.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu {

struct SpirvCompute;
struct RegState;

// (vgpr, lane) pairs that a constant-lane v_writelane_b32 in [lo, hi) writes.
std::set<std::pair<int, int>> lane_slots_written(const std::vector<Rdna2Inst>& ins, uint32_t lo,
                                                 uint32_t hi);

// One predecessor's slot state at a structured merge.
struct LaneSlotEdge {
    std::unordered_map<int, std::unordered_map<int, uint32_t>> data, mask;
    std::unordered_set<int> invalidated;
    uint32_t block = 0;
    static LaneSlotEdge of(const RegState& rs, uint32_t block);
};

// Join two predecessors at an if merge whose label is the current block. `rs` receives the joined
// slots; every other field of `rs` is left alone. Emits only OpPhi, so it may run among the merge's
// other phis. A slot written on one edge only is marked fabricated. A spill array that one edge
// ended, or whose slot is data on one edge and a mask on the other, is left with that slot (or every
// slot) missing, so a reload refuses; an array left with no slot stays as an EMPTY entry, so an
// ordinary read of the VGPR refuses too.
void join_lane_slots(SpirvCompute& b, RegState& rs, const LaneSlotEdge& first,
                     const LaneSlotEdge& second, uint32_t branch_pc);

// The loop-carried slots of one structured loop.
class LaneSlotLoopCarry {
public:
    // Before mark_loop_carried(): which slots were fabricated-marked on the preheader edge.
    void note_preheader_marks(const RegState& rs);
    // At the header label, after the other header phis: one phi per slot written in
    // [header_pc, backedge_pc), seeded from the preheader (a placeholder when it is unwritten there).
    void open(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins, uint32_t header_pc,
              uint32_t backedge_pc, uint32_t preheader);
    // At the end of the check block: the values the exit edge carries.
    void record_check(const RegState& rs);
    // In the continue block: close each phi with the body's value. False (logged) when the body
    // left a slot the loop reloads without a value in the phi's domain.
    bool patch_backedge(SpirvCompute& b, const RegState& rs, uint32_t cont);
    // At the merge label. Without a body edge the merge's only predecessor is the check block;
    // with one (a direct break from the body) each slot is phi'd between `check_end` and `body_end`.
    // False (logged) when the body edge holds a slot in the other domain or not at all.
    bool finish_exit(SpirvCompute& b, RegState& rs, bool body_edge, uint32_t check_end,
                     uint32_t body_end) const;
    // After mark_loop_exit_slots(), for an exit without a body edge: a carried slot the check block
    // left as its header phi is fabricated exactly when its preheader value or its back-edge value
    // was. mark_loop_carried() marked the phi before the body existed; the back-edge mark, computed
    // under that pessimistic assumption, is an upper bound, so this only ever lifts a mark that no
    // input carried (#4749 review: Kena's EXEC_HI re-spill at pc 853).
    void refine_exit_marks(RegState& rs) const;
    size_t carried() const { return slots_.size(); }

private:
    struct Slot {
        int vgpr;
        int lane;
        bool mask;
        uint32_t phi;
        size_t patch;
        uint32_t at_check;   // the value on the exit edge; 0 when the check block names none
        uint32_t seed;   // the preheader value, or the placeholder for an unwritten lane
        bool seeded;   // the preheader held this slot
        bool uncarried;   // the back-edge closed with `seed`, not with a body value
        bool seed_marked;   // the preheader value is fabricated (or the lane was unwritten there)
        bool backedge_marked;   // the body's value on the back edge is fabricated
    };
    std::vector<Slot> slots_;
    std::set<std::pair<int, int>> preheader_marks_;
    std::set<std::pair<int, int>> read_in_loop_;   // constant-lane V_READLANEs in the loop
    uint32_t header_pc_ = 0;
};

}   // namespace prosper::gpu
