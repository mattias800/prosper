// structured_loop_cursor.hpp -- where emit_body's structured route reads its loop records
// (IR_CONTROL_FLOW_PLAN.md step 4b).
//
// The divergent-loop cursor (`emit_structured` over DivLoop) and the counted route's CountedLoop
// read the region tree's Loop and CountedLoop regions, but only when the tree's complete loop
// sequence equals the detector's records field for field and in the same order. Any difference,
// an Unstructured region or a malformed walk keeps the detector's records for the whole program.
// Equal records make emission byte-identical by construction. Register state, joins, loop carry
// and proofs stay in emit_body; nothing here allocates a SPIR-V id.
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The admission inputs emit_body passes to detect_divergent_loops and detect_counted_loop, as a
// RegionPolicy. `linearized` is the stage's `safe` set; fragment programs take the forced wave64
// vote (`exact_fragment_wave_breaks`).
RegionPolicy structured_loop_policy(bool compute, bool fragment,
                                    const std::unordered_set<uint32_t>* linearized,
                                    const uint32_t* code, size_t dwords);

// Which source the records came from, and why. Tree is the only route that reads the tree.
enum class LoopRecordRoute : uint8_t { Tree, NoLoops, DetectorRefused, Unstructured, Mismatch };

class StructuredLoopCursor {
public:
    // `loops` is emit_body's final DivLoop vector (after the compute guards and the cf_rejected
    // clear); the cursor keeps a reference to it.
    StructuredLoopCursor(const std::vector<Rdna2Inst>& ins, const std::vector<DivLoop>& loops,
                         const RegionPolicy& policy, bool detector_refused);
    LoopRecordRoute route() const { return route_; }
    bool uses_tree() const { return route_ == LoopRecordRoute::Tree; }
    // The next loop's header pc when it lies below `hi`, otherwise `hi`.
    uint32_t next_before(uint32_t hi) const;
    // The next loop's header pc, UINT32_MAX after the last one.
    uint32_t next_pc() const;
    // Consumes the next loop. The reference lives as long as the cursor (and the detector's
    // vector).
    const DivLoop& take();

private:
    const std::vector<DivLoop>& records() const { return uses_tree() ? tree_loops_ : detector_; }
    const std::vector<DivLoop>& detector_;
    std::vector<DivLoop> tree_loops_;
    size_t next_ = 0;
    LoopRecordRoute route_ = LoopRecordRoute::NoLoops;
};

struct CountedLoopRecord {
    CountedLoop loop;
    LoopRecordRoute route = LoopRecordRoute::NoLoops;
};

// The counted route's loop: the tree's single CountedLoop when it equals `detector` field for
// field, otherwise `detector`. NoLoops when the detector found none (the tree is not built).
CountedLoopRecord counted_loop_record(const std::vector<Rdna2Inst>& ins,
                                      const CountedLoop& detector, const RegionPolicy& policy);

}   // namespace prosper::gpu::ir::cfg
