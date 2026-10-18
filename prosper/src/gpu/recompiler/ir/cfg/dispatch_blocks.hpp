// dispatch_blocks.hpp -- the CFG dispatcher's block list and successor relation, read from GuestCfg.
//
// IR_CONTROL_FLOW_PLAN.md step 5. emit_cfg_state_machine numbers its cases by block (`starts`) and
// walks `successors` in its dataflow passes. adopt_guest_cfg_blocks builds the same two from
// build_guest_cfg with SplitPolicy::split_at_services and hands them to the dispatcher only when
// they are identical to the dispatcher's own, so emission is byte-identical by construction. Every
// verdict can be observed with DispatchBlocksCapture (the shadow test and `shader_inspect
// --cfg-dispatch-blocks`). Contract: AGENTS.md next to this file.
#pragma once

#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace prosper::gpu::ir::cfg {

// The dispatcher's form of a block graph.
struct DispatchBlocks {
    std::vector<uint32_t> starts;   // block start pcs, ascending; case k runs [starts[k], starts[k+1])
    // Per block: the taken successor, then the fall-through one, without repeats. A target past
    // s_endpgm has no edge (the per-case emission handles a proven early exit itself).
    std::vector<std::vector<uint32_t>> successors;
};

// The dispatcher's view of a GuestCfg built with split_at_services.
DispatchBlocks dispatch_blocks(const GuestCfg& cfg);

// "" when equal, else the first difference in words (block count, a start, or a successor list).
std::string dispatch_blocks_difference(const DispatchBlocks& dispatcher, const DispatchBlocks& cfg);

// The step 5 guard. Builds the GuestCfg for `ins` under `policy`; when its blocks and successors
// equal `starts` and `successors`, replaces them with the GuestCfg's and returns true. Otherwise
// leaves them as they are and returns false. Records the verdict for DispatchBlocksCapture.
bool adopt_guest_cfg_blocks(std::span<const Rdna2Inst> ins, const SplitPolicy& policy,
                            std::vector<uint32_t>& starts,
                            std::vector<std::vector<uint32_t>>& successors);

// One adopt_guest_cfg_blocks call.
struct DispatchBlocksRecord {
    uint32_t entry_pc = 0;   // pc of the region's first instruction
    size_t blocks = 0;   // the dispatcher's own block count
    size_t cfg_blocks = 0;   // the GuestCfg's
    size_t services = 0;   // GuestCfg blocks that end in a service (GuestBlock::service_event)
    bool adopted = false;
    std::string difference;   // dispatch_blocks_difference, "" when adopted
};

// Observation only: while alive, collects every adopt_guest_cfg_blocks verdict made on this thread,
// in order. Nested captures restore the outer one when they end. Nothing reads the records back
// into compilation.
class DispatchBlocksCapture {
public:
    DispatchBlocksCapture();
    ~DispatchBlocksCapture();
    DispatchBlocksCapture(const DispatchBlocksCapture&) = delete;
    DispatchBlocksCapture& operator=(const DispatchBlocksCapture&) = delete;
    const std::vector<DispatchBlocksRecord>& records() const { return records_; }

private:
    std::vector<DispatchBlocksRecord> records_;
    std::vector<DispatchBlocksRecord>* previous_ = nullptr;
};

}   // namespace prosper::gpu::ir::cfg
