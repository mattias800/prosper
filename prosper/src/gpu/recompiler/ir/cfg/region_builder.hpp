// region_builder.hpp -- the region tree's builder, internal to ir/cfg. build_region_tree
// (region_tree.hpp) is the interface. region_tree.cpp implements the walk, blocks and ifs;
// region_loops.cpp finds the loops in the CFG and builds Loop and CountedLoop regions.
#pragma once

#include "gpu/recompiler/ir/cfg/region_tree.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace prosper::gpu::ir::cfg::detail {

// One loop the CFG's back edges name and the detectors' admission rules accept, in block terms.
struct FoundLoop {
    uint32_t header = kNone, latch = kNone, exit_branch = kNone;
    std::vector<uint32_t> breaks;
    bool counted = false;
    BranchCondition::Flag flag = BranchCondition::Flag::None;
    LoopShape shape;
};

class RegionBuilder {
public:
    RegionBuilder(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins,
                  const RegionPolicy& policy);
    RegionTree build();

private:
    // region_tree.cpp: the walk, blocks and ifs.
    const Rdna2Inst& last_of(uint32_t block) const { return ins_[cfg_.blocks[block].last]; }
    uint32_t range_end_pc(uint32_t end) const;
    bool fail(std::string why);
    uint32_t push(Region r);
    uint32_t leaf(uint32_t b);
    uint32_t seq(uint32_t begin, uint32_t end);
    uint32_t step(uint32_t b, uint32_t end, uint32_t* next);
    uint32_t conditional(uint32_t b, uint32_t end, uint32_t* next);
    bool early_out(uint32_t b, uint32_t end, Region* r);
    bool forward_if(uint32_t b, uint32_t end, uint32_t target, Region* r);
    bool if_else(uint32_t b, uint32_t end, uint32_t jump, Region* r);
    bool admit(const Rdna2Inst& br, uint32_t target, Region& r);

    // region_loops.cpp: loops.
    bool find_loops();
    bool reducible(uint32_t latch, uint32_t header) const;
    bool sort_back_edges(std::vector<uint32_t>* counted, std::vector<uint32_t>* divergent);
    bool counted_route(uint32_t latch, const std::vector<uint32_t>& divergent, bool* taken);
    bool divergent_loops(const std::vector<uint32_t>& latches);
    bool record(const FoundLoop& loop, uint32_t backedge_pc);
    bool single_entry_and_exit(const FoundLoop& loop);
    uint32_t block_ending_at(uint32_t pc) const;
    uint32_t loop(uint32_t b, uint32_t end, uint32_t* next);
    uint32_t loop_break(uint32_t b, uint32_t end, uint32_t* next, bool uniform_vcc);

    const GuestCfg& cfg_;
    const std::vector<Rdna2Inst>& ins_;
    const RegionPolicy& policy_;
    const std::unordered_set<uint32_t> no_linearized_;
    const std::unordered_set<uint32_t>& linearized_;
    uint32_t end_pc_ = UINT32_MAX;
    RegionTree tree_;
    // Blocks whose terminating branch belongs to a structure and is not an if: else-arm jumps,
    // back edges and canonical loop exits.
    std::unordered_set<uint32_t> claimed_;
    std::unordered_set<uint32_t> latches_;   // blocks ending in a loop's back edge
    std::vector<FoundLoop> loops_;
    std::vector<DivLoop> div_loops_;   // the divergent loops, for classify_forward_if_branch
    CountedLoop counted_;   // the counted loop, when the counted route was taken
    std::vector<uint32_t> loop_at_header_, loop_at_break_;   // per block: index into loops_
    std::unordered_set<uint32_t> open_headers_;   // loops whose body is being built
    std::string failure_;
};

}   // namespace prosper::gpu::ir::cfg::detail
