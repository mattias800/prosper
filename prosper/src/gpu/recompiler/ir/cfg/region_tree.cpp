// region_tree.cpp -- see region_tree.hpp and AGENTS.md (the RegionPolicy rules).
//
// The builder walks block ranges in pc order. A conditional branch that the policy admits opens an
// if whose arms are the block ranges its CFG edges delimit; each arm is built recursively over its
// own range. Any refusal makes the whole program one Unstructured region, which is what the
// emitter does today: detect_forward_ifs refuses the program and the CFG dispatcher, which is
// always whole-program, takes it (plan step 11 makes Unstructured local).
#include "gpu/recompiler/ir/cfg/region_tree.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"

#include <algorithm>

namespace prosper::gpu::ir::cfg {

namespace {

using Kind = Region::Kind;

BranchCondition condition_of(const Rdna2Inst& in, bool wave_vote) {
    BranchCondition c;
    switch (in.opcode) {
        case 0x04: c = {BranchCondition::Flag::Scc, true}; break;
        case 0x05: c = {BranchCondition::Flag::Scc, false}; break;
        case 0x06: c = {BranchCondition::Flag::Vcc, true}; break;
        case 0x07: c = {BranchCondition::Flag::Vcc, false}; break;
        case 0x08: c = {BranchCondition::Flag::Exec, true}; break;
        default: break;   // execnz is never an if (classify_forward_if_branch)
    }
    c.wave_vote = wave_vote;
    return c;
}

class Builder {
public:
    Builder(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins, const RegionPolicy& policy)
        : cfg_(cfg), ins_(ins), policy_(policy) {
        for (const auto& in : ins)
            if (in.is_end) {
                end_pc_ = in.pc;
                break;
            }
    }

    RegionTree build() {
        const uint32_t n = static_cast<uint32_t>(cfg_.blocks.size());
        if (n) tree_.root = seq(0, n);
        if (!failure_.empty() || !n) {
            tree_.regions.clear();
            Region whole;
            whole.kind = Kind::Unstructured;
            whole.header = n ? 0 : kNone;
            whole.first_block = 0;
            whole.end_block = n;
            whole.reason = n ? failure_ : "empty program";
            tree_.regions.push_back(whole);
            tree_.root = 0;
        }
        return std::move(tree_);
    }

private:
    const Rdna2Inst& last_of(uint32_t block) const { return ins_[cfg_.blocks[block].last]; }
    uint32_t range_end_pc(uint32_t end) const {
        return end < cfg_.blocks.size() ? cfg_.blocks[end].pc : UINT32_MAX;
    }
    bool fail(std::string why) {
        if (failure_.empty()) failure_ = std::move(why);
        return false;
    }
    uint32_t push(Region r) {
        tree_.regions.push_back(std::move(r));
        return static_cast<uint32_t>(tree_.regions.size() - 1);
    }
    uint32_t leaf(uint32_t b) {
        Region r;
        r.kind = Kind::Block;
        r.header = b;
        r.first_block = b;
        r.end_block = b + 1;
        return push(std::move(r));
    }

    // The blocks [begin, end) as a Seq. kNone when the range is empty or a refusal was recorded.
    uint32_t seq(uint32_t begin, uint32_t end) {
        if (begin >= end || !failure_.empty()) return kNone;
        Region r;
        r.kind = Kind::Seq;
        r.header = begin;
        r.first_block = begin;
        r.end_block = end;
        for (uint32_t b = begin; b < end && failure_.empty();) {
            uint32_t next = b + 1;
            const uint32_t part = step(b, end, &next);
            if (part != kNone) r.children.push_back(part);
            b = next;
        }
        if (!failure_.empty()) return kNone;
        return push(std::move(r));
    }

    // The region starting at block b inside a range ending at `end`; sets *next past it.
    uint32_t step(uint32_t b, uint32_t end, uint32_t* next) {
        const GuestBlock& block = cfg_.blocks[b];
        switch (block.term) {
            case Terminator::Fallthrough:
            case Terminator::End:
            // The structured emitter treats s_trap as an ordinary instruction (emit_alu ends the
            // invocation); it does not change the region structure.
            case Terminator::Trap: return leaf(b);
            case Terminator::IndirectUnknown:
                // An indirect jump that never returns (s_setpc, s_rfe) leaves the program, like
                // s_endpgm: the merged NGG ES prolog ends in `s_setpc_b64` into its GS part, and
                // what follows it is padding or data. A call returns to the next instruction
                // after code the CFG cannot see, so it is refused.
                if (block.succ[0] == kNone) return leaf(b);
                fail("indirect or call terminator pc=" + std::to_string(last_of(b).pc));
                return kNone;
            case Terminator::Branch:
                if (claimed_jumps_.count(b)) return leaf(b);
                fail("unclaimed s_branch pc=" + std::to_string(last_of(b).pc));
                return kNone;
            default: return conditional(b, end, next);
        }
    }

    uint32_t conditional(uint32_t b, uint32_t end, uint32_t* next) {
        const Rdna2Inst& br = last_of(b);
        ForwardIfRefusal why;
        const ForwardIfBranchClass cls =
            classify_forward_if_branch(br, policy_.allow_vcc, policy_.compute_wave_branches,
                                       policy_.linearized, /*loops*/ nullptr, &why);
        if (cls == ForwardIfBranchClass::Claimed) return leaf(b);   // linearized: not an if
        if (cls != ForwardIfBranchClass::If && cls != ForwardIfBranchClass::UniformVccIf) {
            fail(why.text);
            return kNone;
        }
        uint32_t target = 0;
        bool early = false;
        if (!resolve_forward_if_target(br, end_pc_, policy_.code, policy_.dwords, &target, &early,
                                       &why)) {
            fail(why.text);
            return kNone;
        }
        Region r;
        r.header = b;
        r.first_block = b;
        r.condition = condition_of(br, cls == ForwardIfBranchClass::UniformVccIf);
        if (early || target == end_pc_) {
            if (!early_out(b, end, &r)) return kNone;
        } else if (!forward_if(b, end, target, &r)) {
            return kNone;
        }
        *next = r.end_block;
        if (!admit(br, target, r)) return kNone;
        return push(std::move(r));
    }

    // An if whose then-arm runs to s_endpgm. Its range must reach the end of the program, or the
    // arm would leave the enclosing region (the detector's nesting check).
    bool early_out(uint32_t b, uint32_t end, Region* r) {
        if (range_end_pc(end) < end_pc_)
            return fail("early-out pc=" + std::to_string(last_of(b).pc) + " leaves its region");
        r->kind = Kind::IfThen;
        r->end_block = end;
        r->children = {seq(b + 1, end)};
        return failure_.empty();
    }

    bool forward_if(uint32_t b, uint32_t end, uint32_t target, Region* r) {
        const std::string at = "pc=" + std::to_string(last_of(b).pc);
        const uint32_t taken = cfg_.block_at_pc(target);
        if (taken == kNone) return fail("target " + at + " is not an instruction boundary");
        if (taken > end) return fail("if " + at + " leaves its region");
        r->taken = taken;
        const uint32_t jump = taken - 1;   // the block before the target ends the then-arm
        if (jump > b && cfg_.blocks[jump].term == Terminator::Branch)
            return if_else(b, end, jump, r);
        r->kind = Kind::IfThen;
        r->merge = taken;
        r->end_block = taken;
        r->children = {seq(b + 1, taken)};
        return failure_.empty();
    }

    // The then-arm [b+1, taken) ends in `s_branch merge`; the else-arm is [taken, merge), clamped
    // to the enclosing range for the shared-outer-merge cascade.
    bool if_else(uint32_t b, uint32_t end, uint32_t jump, Region* r) {
        const Rdna2Inst& sb = last_of(jump);
        const uint32_t target_pc = cfg_.blocks[r->taken].pc;
        if (sb.simm16 <= 0) return fail("backward else branch pc=" + std::to_string(sb.pc));
        const uint32_t merge_pc = branch_target(sb);
        if (merge_pc <= target_pc || merge_pc > end_pc_)
            return fail("invalid else merge branch pc=" + std::to_string(sb.pc));
        const uint32_t merge = cfg_.block_at_pc(merge_pc);
        if (merge == kNone)
            return fail("else merge pc=" + std::to_string(merge_pc) + " is not a boundary");
        claimed_jumps_.insert(jump);
        const uint32_t else_end = std::min(merge, end);
        r->kind = Kind::IfElse;
        r->merge = merge;
        r->else_jump = jump;
        r->end_block = else_end;
        const uint32_t then_arm = seq(b + 1, r->taken);
        const uint32_t else_arm = seq(r->taken, else_end);
        r->children = {then_arm, else_arm};
        return failure_.empty();
    }

    // The per-branch rules that need the region's extent: the compute EXECZ region, workgroup
    // uniformity and the per-wave VCC region. `region_end` is the detector's raw pc (the merge of
    // an if/else, else the target), not the clamped range.
    bool admit(const Rdna2Inst& br, uint32_t target, Region& r) {
        const uint32_t region_end = r.kind == Kind::IfElse ? cfg_.blocks[r.merge].pc : target;
        ForwardIfRefusal why;
        const bool exec = r.condition.flag == BranchCondition::Flag::Exec;
        if (!policy_.allow_vcc && policy_.compute_wave_branches && exec &&
            !compute_execz_region_admissible(ins_, br.pc, region_end, &why))
            return fail(why.text);
        const ForwardIfWorkgroupUniform uniform = forward_if_workgroup_uniform(
            ins_, br, r.condition.wave_vote, policy_.compute_wave_branches);
        r.condition.uniform_workgroup = uniform != ForwardIfWorkgroupUniform::No;
        if (r.condition.wave_vote && !r.condition.uniform_workgroup &&
            !wave_vcc_region_admissible(ins_, br.pc, region_end, &why))
            return fail(why.text);
        return true;
    }

    const GuestCfg& cfg_;
    const std::vector<Rdna2Inst>& ins_;
    const RegionPolicy& policy_;
    uint32_t end_pc_ = UINT32_MAX;
    RegionTree tree_;
    std::unordered_set<uint32_t> claimed_jumps_;
    std::string failure_;
};

}   // namespace

bool RegionTree::has_unstructured() const {
    return std::any_of(regions.begin(), regions.end(),
                       [](const Region& r) { return r.kind == Kind::Unstructured; });
}

RegionTree build_region_tree(const GuestCfg& cfg, const std::vector<Rdna2Inst>& ins,
                             const RegionPolicy& policy) {
    return Builder(cfg, ins, policy).build();
}

}   // namespace prosper::gpu::ir::cfg
