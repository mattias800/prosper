#include "gpu/recompiler/rdna2_dpp_quad_or.hpp"

#include "gpu/recompiler/rdna2_compute_quad_init.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

namespace prosper::gpu {
bool ComputeDppQuadOrService::prepare(SpirvCompute& b, const RegState& entry,
                                      const std::vector<Rdna2Inst>& ins, const uint32_t* code,
                                      size_t dwords, uint32_t initial_active) {
    if (!std::any_of(ins.begin(), ins.end(), compute_dpp_quad_or)) return true;
    if (!b.guest_compute_complete_groups) {
        rejection_ = "dpp-quad-or-entry-launch-unproven";
        return false;
    }
    const auto proof = proven_compute_quad_or_sources(b, entry, ins, code, dwords, initial_active);
    for (const auto& in : ins) {
        if (!compute_dpp_quad_or(in)) continue;
        if (!proof.contains(in.pc)) return false;
        events_.emplace(in.pc, static_cast<uint32_t>(events_.size() + 1));
        destinations_.insert(in.dst.value);
    }
    return true;
}
void ComputeDppQuadOrService::split(std::set<uint32_t>& starts,
                                    const std::vector<Rdna2Inst>& ins) const {
    for (const auto& in : ins)
        if (contains(in.pc)) {
            starts.insert(in.pc);
            starts.insert(in.pc + in.len_dwords);
        }
}
void ComputeDppQuadOrService::begin(SpirvCompute& b, uint32_t ptr_bool, uint32_t ptr_u32) {
    if (!present()) return;
    for (size_t i = 0; i < vars_.size(); ++i)
        vars_[i] =
            b.function_var(i <= Active ? b.t_bool : b.t_u32, i <= Active ? ptr_bool : ptr_u32);
}
void ComputeDppQuadOrService::reset(SpirvCompute& b) const {
    if (!present()) return;
    for (size_t i = 0; i < vars_.size(); ++i)
        b.store_function(vars_[i], i <= Active ? b.bfalse() : b.uconst(0));
}
bool ComputeDppQuadOrService::publish(SpirvCompute& b, RegState& rs, const Rdna2Inst& in) const {
    const auto event = events_.find(in.pc);
    if (event == events_.end() || !compute_dpp_quad_or(in)) return false;
    const auto a = rs.vreg.find(in.src[0].value), c = rs.vreg.find(in.src[1].value);
    if (a == rs.vreg.end() || c == rs.vreg.end() || !rs.exec) return false;
    const auto marks = vector_source_marks(rs, in);
    if (marks.placeholder || marks.hard || !marks.blanket_roots.empty() ||
        rs.vgpr_lane_slots.contains(in.src[0].value) ||
        rs.vgpr_lane_slots.contains(in.src[1].value) ||
        rs.vgpr_lane_mask_slots.contains(in.src[0].value) ||
        rs.vgpr_lane_mask_slots.contains(in.src[1].value))
        return false;
    b.store_function(vars_[Pending], b.btrue());
    b.store_function(vars_[Active], rs.exec);
    b.store_function(vars_[Source0], a->second);
    b.store_function(vars_[Source1], c->second);
    b.store_function(vars_[Control], b.uconst(in.dpp_ctrl));
    b.store_function(vars_[Destination], b.uconst(static_cast<uint32_t>(in.dst.value)));
    b.store_function(vars_[Event], b.uconst(event->second));
    record_vector_word_marks(rs, in, marks);
    return true;
}
bool ComputeDppQuadOrService::emit_phase(SpirvCompute& b, uint32_t value_base,
                                         uint32_t metadata_base, const std::map<int, uint32_t>& vv,
                                         const std::map<std::pair<int, int>, uint32_t>& lv,
                                         const std::map<std::pair<int, int>, uint32_t>& lmv) const {
    if (!present()) return true;
    const uint32_t zero = b.uconst(0);
    const auto load = [&](Variable v) {
        return b.load_function(v <= Active ? b.t_bool : b.t_u32, vars_[v]);
    };
    const uint32_t pending = load(Pending), active = load(Active), event = load(Event);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(value_base), b.linear_localid), load(Source0));
    const uint32_t metadata =
        b.sel(pending,
              b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, event, b.uconst(1)),
                     b.sel(active, b.uconst(1), zero)),
              zero);
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(metadata_base), b.linear_localid), metadata);
    b.barrier();
    // RDNA2 QUAD_PERM selects one lane in this guest logical quad. Vulkan subgroup packing is
    // irrelevant: linear local order owns the guest topology, and every worker reaches this phase.
    const uint32_t lane = b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(3));
    const uint32_t shift = b.ibin(Op_ShiftLeftLogical, lane, b.uconst(1));
    const uint32_t selected =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, load(Control), shift), b.uconst(3));
    const uint32_t peer =
        b.ibin(Op_BitwiseOr, b.ibin(Op_BitwiseAnd, b.linear_localid, b.uconst(~3u)), selected);
    const uint32_t in_bounds = b.ucmp(Op_ULessThan, peer, b.uconst(b.local_count));
    const uint32_t safe_peer = b.sel(in_bounds, peer, b.linear_localid);
    const uint32_t peer_metadata =
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(metadata_base), safe_peer));
    const uint32_t same_event =
        b.ucmp(Op_IEqual, b.ibin(Op_ShiftRightLogical, peer_metadata, b.uconst(1)), event);
    const uint32_t peer_active =
        b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, peer_metadata, b.uconst(1)), zero);
    const uint32_t valid = b.land(in_bounds, b.land(peer_active, same_event));
    const uint32_t payload = b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(value_base), safe_peer));
    // CONFIDENCE: HIGH. AMD RDNA2 ISA 70648, section 13.3.9 tables 90/91: FI0 reads zero
    // for an inactive peer; BC1 writes the result. Missing final-quad workers are inactive
    // guest lanes, independently bounds-checked here. SRC1 stays local.
    const uint32_t result = b.ibin(Op_BitwiseOr, b.sel(valid, payload, zero), load(Source1));
    const uint32_t dst = load(Destination), write = b.land(pending, active);
    for (int reg : destinations_) {
        const auto value = vv.find(reg);
        if (value == vv.end()) return false;
        const uint32_t enabled =
            b.land(write, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(reg))));
        b.store_function(value->second,
                         b.sel(enabled, result, b.load_function(b.t_u32, value->second)));
    }
    for (const auto& [slot, value] : lv)
        if (destinations_.contains(slot.first)) {
            const uint32_t enabled = b.land(
                pending, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(slot.first))));
            b.store_function(value, b.sel(enabled, zero, b.load_function(b.t_u32, value)));
        }
    for (const auto& [slot, value] : lmv)
        if (destinations_.contains(slot.first)) {
            const uint32_t enabled = b.land(
                pending, b.ucmp(Op_IEqual, dst, b.uconst(static_cast<uint32_t>(slot.first))));
            b.store_function(value, b.bsel(enabled, b.bfalse(), b.load_function(b.t_bool, value)));
        }
    b.barrier();
    return true;
}
}   // namespace prosper::gpu
