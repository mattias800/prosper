#include "gpu/recompiler/rdna2_packet_raw_masks.hpp"

namespace prosper::gpu {
bool emit_cfg_mask_ffbh_phase(SpirvCompute& b, uint32_t mask_ffbh_pending_var,
                              uint32_t mask_ffbh_mask_var, uint32_t mask_ffbh_write_var,
                              uint32_t mask_ffbh_event_var, uint32_t mask_ffbh_half_var,
                              uint32_t mask_ffbh_dst_var, uint32_t mbcnt_lane,
                              uint32_t mbcnt_wave_index, uint32_t wave_result_base,
                              const std::set<int>& portable_mask_ffbh_dsts,
                              const std::map<int, uint32_t>& vv,
                              const std::map<std::pair<int, int>, uint32_t>& lv,
                              const std::map<std::pair<int, int>, uint32_t>& lmv) {
    const uint32_t zero = b.uconst(0), no = b.bfalse();
    // Portable Wave64 saved-mask FFBH phase. Each publishing lane contributes its one predicate bit
    // with a static-event tag. Lane zero assembles the selected architectural 32-bit half in LDS,
    // after which every lane applies the ordinary V_FFBH_U32 semantics and predicates the VGPR
    // write by its own EXEC. This deliberately does not use a host subgroup ballot: the portable
    // route has no exact-width contract, and a narrower ballot would silently lose guest lanes.
    const uint32_t mask_ffbh_pending = b.load_function(b.t_bool, mask_ffbh_pending_var);
    const uint32_t mask_ffbh_mask = b.load_function(b.t_bool, mask_ffbh_mask_var);
    const uint32_t mask_ffbh_tag = b.load_function(b.t_u32, mask_ffbh_event_var);
    const uint32_t mask_ffbh_encoded =
        b.sel(mask_ffbh_pending,
              b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, mask_ffbh_tag, b.uconst(1)),
                     b.sel(mask_ffbh_mask, b.uconst(1), zero)),
              zero);
    b.cfg_scratch_store(b.linear_localid, mask_ffbh_encoded);
    b.barrier();

    const uint32_t mask_ffbh_leader = b.id(), mask_ffbh_assembled = b.id();
    const uint32_t mask_ffbh_is_leader =
        b.land(mask_ffbh_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    b.emit_selmerge(mask_ffbh_assembled);
    b.emit_condbranch(mask_ffbh_is_leader, mask_ffbh_leader, mask_ffbh_assembled);
    b.emit_label(mask_ffbh_leader);
    const uint32_t mask_ffbh_wave_base = b.ibin(Op_ShiftLeftLogical, mbcnt_wave_index, b.uconst(6));
    const uint32_t mask_ffbh_half = b.load_function(b.t_u32, mask_ffbh_half_var);
    uint32_t mask_ffbh_word = zero;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        const uint32_t candidate_lane = b.ibin(
            Op_IAdd, b.uconst(bit), b.ibin(Op_ShiftLeftLogical, mask_ffbh_half, b.uconst(5)));
        const uint32_t candidate_index = b.ibin(Op_IAdd, mask_ffbh_wave_base, candidate_lane);
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(Op_IEqual, candidate_tag, mask_ffbh_tag);
        include = b.land(include, b.ucmp(Op_ULessThan, candidate_index, b.uconst(b.local_count)));
        const uint32_t candidate_bit = b.ibin(Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t positioned = b.ibin(Op_ShiftLeftLogical, candidate_bit, b.uconst(bit));
        mask_ffbh_word = b.ibin(Op_BitwiseOr, mask_ffbh_word, b.sel(include, positioned, zero));
    }
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index),
                        mask_ffbh_word);
    b.emit_branch(mask_ffbh_assembled);
    b.emit_label(mask_ffbh_assembled);
    b.barrier();

    const uint32_t mask_ffbh_result = b.ffbh_u32(
        b.cfg_scratch_load(b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index)));
    const uint32_t mask_ffbh_dst = b.load_function(b.t_u32, mask_ffbh_dst_var);
    const uint32_t mask_ffbh_write =
        b.land(mask_ffbh_pending, b.load_function(b.t_bool, mask_ffbh_write_var));
    for (int reg : portable_mask_ffbh_dsts) {
        const auto destination = vv.find(reg);
        if (destination == vv.end()) return false;
        const uint32_t selected =
            b.land(mask_ffbh_write,
                   b.ucmp(Op_IEqual, mask_ffbh_dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, destination->second);
        b.store_function(destination->second, b.sel(selected, mask_ffbh_result, old));
    }
    // As for every ordinary VALU destination, the physical write ends scalar-spill aliases even
    // where EXEC suppresses this lane's data update.
    for (const auto& kv : lv) {
        const uint32_t selected =
            b.land(mask_ffbh_pending, b.ucmp(Op_IEqual, mask_ffbh_dst,
                                             b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected =
            b.land(mask_ffbh_pending, b.ucmp(Op_IEqual, mask_ffbh_dst,
                                             b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    return true;
}
}   // namespace prosper::gpu
