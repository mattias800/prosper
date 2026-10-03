#include "gpu/recompiler/rdna2_packet_raw_masks.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <iterator>

namespace prosper::gpu {
PacketRawMasks::PacketRawMasks(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins) {
    if (!b.is_fragment_packet() || b.wave_size != 64 || b.native_subgroup_size ||
        b.local_count != 64)
        return;
    for (const auto& in : ins)
        if (in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == 0x08 || in.opcode == kSop1OpcodeAndSaveexecB64))
            scc_sites.insert(in.pc);
    std::set<int> possible;
    for (const auto& in : ins)
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (base >= 0 && base < 105 && width == 2 && scalar_write_is_b64_mask(in, base))
                possible.insert(base);
        });
    for (int root : possible)
        for (const auto& in : ins) {
            // A physical half read or partial replacement requires actual words, not a Bool alias.
            for_each_scalar_write(in, [&](int first, uint32_t width) {
                if (first <= root + 1 && first + static_cast<int>(width) > root &&
                    !(first <= root && first + static_cast<int>(width) >= root + 2))
                    roots.insert(root);
            });
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const auto& operand = in.src[source];
                if (operand.kind != OperandKind::SGPR) continue;
                const uint32_t width = scalar_alu_source_words(in, source);
                if (operand.value <= root + 1 && operand.value + static_cast<int>(width) > root &&
                    width != 2)
                    roots.insert(root);
            }
        }
    for (const auto& in : ins)
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (width == 2 && roots.contains(base) && scalar_write_is_b64_mask(in, base))
                sites.emplace(in.pc, base);
        });
}
uint32_t cfg_b64_mask_scc_value(const Rdna2Inst& in, const RegState& state, bool owned_packet) {
    if ((owned_packet && in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeAndSaveexecB64) ||
        in.dst.value == 126 || in.dst.value == 127)
        return state.exec;
    if (in.dst.value == 106 || in.dst.value == 107) return state.vcc;
    const auto saved = state.sreg_bool.find(in.dst.value);
    return saved != state.sreg_bool.end() ? saved->second : 0;
}
void PacketRawMasks::begin(SpirvCompute& b, uint32_t base) {
    if (sites.empty()) return;
    result_base = base;
    uint32_t bool_ptr = 0, word_ptr = 0;
    pending_var = b.function_var(b.t_bool, bool_ptr);
    mask_var = b.function_var(b.t_bool, bool_ptr);
    deferred_var = b.function_var(b.t_bool, bool_ptr);
    event_var = b.function_var(b.t_u32, word_ptr);
    dst_var = b.function_var(b.t_u32, word_ptr);
    b.store_function(mask_var, b.bfalse());
    b.store_function(deferred_var, b.bfalse());
    b.store_function(event_var, b.uconst(0));
    b.store_function(dst_var, b.uconst(0));
    reset(b);
}
void PacketRawMasks::reset(SpirvCompute& b) {
    if (pending_var) b.store_function(pending_var, b.bfalse());
}
bool PacketRawMasks::stage(SpirvCompute& b, const RegState& state, const Rdna2Inst& in,
                           bool deferred) {
    const auto site = sites.find(in.pc);
    if (site == sites.end()) return false;
    const auto mask = state.sreg_bool.find(site->second);
    // Syntax alone does not grant a mask. Numeric writers retain their existing lowering.
    if (!deferred && mask == state.sreg_bool.end()) return false;
    b.store_function(pending_var, b.btrue());
    b.store_function(mask_var, deferred ? b.bfalse() : mask->second);
    b.store_function(deferred_var, deferred ? b.btrue() : b.bfalse());
    // Nonzero static ordinal is independent of guest PC magnitude and reused only after barriers.
    b.store_function(event_var,
                     b.uconst(static_cast<uint32_t>(std::distance(sites.begin(), site)) + 1));
    b.store_function(dst_var, b.uconst(static_cast<uint32_t>(site->second)));
    return true;
}
void PacketRawMasks::expire(RegState& state, const Rdna2Inst& in) const {
    for_each_scalar_write(in, [&](int first, uint32_t width) {
        for (int root : roots) {
            if (first > root + 1 || first + static_cast<int>(width) <= root) continue;
            // A complete mask producer publishes its NEW Bool. A one-word writer cannot preserve
            // the old complete alias, even when it addresses only the high physical word.
            if (first == root && width == 2 && scalar_write_is_b64_mask(in, root)) continue;
            state.sreg_bool.erase(root);
            state.sreg_bool_narrowed.erase(root);
        }
    });
}
void PacketRawMasks::phase(SpirvCompute& b, const std::map<int, uint32_t>& scalars,
                           const std::map<int, uint32_t>& masks) {
    if (!pending_var) return;
    const auto zero = b.uconst(0);
    const auto pending = b.load_function(b.t_bool, pending_var);
    const auto tag = b.load_function(b.t_u32, event_var);
    const auto dst = b.load_function(b.t_u32, dst_var);
    auto mask = b.load_function(b.t_bool, mask_var);
    // WQM runs in an earlier uniform service phase, so its newly published predicate is loaded
    // here rather than guessing its value at issue time. Ordinary producers retain the SSA bit.
    auto deferred_mask = b.bfalse();
    for (int root : roots)
        if (masks.contains(root))
            deferred_mask = b.bsel(b.ucmp(Op_IEqual, dst, b.uconst(root)),
                                   b.load_function(b.t_bool, masks.at(root)), deferred_mask);
    mask = b.bsel(b.load_function(b.t_bool, deferred_var), deferred_mask, mask);
    b.cfg_scratch_store(b.linear_localid,
                        b.sel(pending,
                              b.ibin(Op_BitwiseOr, b.ibin(Op_ShiftLeftLogical, tag, b.uconst(1)),
                                     b.sel(mask, b.uconst(1), zero)),
                              zero));
    b.barrier();
    // All64 workers reach the phase irrespective of EXEC/export/helper state. The packet has one
    // scalar PC per logical wave, and a complete producer is scalar-unconditional. Every physical
    // mask bit therefore has an actual same-event publisher; no absent participant is padded in.
    // Only the logical wave leader assembles the pair: O(64) LDS reads, not O(64*64). The two
    // result slots are private and disjoint from the reused input plane and definedness metadata.
    const auto leader = b.id(), assembled = b.id();
    const auto is_leader = b.land(pending, b.ucmp(Op_IEqual, b.linear_localid, zero));
    b.emit_selmerge(assembled);
    b.emit_condbranch(is_leader, leader, assembled);
    b.emit_label(leader);
    uint32_t words[2]{zero, zero};
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto candidate = b.cfg_scratch_load(b.uconst(lane));
        const auto same =
            b.ucmp(Op_IEqual, b.ibin(Op_ShiftRightLogical, candidate, b.uconst(1)), tag);
        const auto positioned =
            b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, candidate, b.uconst(1)),
                   b.uconst(lane & 31));
        words[lane / 32] = b.ibin(Op_BitwiseOr, words[lane / 32], b.sel(same, positioned, zero));
    }
    for (uint32_t half = 0; half < 2; ++half)
        b.cfg_scratch_store(b.uconst(result_base + half), words[half]);
    b.emit_branch(assembled);
    b.emit_label(assembled);
    b.barrier();
    for (uint32_t half = 0; half < 2; ++half)
        // Nonparticipants load the already published harmless input slot, never an uninitialized
        // result. That does not define any guest word: their destination selector remains false.
        words[half] = b.cfg_scratch_load(b.sel(pending, b.uconst(result_base + half), zero));
    for (int root : roots) {
        const auto selected = b.land(pending, b.ucmp(Op_IEqual, dst, b.uconst(root)));
        for (uint32_t half = 0; half < 2; ++half) {
            const auto var = scalars.at(root + static_cast<int>(half));
            b.store_function(var, b.sel(selected, words[half], b.load_function(b.t_u32, var)));
        }
    }
    // Reuse of the shared service plane waits for every participant's reads, including ended peers.
    b.barrier();
}
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
