// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void SpirvCompute::mark_fragment_wave_vote_value(uint32_t value) {
        std::vector<uint32_t> pending{value};
        while (!pending.empty()) {
            const uint32_t current = pending.back();
            pending.pop_back();
            if (!fragment_wave_vote_values.insert(current).second) continue;
            if (fragment_wave_vote_scalar_consumers.contains(current))
                fragment_wave_reasons |= kFragmentWaveReasonScalarReduce;
            const auto dependent = fragment_wave_vote_dependents.find(current);
            if (dependent != fragment_wave_vote_dependents.end())
                pending.insert(pending.end(), dependent->second.begin(), dependent->second.end());
        }
    }

void SpirvCompute::add_fragment_wave_vote_dependency(uint32_t result, uint32_t source) {
        fragment_wave_vote_dependents[source].push_back(result);
        if (is_fragment_wave_vote_value(source)) mark_fragment_wave_vote_value(result);
    }

void SpirvCompute::propagate_fragment_wave_vote(uint32_t result, uint32_t a, uint32_t b_) {
        add_fragment_wave_vote_dependency(result, a);
        add_fragment_wave_vote_dependency(result, b_);
    }

void SpirvCompute::mark_fragment_wave_scalar_use(uint32_t value) {
        fragment_wave_vote_scalar_consumers.insert(value);
        if (is_fragment_wave_vote_value(value))
            fragment_wave_reasons |= kFragmentWaveReasonScalarReduce;
    }

uint32_t SpirvCompute::subgroup_local_id() {
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!v_subgroup_localid) {
            if (!t_ptr_in_u32) {
                t_ptr_in_u32 = id();
                put(types, Op_TypePointer, {t_ptr_in_u32, SC_Input, t_u32});
            }
            v_subgroup_localid = id();
            put(types, Op_Variable, {t_ptr_in_u32, v_subgroup_localid, SC_Input});
            put(deco, Op_Decorate,
                {v_subgroup_localid, Dec_BuiltIn, BI_SubgroupLocalInvocationId});
            put(deco, Op_Decorate, {v_subgroup_localid, Dec_Flat});
            iface.push_back(v_subgroup_localid);
        }
        // Fragment lane ids model the guest RDNA wave, not the implementation's default subgroup.
        // Record that exact-width contract as non-semantic module metadata in finish().
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonLaneId;
        uint32_t lane = id();
        put(code, Op_Load, {t_u32, lane, v_subgroup_localid});
        return lane;
    }

uint32_t SpirvCompute::fragment_wave_any(uint32_t active_bit) {
        if (!is_fragment) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_vote) {
            put(caps, Op_Capability, {Cap_GroupNonUniformVote});
            declared_subgroup_vote = true;
        }
        fragment_required_subgroup_size = wave_size;
        fragment_wave_reasons |= kFragmentWaveReasonWaveAny;
        uint32_t result = id();
        put(code, Op_GroupNonUniformAny,
            {t_bool, result, uconst(Scope_Subgroup), active_bit});
        mark_fragment_wave_vote_value(result);
        return result;
    }

uint32_t SpirvCompute::fragment_wave_ballot_half(uint32_t mask_bit, uint32_t half) {
        if (!is_fragment || !mask_bit) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_ballot) {
            put(caps, Op_Capability, {Cap_GroupNonUniformBallot});
            declared_subgroup_ballot = true;
        }
        fragment_required_subgroup_size = wave_size;
        // Ballot, not vote: the reason bit must say which, because only one of the two can ever be
        // relaxed to a narrower subgroup and the skip diagnostic is where that question gets asked
        // (#2441). This records the same width as before -- the gate keys on
        // fragment_required_subgroup_size, not on the reason -- so behaviour is unchanged.
        fragment_wave_reasons |= kFragmentWaveReasonWaveBallot;
        const uint32_t ballot = id();
        put(code, Op_GroupNonUniformBallot,
            {t_v4u(), ballot, uconst(Scope_Subgroup), mask_bit});
        const uint32_t result = id();
        putv(code, Op_CompositeExtract, {t_u32, result, ballot, half});
        return result;
    }

uint32_t SpirvCompute::subgroup_id() {
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!v_subgroupid) {
            if (!t_ptr_in_u32) {
                t_ptr_in_u32 = id();
                put(types, Op_TypePointer, {t_ptr_in_u32, SC_Input, t_u32});
            }
            v_subgroupid = id();
            put(types, Op_Variable, {t_ptr_in_u32, v_subgroupid, SC_Input});
            put(deco, Op_Decorate, {v_subgroupid, Dec_BuiltIn, BI_SubgroupId});
            put(deco, Op_Decorate, {v_subgroupid, Dec_Flat});
            iface.push_back(v_subgroupid);
        }
        uint32_t subgroup = id();
        put(code, Op_Load, {t_u32, subgroup, v_subgroupid});
        return subgroup;
    }

uint32_t SpirvCompute::fragment_mbcnt(uint32_t mask_bit, uint32_t acc_bits, bool lo) {
        if (!is_fragment) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t in_half = lo
            ? ucmp(Op_ULessThan, lane, uconst(32))
            : ucmp(Op_UGreaterThanEqual, lane, uconst(32));
        // Helper invocations execute subgroup operations under WQM but are not guest lanes and
        // cannot own GDS allocation slots. Excluding them here keeps MBCNT's prefix exactly aligned
        // with fragment GDS append/consume's non-helper population.
        const uint32_t guest_lane = land(mask_bit, logical_not(helper_invocation()));
        const uint32_t selected = sel(land(guest_lane, in_half), uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, selected});
        return ibin(Op_IAdd, acc_bits, prefix);
    }

uint32_t SpirvCompute::native_wave_any(uint32_t value) {
        if (!native_subgroup_size) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_vote) {
            put(caps, Op_Capability, {Cap_GroupNonUniformVote});
            declared_subgroup_vote = true;
        }
        uint32_t result = id();
        put(code, Op_GroupNonUniformAny,
            {t_bool, result, uconst(Scope_Subgroup), value});
        return result;
    }

uint32_t SpirvCompute::native_wave_ballot_half(uint32_t mask_bit, uint32_t half) {
        if (!native_subgroup_size || !mask_bit) return 0;
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_ballot) {
            put(caps, Op_Capability, {Cap_GroupNonUniformBallot});
            declared_subgroup_ballot = true;
        }
        const uint32_t ballot = id();
        put(code, Op_GroupNonUniformBallot,
            {t_v4u(), ballot, uconst(Scope_Subgroup), mask_bit});
        const uint32_t result = id();
        putv(code, Op_CompositeExtract, {t_u32, result, ballot, half});
        return result;
    }

uint32_t SpirvCompute::native_wave_first_active(uint32_t mask_bit) {
        if (!native_subgroup_size) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        // The exclusive population count is zero only for the first active lane. Exactly one lane
        // therefore contributes its index to the reduction; an empty mask is distinguished by the
        // accompanying wave vote. This is exact only under the enforced guest-size subgroup
        // contract checked by the caller.
        const uint32_t contribution = sel(mask_bit, uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(mask_bit, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t selected_lane = sel(elected, lane, uconst(0));
        uint32_t first = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, first, uconst(Scope_Subgroup), GroupOp_Reduce, selected_lane});
        return sel(native_wave_any(mask_bit), first, uconst(0xffffffffu));
    }

uint32_t SpirvCompute::native_wave_popcount(uint32_t mask_bit) {
        if (!native_subgroup_size || !mask_bit) return 0;
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t contribution = sel(mask_bit, uconst(1), uconst(0));
        uint32_t result = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, result, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
        return result;
    }

uint32_t SpirvCompute::native_compute_mbcnt(uint32_t mask_bit, uint32_t acc_bits, uint32_t lo) {
        if (!native_subgroup_size) return 0;
        const uint32_t lane = subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t in_half = bsel(
            lo, ucmp(Op_ULessThan, lane, uconst(32)),
            ucmp(Op_UGreaterThanEqual, lane, uconst(32)));
        const uint32_t selected = sel(land(mask_bit, in_half), uconst(1), uconst(0));
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, selected});
        return ibin(Op_IAdd, acc_bits, prefix);
    }

uint32_t SpirvCompute::helper_invocation() {
        if (!v_helper_invocation) {
            t_ptr_in_bool = id();
            put(types, Op_TypePointer, {t_ptr_in_bool, SC_Input, t_bool});
            v_helper_invocation = id();
            put(types, Op_Variable, {t_ptr_in_bool, v_helper_invocation, SC_Input});
            put(deco, Op_Decorate,
                {v_helper_invocation, Dec_BuiltIn, BI_HelperInvocation});
            iface.push_back(v_helper_invocation);
        }
        uint32_t result = id();
        put(code, Op_Load, {t_bool, result, v_helper_invocation});
        return result;
    }

void SpirvCompute::mark_subgroup_min16() {
        // DPP rows contain 16 contiguous lanes. Keep width metadata independent of SPIR-V
        // capabilities: declaring an unused operation merely as a marker over-requires the host and
        // confuses shaders that genuinely use that capability for unrelated work.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonDppRow16;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 16u);
    }

void SpirvCompute::mark_subgroup_min32() {
        // PERMLANEX16 crosses a pair of 16-lane rows.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonPermLane32;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 32u);
    }

void SpirvCompute::mark_subgroup_min64() {
        // V_READLANE_B32 may address every lane of a wave64.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonReadLane64;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 64u);
    }

uint32_t SpirvCompute::subgroup_shuffle(uint32_t value, uint32_t lane) {
        // Every supported native shuffle at least addresses an architectural quad. Wider row/wave
        // operations raise this contract before calling the common helper.
        if (is_fragment) fragment_required_subgroup_size = wave_size,
                         fragment_wave_reasons |= kFragmentWaveReasonShuffle;
        else compute_min_subgroup_size = std::max(compute_min_subgroup_size, 4u);
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_shuffle) {
            put(caps, Op_Capability, {Cap_GroupNonUniformShuffle});
            declared_subgroup_shuffle = true;
        }
        uint32_t result = id();
        put(code, Op_GroupNonUniformShuffle,
            {t_u32, result, uconst(Scope_Subgroup), value, lane});
        return result;
    }

uint32_t SpirvCompute::subgroup_quad_permute(uint32_t value, uint32_t ctrl) {
        const uint32_t lane = subgroup_local_id();
        const uint32_t quad_lane = ibin(Op_BitwiseAnd, lane, uconst(3));
        uint32_t selected = uconst(ctrl & 3u);
        for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
            selected = sel(ucmp(Op_IEqual, quad_lane, uconst(output_lane)),
                           uconst((ctrl >> (2u * output_lane)) & 3u), selected);
        const uint32_t source_lane = ibin(
            Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~3u)), selected);
        return subgroup_shuffle(value, source_lane);
    }

bool SpirvCompute::ds_swizzle_source_lane(uint32_t offset, uint32_t* source_lane) {
        // RDNA2 ISA 12.13.1: the two basic DS_SWIZZLE modes precede rotate/FFT. The quad form
        // (10xx...) carries four two-bit selectors; the group32 form carries AND/OR/XOR masks.
        // Rotate and FFT remain fail-closed until their distinct mappings are implemented.
        if (!source_lane || offset >= 0xc000u) return false;
        const uint32_t lane = subgroup_local_id();
        if (offset & 0x8000u) {
            const uint32_t quad_lane = ibin(Op_BitwiseAnd, lane, uconst(3));
            uint32_t selected = uconst(offset & 3u);
            for (uint32_t output_lane = 1; output_lane < 4; ++output_lane)
                selected = sel(ucmp(Op_IEqual, quad_lane, uconst(output_lane)),
                               uconst((offset >> (2u * output_lane)) & 3u), selected);
            *source_lane = ibin(
                Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~3u)), selected);
            return true;
        }
        mark_subgroup_min32();
        const uint32_t and_mask = uconst(offset & 0x1fu);
        const uint32_t or_mask = uconst((offset >> 5) & 0x1fu);
        const uint32_t xor_mask = uconst((offset >> 10) & 0x1fu);
        const uint32_t lane_in_group = ibin(Op_BitwiseAnd, lane, uconst(0x1fu));
        const uint32_t selected = ibin(
            Op_BitwiseXor,
            ibin(Op_BitwiseOr, ibin(Op_BitwiseAnd, lane_in_group, and_mask), or_mask),
            xor_mask);
        *source_lane = ibin(
            Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~0x1fu)), selected);
        return true;
    }

void SpirvCompute::declare_wave_lds() {
        if (lds_wave) return;
        // Keep one result slot per guest wave separate from the per-lane publication area.  A
        // completed vote can then be consumed while a faster invocation starts publishing the next
        // one, without needing a third workgroup barrier solely to protect the result.
        const uint32_t wave_count = (local_count + wave_size - 1) / wave_size;
        uint32_t t_arr = id();
        put(types, Op_TypeArray,
            {t_arr, t_u32, uconst(std::max(1u, local_count + wave_count))});
        uint32_t t_ptr = id();  put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        lds_wave = id();        put(types, Op_Variable, {t_ptr, lds_wave, SC_Workgroup});
        t_ptr_wg_u32b = id();   put(types, Op_TypePointer, {t_ptr_wg_u32b, SC_Workgroup, t_u32});
    }

uint32_t SpirvCompute::guest_wave_any(uint32_t active_bool) {
        declare_wave_lds();
        const uint32_t zero = uconst(0);
        const uint32_t bit = sel(active_bool, uconst(1), uconst(0));
        uint32_t p = id();
        putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, linear_localid});
        put(code, Op_Store, {p, bit});
        barrier();

        const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
        const uint32_t wave_index = ibin(
            Op_ShiftRightLogical, linear_localid, uconst(wave_shift));
        const uint32_t wave_base = ibin(
            Op_ShiftLeftLogical, wave_index, uconst(wave_shift));
        const uint32_t lane = ibin(
            Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
        const uint32_t leader = id(), reduced = id();
        const uint32_t is_leader = ucmp(Op_IEqual, lane, zero);
        emit_selmerge(reduced);
        emit_condbranch(is_leader, leader, reduced);
        emit_label(leader);
        uint32_t any_word = zero;
        for (uint32_t i = 0; i < wave_size; ++i) {
            const uint32_t idx = ibin(Op_IAdd, wave_base, uconst(i));
            const uint32_t valid = ucmp(Op_ULessThan, idx, uconst(local_count));
            // Keep the memory access in-bounds for a partial final guest wave. The selected value is
            // ignored when idx names a padded lane, but Vulkan must never observe an OOB OpLoad.
            const uint32_t safe_idx = sel(valid, idx, zero);
            uint32_t q = id();
            putv(code, Op_AccessChain, {t_ptr_wg_u32b, q, lds_wave, safe_idx});
            uint32_t value = id();
            put(code, Op_Load, {t_u32, value, q});
            any_word = ibin(Op_BitwiseOr, any_word, sel(valid, value, zero));
        }
        const uint32_t result_index = ibin(Op_IAdd, uconst(local_count), wave_index);
        uint32_t result_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr, lds_wave, result_index});
        put(code, Op_Store, {result_ptr, any_word});
        emit_branch(reduced);
        emit_label(reduced);
        barrier();
        const uint32_t result_index_all = ibin(
            Op_IAdd, uconst(local_count), wave_index);
        uint32_t result_ptr_all = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr_all, lds_wave, result_index_all});
        uint32_t result = id();
        put(code, Op_Load, {t_u32, result, result_ptr_all});
        return ucmp(Op_INotEqual, result, zero);
}

uint32_t SpirvCompute::guest_wave_popcount(uint32_t active_bool) {
    // Portable whole-guest-wave reduction. Every invocation in the guest wave must reach both
    // barriers; callers must establish a uniform control-flow site before executing this module.
    declare_wave_lds();
    const uint32_t zero = uconst(0);
    const uint32_t bit = sel(active_bool, uconst(1), zero);
    uint32_t slot = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, slot, lds_wave, linear_localid});
    put(code, Op_Store, {slot, bit});
    barrier();

    const uint32_t shift = wave_size == 32 ? 5u : 6u;
    const uint32_t wave_index = ibin(Op_ShiftRightLogical, linear_localid, uconst(shift));
    const uint32_t wave_base = ibin(Op_ShiftLeftLogical, wave_index, uconst(shift));
    const uint32_t lane = ibin(Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
    const uint32_t leader = id(), done = id();
    const uint32_t is_leader = ucmp(Op_IEqual, lane, zero);
    emit_selmerge(done);
    emit_condbranch(is_leader, leader, done);
    emit_label(leader);
    uint32_t count = zero;
    for (uint32_t i = 0; i < wave_size; ++i) {
        const uint32_t index = ibin(Op_IAdd, wave_base, uconst(i));
        const uint32_t valid = ucmp(Op_ULessThan, index, uconst(local_count));
        const uint32_t safe_index = sel(valid, index, zero);
        uint32_t ptr = id();
        putv(code, Op_AccessChain, {t_ptr_wg_u32b, ptr, lds_wave, safe_index});
        uint32_t value = id();
        put(code, Op_Load, {t_u32, value, ptr});
        count = ibin(Op_IAdd, count, sel(valid, value, zero));
    }
    const uint32_t result_index = ibin(Op_IAdd, uconst(local_count), wave_index);
    uint32_t result_ptr = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, result_ptr, lds_wave, result_index});
    put(code, Op_Store, {result_ptr, count});
    emit_branch(done);
    emit_label(done);
    barrier();
    const uint32_t read_index = ibin(Op_IAdd, uconst(local_count), wave_index);
    uint32_t read_ptr = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, read_ptr, lds_wave, read_index});
    uint32_t result = id();
    put(code, Op_Load, {t_u32, result, read_ptr});
    return result;
}

uint32_t SpirvCompute::guest_wave_readlane(uint32_t source_value, uint32_t selector) {
        declare_wave_lds();
        uint32_t source_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, source_ptr, lds_wave, linear_localid});
        put(code, Op_Store, {source_ptr, source_value});
        barrier();

        const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
        const uint32_t wave_base = ibin(
            Op_ShiftLeftLogical,
            ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift)),
            uconst(wave_shift));
        const uint32_t lane = ibin(
            Op_BitwiseAnd, selector, uconst(wave_size - 1u));
        const uint32_t index = ibin(Op_IAdd, wave_base, lane);
        const uint32_t zero = uconst(0);
        const uint32_t valid = ucmp(Op_ULessThan, index, uconst(local_count));
        const uint32_t safe_index = sel(valid, index, zero);
        uint32_t result_ptr = id();
        putv(code, Op_AccessChain,
             {t_ptr_wg_u32b, result_ptr, lds_wave, safe_index});
        uint32_t result = id();
        put(code, Op_Load, {t_u32, result, result_ptr});
        barrier();
        return sel(valid, result, zero);
    }

uint32_t SpirvCompute::mbcnt(uint32_t active_bool, uint32_t acc_bits, bool lo) {
        declare_wave_lds();
        uint32_t bit = sel(active_bool, uconst(1), uconst(0));
        { uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, linear_localid}); put(code, Op_Store, {p, bit}); }
        barrier();
        const uint32_t lane = ibin(Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
        const uint32_t wave_index = ibin(Op_ShiftRightLogical, linear_localid,
                                          uconst(wave_size == 32 ? 5u : 6u));
        uint32_t sum = uconst(0);
        const uint32_t first = lo ? 0u : 32u;
        const uint32_t last = std::min(wave_size, lo ? 32u : 64u);
        for (uint32_t i = 0; i < local_count; i++) {
            const uint32_t candidate_lane = i % wave_size;
            if (candidate_lane < first || candidate_lane >= last) continue;
            uint32_t cond = land(ucmp(Op_IEqual, wave_index, uconst(i / wave_size)),
                                 ucmp(Op_ULessThan, uconst(candidate_lane), lane));
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, uconst(i)});
            uint32_t v = id(); put(code, Op_Load, {t_u32, v, p});
            sum = b_iadd(sum, sel(cond, v, uconst(0)));
        }
        barrier();
        return b_iadd(acc_bits, sum);
    }

uint32_t SpirvCompute::native_wave_append(uint32_t lds_idx, uint32_t active_bool,
                                uint32_t consume) {
        if (!native_subgroup_size) return 0;
        (void)subgroup_local_id();
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t contribution = sel(active_bool, uconst(1), uconst(0));
        uint32_t count = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, count, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(active_bool, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t entry = cur_block, leader = id(), merge = id();
        emit_selmerge(merge);
        emit_condbranch(elected, leader, merge);
        emit_label(leader);
        const uint32_t delta = sel(consume, ibin(Op_ISub, uconst(0), count), count);
        const uint32_t leader_old = lds_atomic_rtn(
            Op_AtomicIAdd, lds_idx, delta, false, btrue(), uconst(0));
        const uint32_t leader_end = cur_block;
        emit_branch(merge);
        emit_label(merge);
        const uint32_t local_old = emit_phi_2way(
            t_u32, leader_old, leader_end, uconst(0), entry);
        uint32_t old = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, old, uconst(Scope_Subgroup), GroupOp_Reduce, local_old});
        return old;
    }

uint32_t SpirvCompute::invocation_within_extent(uint32_t threads_x, uint32_t threads_y,
                                      uint32_t threads_z) {
        uint32_t within = ucmp(Op_ULessThan, globalid_comp[0], uconst(threads_x));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[1], uconst(threads_y)));
        within = land(within, ucmp(Op_ULessThan, globalid_comp[2], uconst(threads_z)));
        return within;
    }

void SpirvCompute::guard_invocation_extent(uint32_t threads_x, uint32_t threads_y, uint32_t threads_z) {
        const uint32_t within = invocation_within_extent(threads_x, threads_y, threads_z);
        const uint32_t active = id();
        invocation_guard_merge = id();
        emit_selmerge(invocation_guard_merge);
        emit_condbranch(within, active, invocation_guard_merge);
        emit_label(active);
    }

uint32_t SpirvCompute::vertex_invocation_id() {
        const uint32_t vertex = load_vertex_index();
        if (!vertices_per_instance) return vertex;
        return ibin(Op_IAdd,
                    ibin(Op_IMul, load_instance_index(), uconst(vertices_per_instance)), vertex);
    }

uint32_t SpirvCompute::guest_lane_id() {
        if (is_fragment) return subgroup_local_id();
        if (is_compute) return linear_localid;
        return ibin(Op_BitwiseAnd, vertex_invocation_id(), uconst(wave_size - 1));
    }

uint32_t SpirvCompute::dpp_quad(uint32_t x_bits, uint32_t ctrl) {
        uint32_t fc = id(); put(code, Op_Load, {t_v4f, fc, fragcoord_var()});
        uint32_t fx = id(); put(code, Op_CompositeExtract, {t_f32, fx, fc, 0});
        uint32_t fy = id(); put(code, Op_CompositeExtract, {t_f32, fy, fc, 1});
        uint32_t pxu = id(); put(code, Op_ConvertFToU, {t_u32, pxu, fx});
        uint32_t pyu = id(); put(code, Op_ConvertFToU, {t_u32, pyu, fy});
        uint32_t px = ibin(Op_BitwiseAnd, pxu, uconst(1)), py = ibin(Op_BitwiseAnd, pyu, uconst(1));
        uint32_t lane = ibin(Op_IAdd, px, ibin(Op_ShiftLeftLogical, py, uconst(1)));
        uint32_t tsel = uconst((uint32_t)(ctrl & 3u));                 // QP[0] default; select QP[lane]
        for (uint32_t k = 1; k < 4; k++)
            tsel = sel(ucmp(Op_IEqual, lane, uconst(k)), uconst((ctrl >> (2 * k)) & 3u), tsel);
        uint32_t tx = ibin(Op_BitwiseAnd, tsel, uconst(1)), ty = ibin(Op_ShiftRightLogical, tsel, uconst(1));
        uint32_t xf = bcf(x_bits);
        uint32_t dx = id(); put(code, Op_DPdx, {t_f32, dx, xf});
        uint32_t dy = id(); put(code, Op_DPdy, {t_f32, dy, xf});
        // (tx-px) and (ty-py) as floats (each in {-1,0,1}).
        uint32_t dtx = id(); put(code, Op_ConvertSToF, {t_f32, dtx, bcs(ibin(Op_ISub, tx, px))});
        uint32_t dty = id(); put(code, Op_ConvertSToF, {t_f32, dty, bcs(ibin(Op_ISub, ty, py))});
        uint32_t r0 = id(); put(code, Op_FMul, {t_f32, r0, dtx, dx});
        uint32_t r1 = id(); put(code, Op_FMul, {t_f32, r1, dty, dy});
        uint32_t s0 = id(); put(code, Op_FAdd, {t_f32, s0, xf, r0});
        uint32_t s1 = id(); put(code, Op_FAdd, {t_f32, s1, s0, r1});
        return bcu(s1);
    }

}  // namespace prosper::gpu
