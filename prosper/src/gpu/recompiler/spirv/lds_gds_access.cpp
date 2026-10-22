// LDS/GDS storage declarations, predicated stores and scratch-wave append.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
void SpirvCompute::declare_internal_gds(uint32_t set, uint32_t binding) {
    if (v_internal_gds) return;
    uint32_t runtime_array = id(), block = id(), block_ptr = id();
    put(deco, Op_Decorate, {runtime_array, Dec_ArrayStride, 4});
    put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    put(deco, Op_Decorate, {block, Dec_Block});
    put(types, Op_TypeRuntimeArray, {runtime_array, t_u32});
    put(types, Op_TypeStruct, {block, runtime_array});
    put(types, Op_TypePointer, {block_ptr, SC_StorageBuffer, block});
    t_ptr_gds_u32 = id();
    put(types, Op_TypePointer, {t_ptr_gds_u32, SC_StorageBuffer, t_u32});
    v_internal_gds = id();
    put(types, Op_Variable, {block_ptr, v_internal_gds, SC_StorageBuffer});
    put(deco, Op_Decorate, {v_internal_gds, Dec_DescriptorSet, set});
    put(deco, Op_Decorate, {v_internal_gds, Dec_Binding, binding});
}

void SpirvCompute::lds_store(uint32_t idx, uint32_t value, bool predicated, uint32_t pred,
                             bool atomicize) {
    if (atomicize) {
        lds_atomic(Op_AtomicExchange, idx, value, predicated, pred);
        return;
    }
    if (!predicated) {
        uint32_t p = id();
        putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
        put(code, Op_Store, {p, value});
        return;
    }
    uint32_t then = id(), merge = id();
    put(code, Op_SelectionMerge, {merge, 0});
    put(code, Op_BranchConditional, {pred, then, merge});
    put(code, Op_Label, {then});
    cur_block = then;
    uint32_t p = id();
    putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
    put(code, Op_Store, {p, value});
    put(code, Op_Branch, {merge});
    put(code, Op_Label, {merge});
    cur_block = merge;
}

uint32_t SpirvCompute::wave_append(uint32_t index, uint32_t active_bool, bool consume, bool gds) {
    declare_wave_lds();
    const uint32_t active = sel(active_bool, uconst(1), uconst(0));
    uint32_t p = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, p, lds_wave, linear_localid});
    put(code, Op_Store, {p, active});
    barrier();

    const uint32_t wave_shift = wave_size == 32 ? 5u : 6u;
    const uint32_t lane = ibin(Op_BitwiseAnd, linear_localid, uconst(wave_size - 1));
    const uint32_t wave_base =
        ibin(Op_ShiftLeftLogical, ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift)),
             uconst(wave_shift));
    const uint32_t wave_index = ibin(Op_ShiftRightLogical, linear_localid, uconst(wave_shift));
    uint32_t count = uconst(0);
    for (uint32_t i = 0; i < local_count; ++i) {
        uint32_t q = id();
        putv(code, Op_AccessChain, {t_ptr_wg_u32b, q, lds_wave, uconst(i)});
        uint32_t v = id();
        put(code, Op_Load, {t_u32, v, q});
        const uint32_t same_wave = ucmp(Op_IEqual, wave_index, uconst(i / wave_size));
        count = b_iadd(count, sel(same_wave, v, uconst(0)));
    }

    const uint32_t is_leader = ucmp(Op_IEqual, lane, uconst(0));
    const uint32_t leader = id(), reduced = id();
    emit_selmerge(reduced);
    emit_condbranch(is_leader, leader, reduced);
    emit_label(leader);
    const uint32_t old =
        gds ? compute_gds_atomic_rtn(consume ? Op_AtomicISub : Op_AtomicIAdd, index, count)
            : lds_atomic_rtn(consume ? Op_AtomicISub : Op_AtomicIAdd, index, count, false, btrue(),
                             uconst(0));
    uint32_t result_slot = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, result_slot, lds_wave, wave_base});
    put(code, Op_Store, {result_slot, old});
    emit_branch(reduced);
    emit_label(reduced);
    barrier();
    uint32_t result_ptr = id();
    putv(code, Op_AccessChain, {t_ptr_wg_u32b, result_ptr, lds_wave, wave_base});
    uint32_t result = id();
    put(code, Op_Load, {t_u32, result, result_ptr});
    barrier();
    return result;
}
}  // namespace prosper::gpu
