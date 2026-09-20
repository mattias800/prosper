// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void SpirvCompute::compute_gds_atomic_minmax(uint16_t opcode, uint32_t index, uint32_t value,
                                   uint32_t pred) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t old = id();
        put(code, opcode, {t_u32, old, pointer, uconst(Scope_Device),
                           uconst(MemSem_UniformAcqRel), value});
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }

void SpirvCompute::compute_gds_store(uint32_t index, uint32_t value, bool predicated, uint32_t pred) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        auto emit = [&]() {
            uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
            put(code, Op_Store, {pointer, value});
        };
        if (!predicated) { emit(); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        emit();
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }

uint32_t SpirvCompute::compute_gds_atomic_rtn(uint32_t op, uint32_t index, uint32_t value) {
        declare_internal_gds(0, kComputeInternalGdsBinding);
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t result = id();
        put(code, op,
            {t_u32, result, pointer, uconst(Scope_Device),
             uconst(MemSem_UniformAcqRel), value});
        return result;
    }

uint32_t SpirvCompute::native_gds_append(uint32_t index, uint32_t exec_bit, bool consume) {
        declare_internal_gds(is_compute ? 0 : 1,
                             is_compute ? kComputeInternalGdsBinding : 0);
        if (!declared_subgroup) {
            put(caps, Op_Capability, {Cap_GroupNonUniform});
            declared_subgroup = true;
        }
        if (!declared_subgroup_arithmetic) {
            put(caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
            declared_subgroup_arithmetic = true;
        }
        const uint32_t active_bit = is_fragment
            ? land(exec_bit, logical_not(helper_invocation())) : exec_bit;
        const uint32_t contribution = sel(active_bit, uconst(1), uconst(0));
        uint32_t count = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, count, uconst(Scope_Subgroup), GroupOp_Reduce, contribution});
        uint32_t prefix = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, prefix, uconst(Scope_Subgroup), GroupOp_ExclusiveScan, contribution});
        const uint32_t elected = land(active_bit, ucmp(Op_IEqual, prefix, uconst(0)));
        const uint32_t entry = cur_block, leader = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {elected, leader, merge});
        put(code, Op_Label, {leader}); cur_block = leader;
        uint32_t pointer = id();
        putv(code, Op_AccessChain,
             {t_ptr_gds_u32, pointer, v_internal_gds, uconst(0), index});
        uint32_t leader_old = id();
        put(code, consume ? Op_AtomicISub : Op_AtomicIAdd,
            {t_u32, leader_old, pointer, uconst(Scope_Device),
             uconst(MemSem_UniformAcqRel), count});
        const uint32_t leader_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        const uint32_t local_old = emit_phi_2way(
            t_u32, leader_old, leader_end, uconst(0), entry);
        uint32_t old = id();
        put(code, Op_GroupNonUniformIAdd,
            {t_u32, old, uconst(Scope_Subgroup), GroupOp_Reduce, local_old});
        return old;
    }

uint32_t SpirvCompute::lds_load(uint32_t idx) {
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
        uint32_t r = id(); put(code, Op_Load, {t_u32, r, p}); return r;
    }

uint32_t SpirvCompute::lds_load(uint32_t idx, bool predicated, uint32_t pred, uint32_t fallback) {
        if (!predicated) return lds_load(idx);
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = lds_load(idx);
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }

void SpirvCompute::lds_atomic(uint32_t op, uint32_t idx, uint32_t value, bool predicated, uint32_t pred) {
        auto emit = [&]() {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel), value});
        };
        if (!predicated) { emit(); return; }
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        emit();
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }

void SpirvCompute::lds_atomic_fminmax(uint32_t idx, uint32_t value, bool is_min,
                            bool predicated, uint32_t pred) {
        auto emit = [&]() {
            const uint32_t p = id();
            putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            const uint32_t initial = id();
            put(code, Op_AtomicLoad,
                {t_u32, initial, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcquire)});

            const uint32_t preheader = cur_block;
            const uint32_t header = id(), again = id(), merge = id();
            put(code, Op_Branch, {header});
            put(code, Op_Label, {header}); cur_block = header;
            size_t retry_patch = 0;
            const uint32_t expected = emit_phi2(t_u32, initial, preheader, retry_patch);
            const uint32_t desired = atomic_fminmax_bits(expected, value, is_min);
            const uint32_t observed = id();
            put(code, Op_AtomicCompareExchange,
                {t_u32, observed, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel),
                 uconst(MemSem_WGAcquire), desired, expected});
            const uint32_t succeeded = ucmp(Op_IEqual, observed, expected);
            put(code, Op_LoopMerge, {merge, again, 0});
            put(code, Op_BranchConditional, {succeeded, merge, again});
            put(code, Op_Label, {again}); cur_block = again;
            put(code, Op_Branch, {header});
            patch_phi(retry_patch, observed, again);
            put(code, Op_Label, {merge}); cur_block = merge;
        };
        if (!predicated) { emit(); return; }
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        emit();
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
    }

uint32_t SpirvCompute::lds_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value,
                            bool predicated, uint32_t pred, uint32_t fallback) {
        auto emit = [&]() {
            uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_lds_u32, p, lds_var, idx});
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel), value});
            return result;
        };
        if (!predicated) return emit();
        const uint32_t entry = cur_block;
        uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }

}  // namespace prosper::gpu
