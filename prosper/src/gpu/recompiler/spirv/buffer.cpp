// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t SpirvCompute::elem_ptr(uint32_t bufvar, uint32_t k) {
        uint32_t idx = gidx;
        if (stride != 1) { uint32_t m = id(); put(code, Op_IMul, {t_u32, m, gidx, uconst(stride)}); idx = m; }
        if (k != 0) { uint32_t a = id(); put(code, Op_IAdd, {t_u32, a, idx, uconst(k)}); idx = a; }
        uint32_t p = id(); putv(code, Op_AccessChain, {t_ptr_sb_f32, p, bufvar, uconst(0), idx}); return p;
    }

uint32_t SpirvCompute::buf_for_binding(uint32_t binding) {
        if (binding == 2) return v_cbuf;
        if (binding == 3) return v_cbuf1;
        auto it = cbuf_var.find(binding); return it != cbuf_var.end() ? it->second : v_cbuf;
    }

void SpirvCompute::mark_cbuf_coherent(uint32_t binding) {
        const uint32_t buf = buf_for_binding(binding);
        if (cbuf_coherent_vars.insert(buf).second)
            put(deco, Op_Decorate, {buf, Dec_Coherent});
    }

uint32_t SpirvCompute::cbuf_load_impl(uint32_t idx, uint32_t binding, bool coherent_access) {
        uint32_t buf = buf_for_binding(binding);
        if (coherent_access) mark_cbuf_coherent(binding);
        uint32_t array_valid = 0;
        uint32_t p = cbuf_element_ptr(buf, binding, idx, &array_valid);
        uint32_t r = id();
        if (coherent_access) put(code, Op_Load, {t_u32, r, p, MemAccess_Volatile});
        else                 put(code, Op_Load, {t_u32, r, p});
        return array_valid ? sel(array_valid, r, uconst(0)) : r;
    }

uint32_t SpirvCompute::cbuf_atomic_rtn(uint32_t op, uint32_t idx, uint32_t value, uint32_t binding,
                             bool predicated, uint32_t pred, uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
        const uint32_t buf = buf_for_binding(binding);
        auto emit = [&]() {
            uint32_t p = cbuf_element_ptr(buf, binding, idx);
            uint32_t result = id();
            put(code, op, {t_u32, result, p, uconst(Scope_Device),
                           uconst(MemSem_UniformAcqRel), value});
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

uint32_t SpirvCompute::cbuf_atomic_fminmax_rtn(uint32_t idx, uint32_t value, uint32_t binding,
                                     bool is_min, bool predicated, uint32_t pred,
                                     uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
        const uint32_t buf = buf_for_binding(binding);
        auto emit = [&]() {
            const uint32_t pointer = cbuf_element_ptr(buf, binding, idx);
            const uint32_t initial = id();
            put(code, Op_AtomicLoad,
                {t_u32, initial, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcquire)});

            const uint32_t preheader = cur_block;
            const uint32_t header = id(), again = id(), merge = id();
            put(code, Op_Branch, {header});
            put(code, Op_Label, {header}); cur_block = header;
            size_t retry_patch = 0;
            const uint32_t expected = emit_phi2(t_u32, initial, preheader, retry_patch);
            const uint32_t desired = atomic_fminmax_bits(expected, value, is_min);
            const uint32_t observed = id();
            put(code, Op_AtomicCompareExchange,
                {t_u32, observed, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcqRel), uconst(MemSem_UniformAcquire),
                 desired, expected});
            const uint32_t succeeded = ucmp(Op_IEqual, observed, expected);
            put(code, Op_LoopMerge, {merge, again, 0});
            put(code, Op_BranchConditional, {succeeded, merge, again});
            put(code, Op_Label, {again}); cur_block = again;
            put(code, Op_Branch, {header});
            patch_phi(retry_patch, observed, again);
            put(code, Op_Label, {merge}); cur_block = merge;
            return expected;
        };
        if (!predicated) return emit();
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u32, result, then_end, fallback, entry);
    }

uint32_t SpirvCompute::cbuf_atomic_x2_rtn(uint32_t op, uint32_t index, uint32_t value, uint32_t binding,
                                uint32_t pred, uint32_t fallback) {
        cbuf_ordinary_accesses.insert(binding);
        if (!declared_int64_atomics) {
            put(caps, Op_Capability, {Cap_Int64Atomics});
            declared_int64_atomics = true;
        }
        const uint32_t buf = u64_buf_for_binding(binding);
        auto emit = [&]() {
            const uint32_t pointer = id();
            putv(code, Op_AccessChain,
                 {t_ptr_sb_u64, pointer, buf, uconst(0), index});
            const uint32_t result = id();
            put(code, op,
                {t_u64(), result, pointer, uconst(Scope_Device),
                 uconst(MemSem_UniformAcqRel), value});
            return result;
        };
        const uint32_t entry = cur_block;
        const uint32_t then = id(), merge = id();
        put(code, Op_SelectionMerge, {merge, 0});
        put(code, Op_BranchConditional, {pred, then, merge});
        put(code, Op_Label, {then}); cur_block = then;
        const uint32_t result = emit();
        const uint32_t then_end = cur_block;
        put(code, Op_Branch, {merge});
        put(code, Op_Label, {merge}); cur_block = merge;
        return emit_phi_2way(t_u64(), result, then_end, fallback, entry);
    }

uint32_t SpirvCompute::load_push_constant(uint32_t index) {
        uint32_t pointer = id();
        put(code, Op_AccessChain,
            {t_ptr_push_u32, pointer, v_push_constants, uconst(0), uconst(index)});
        uint32_t value = id();
        put(code, Op_Load, {t_u32, value, pointer});
        return value;
    }

}  // namespace prosper::gpu
