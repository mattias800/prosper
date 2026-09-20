// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void SpirvCompute::declare_descriptor_indexing() {
        if (descriptor_indexing_declared) return;
        descriptor_indexing_declared = true;
        put(caps, Op_Capability, {Cap_ShaderNonUniform});
        put(caps, Op_Capability, {Cap_StorageBufferArrayNonUniformIndexing});
        std::vector<uint32_t> o; pstr(o, "SPV_EXT_descriptor_indexing");
        putv(exts, Op_Extension, o);
    }

void SpirvCompute::declare_float_controls(uint32_t entry) {
        if (float_controls_declared) return;
        if (!signed_zero_inf_nan_preserve_declared()) return;
        float_controls_declared = true;
        put(caps, Op_Capability, {Cap_SignedZeroInfNanPreserve});
        std::vector<uint32_t> o; pstr(o, "SPV_KHR_float_controls");
        putv(exts, Op_Extension, o);
        put(exec, Op_ExecutionMode, {entry, EM_SignedZeroInfNanPreserve, 32});
    }

void SpirvCompute::pstr(std::vector<uint32_t>& v, const char* s) {
        size_t len = std::strlen(s);
        for (size_t i = 0; i <= len; i += 4) { uint32_t w = 0;
            for (size_t k = 0; k < 4; k++) { size_t j = i + k; if (j <= len) w |= (uint32_t)(uint8_t)s[j] << (8*k); }
            v.push_back(w); }
    }

uint32_t SpirvCompute::fconst(float f) {
        uint32_t b = fbits(f); auto it = fconst_cache.find(b); if (it != fconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_f32, c, b}); fconst_cache[b] = c; return c;
    }

uint32_t SpirvCompute::uconst(uint32_t v) {
        auto it = uconst_cache.find(v); if (it != uconst_cache.end()) return it->second;
        uint32_t c = id(); put(types, Op_Constant, {t_u32, c, v}); uconst_cache[v] = c; return c;
    }

bool SpirvCompute::uconst_literal(uint32_t id_, uint32_t* out) const {
        for (const auto& kv : uconst_cache)
            if (kv.second == id_) { if (out) *out = kv.first; return true; }
        return false;
    }

void SpirvCompute::declare_guest_scratch(const StaticScratchLayout& layout) {
        if (!layout.valid || !layout.used || !layout.dwords || guest_scratch) return;
        guest_scratch_min_byte = layout.min_byte;
        guest_scratch_saddr = layout.saddr;
        guest_scratch_dwords = layout.dwords;
        const uint32_t array = id();
        put(types, Op_TypeArray, {array, t_u32, uconst(layout.dwords)});
        uint32_t ptr_array = 0;
        guest_scratch = function_var(array, ptr_array);
        t_ptr_guest_scratch_u32 = id();
        put(types, Op_TypePointer, {t_ptr_guest_scratch_u32, SC_Function, t_u32});
    }

void SpirvCompute::declare_external_storage_buffer(uint32_t pointer_type, uint32_t variable) {
        put(types, Op_Variable, {pointer_type, variable, SC_StorageBuffer});
        put(deco, Op_Decorate, {variable, Dec_Aliased});
    }

uint32_t SpirvCompute::uconst64(uint64_t v) {
        // For values that fit in 32 bits (all current uses are shift amounts of 32), materialize the
        // u64 by widening a 32-bit constant in the body rather than emitting a 2-word 64-bit OpConstant
        // literal. Two reasons, both satisfied by this form: (1) MoltenVK's bundled SPIRV-Cross
        // mis-parses 64-bit OpConstant literals (reads the high value word as a zero Id -> "Cannot
        // resolve expression type", failing the whole shader) — a u32->u64 OpUConvert has no 2-word
        // literal; (2) the result is still u64-typed, so llvmpipe's requirement that a u64 shift amount
        // be u64 (a u32 shift operand drops the high half there) is preserved. Semantically identical on
        // every driver. A genuine >32-bit constant still needs the 2-word form (not currently emitted).
        if (v <= 0xffffffffull) { uint32_t r = id(); put(code, Op_UConvert, {t_u64(), r, uconst((uint32_t)v)}); return r; }
        uint32_t c = id(); put(types, Op_Constant, {t_u64(), c, (uint32_t)v, (uint32_t)(v >> 32)}); return c;
    }

void SpirvCompute::declare_indirect_pointer_descriptor_capture() {
        uint32_t pointer_type = 0;
        indirect_pointer_source_record_var = function_var(t_u32, pointer_type);
        indirect_pointer_source_root_lo_var = function_var(t_u32, pointer_type);
        indirect_pointer_source_root_hi_var = function_var(t_u32, pointer_type);
        // UINT32_MAX cannot name a record in the validated source table. It is also the safe
        // fail-closed state for a lane which reaches a consumer without executing the producer.
        store_function(indirect_pointer_source_record_var, uconst(UINT32_MAX));
        store_function(indirect_pointer_source_root_lo_var, uconst(0));
        store_function(indirect_pointer_source_root_hi_var, uconst(0));
    }

bool SpirvCompute::declare_compute_atomic_image_buffer(uint32_t binding) {
        if (!is_compute || !t_ptr_sb_struct_u) return false;
        if (atomic_img_buf_bindings.count(binding)) return true;   // ours already; reuse it
        if (cbuf_var.count(binding)) return false;                 // someone else's; refuse
        const uint32_t variable = id();
        put(deco, Op_Decorate, {variable, Dec_DescriptorSet, desc_set});
        put(deco, Op_Decorate, {variable, Dec_Binding, binding});
        declare_external_storage_buffer(t_ptr_sb_struct_u, variable);
        cbuf_var[binding] = variable;
        atomic_img_buf_bindings.insert(binding);
        return true;
    }

void SpirvCompute::declare_lds() {
        if (lds_var) return;
        const uint32_t dwords = is_compute ? lds_dwords : vertex_lds_dwords;
        if (!dwords) return;
        uint32_t len = uconst(dwords);
        uint32_t t_arr = id();        put(types, Op_TypeArray, {t_arr, t_u32, len});
        if (is_vertex) {
            // Vertex-stage LDS is only legal for the exact NGG wrapper selected by recompile_vertex.
            // All other vertex DS shapes reject before reaching this declaration.
            uint32_t t_ptr_fn_arr = 0;
            lds_var = function_var(t_arr, t_ptr_fn_arr);
            t_ptr_lds_u32 = id();
            put(types, Op_TypePointer, {t_ptr_lds_u32, SC_Function, t_u32});
        } else {
            uint32_t t_ptr_wg_arr = id(); put(types, Op_TypePointer, {t_ptr_wg_arr, SC_Workgroup, t_arr});
            lds_var = id();               put(types, Op_Variable, {t_ptr_wg_arr, lds_var, SC_Workgroup});
            t_ptr_lds_u32 = id();
            put(types, Op_TypePointer, {t_ptr_lds_u32, SC_Workgroup, t_u32});
        }
    }

bool SpirvCompute::declare_cfg_scratch(uint32_t dwords) {
        // OpTypeArray is immutable once emitted. A later phased dispatcher may require a wider
        // layout than the first phase; fail closed if its caller did not pre-size from the complete
        // stream instead of emitting an out-of-bounds Workgroup access.
        if (cfg_scratch) return dwords <= cfg_scratch_dwords;
        uint32_t t_arr = id(); put(types, Op_TypeArray, {t_arr, t_u32, uconst(dwords)});
        uint32_t t_ptr = id(); put(types, Op_TypePointer, {t_ptr, SC_Workgroup, t_arr});
        cfg_scratch = id(); put(types, Op_Variable, {t_ptr, cfg_scratch, SC_Workgroup});
        t_ptr_cfg_u32 = id(); put(types, Op_TypePointer, {t_ptr_cfg_u32, SC_Workgroup, t_u32});
        cfg_scratch_dwords = dwords;
        return true;
    }

}  // namespace prosper::gpu
