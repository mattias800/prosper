// Buffer selectors, loads and predicated stores.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
uint32_t SpirvCompute::cbuf_element_ptr(uint32_t buf, uint32_t binding, uint32_t idx,
                                        uint32_t* array_valid) {
    if (array_valid) *array_valid = 0;
    auto arity = cbuf_table_arity.find(binding);
    auto index_sgpr = cbuf_table_index_sgpr.find(binding);
    auto live_selector = cbuf_table_selector_value.find(binding);
    auto selector_block = cbuf_table_selector_block.find(binding);
    uint32_t ptr = id();
    if (arity != cbuf_table_arity.end()) {
        // A live GPU-computed selector takes precedence over the user-SGPR convention: the two
        // are different contracts, and a table whose index the guest derives on the GPU has no
        // push-constant home at all.
        const bool have_live_selector =
            is_compute && arity->second <= 4096u &&
            live_selector != cbuf_table_selector_value.end() && live_selector->second != 0 &&
            // Dominance, see cbuf_table_selector_block. Failing this leaves `have_live_selector`
            // false and, since this mode never sets `table_index_sgpr`, drops through to
            // `invalid_cbuf_array_access` below -- the module is discarded. That is exactly what
            // the same shader does WITHOUT this feature (its MUBUF cannot resolve), so the guard
            // cannot regress anything: it can only decline a case that was never sound.
            selector_block != cbuf_table_selector_block.end() &&
            selector_block->second == cur_block;
        // Only the compute shell declares the user-SGPR push-constant block. Graphics paths do
        // not currently have a runtime source for this selector, so emitting an access there
        // would manufacture type/variable id zero and an invalid module.
        if (!have_live_selector &&
            (!is_compute || arity->second > 4096u || index_sgpr == cbuf_table_index_sgpr.end() ||
             index_sgpr->second >= push_constant_dword_count)) {
            invalid_cbuf_array_access = true;
            putv(code, Op_AccessChain, {t_ptr_sb_u32, ptr, buf, uconst(0), uconst(0), idx});
            return ptr;
        }
        const uint32_t selector =
            have_live_selector ? live_selector->second : load_push_constant(index_sgpr->second);
        const uint32_t valid = ucmp(Op_ULessThan, selector, uconst(arity->second));
        const uint32_t safe_selector = sel(valid, selector, uconst(0));
        if (array_valid) *array_valid = valid;
        // NonUniform on both the selector and the pointer, unconditionally. Measured: of
        // PPSA04263's 51 compute launches, 5 are local=256x1x1 -- four waves per workgroup, so four
        // EXECs and four distinct scalar indices, wave-uniform but NOT dynamically uniform across
        // the invocation group. 43 of 51 are one wave per group, which is why the "obviously
        // uniform" reading looks safe and is wrong. `load_push_constant` mints a fresh id per call
        // and does not cache, so these decorations never duplicate onto one id.
        put(deco, Op_Decorate, {safe_selector, Dec_NonUniform});
        putv(code, Op_AccessChain, {t_ptr_sb_u32, ptr, buf, safe_selector, uconst(0), idx});
        put(deco, Op_Decorate, {ptr, Dec_NonUniform});
        return ptr;
    }
    putv(code, Op_AccessChain, {t_ptr_sb_u32, ptr, buf, uconst(0), idx});
    return ptr;
}

uint32_t SpirvCompute::cbuf_load(uint32_t idx, uint32_t binding, bool coherent_access) {
    cbuf_ordinary_accesses.insert(binding);
    return cbuf_load_impl(idx, binding, coherent_access);
}

void SpirvCompute::cbuf_store(uint32_t idx, uint32_t value, uint32_t binding, bool predicated,
                              uint32_t pred, bool coherent_access) {
    cbuf_ordinary_accesses.insert(binding);
    if (cbuf_table_arity.count(binding)) invalid_cbuf_array_access = true;
    uint32_t buf = buf_for_binding(binding);
    if (coherent_access) mark_cbuf_coherent(binding);
    auto emit = [&]() {
        uint32_t p = cbuf_element_ptr(buf, binding, idx);
        if (coherent_access)
            put(code, Op_Store, {p, value, MemAccess_Volatile});
        else
            put(code, Op_Store, {p, value});
    };
    if (!predicated) {
        emit();
        return;
    }
    uint32_t then = id(), merge = id();
    put(code, Op_SelectionMerge, {merge, 0});
    put(code, Op_BranchConditional, {pred, then, merge});
    put(code, Op_Label, {then});
    cur_block = then;
    emit();
    put(code, Op_Branch, {merge});
    put(code, Op_Label, {merge});
    cur_block = merge;
}
}   // namespace prosper::gpu
