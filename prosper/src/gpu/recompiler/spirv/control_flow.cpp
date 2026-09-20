// Generated move, not new code: `SpirvCompute`'s method bodies, lifted out of
// gpu/recompiler/rdna2_to_spirv_internal.hpp. outline_methods.py checked each body byte for byte.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

uint32_t SpirvCompute::emit_phi2(uint32_t type, uint32_t v0, uint32_t l0, size_t& patch_off) {
        uint32_t r = id();
        put(code, Op_Phi, {type, r, v0, l0, 0u, 0u});
        patch_off = code.size() - 2;   // the trailing {v1, l1} placeholders
        propagate_fragment_wave_vote(r, v0);
        fragment_wave_vote_phi_patch_results[patch_off] = r;
        return r;
    }

void SpirvCompute::patch_phi(size_t patch_off, uint32_t v1, uint32_t l1) {
        code[patch_off] = v1;
        code[patch_off + 1] = l1;
        const auto result = fragment_wave_vote_phi_patch_results.find(patch_off);
        if (result != fragment_wave_vote_phi_patch_results.end())
            propagate_fragment_wave_vote(result->second, v1);
    }

void SpirvCompute::emit_switch(uint32_t selector, uint32_t fallback,
                     const std::vector<std::pair<uint32_t, uint32_t>>& cases) {
        std::vector<uint32_t> operands{selector, fallback};
        for (const auto& c : cases) { operands.push_back(c.first); operands.push_back(c.second); }
        putv(code, Op_Switch, operands);
    }

uint32_t SpirvCompute::logical_not(uint32_t value) {
        uint32_t result = id();
        put(code, Op_LogicalNot, {t_bool, result, value});
        propagate_fragment_wave_vote(result, value);
        return result;
    }

void SpirvCompute::discard_unless(uint32_t alive) {
        uint32_t killL = id(), mergeL = id();
        put(code, Op_SelectionMerge, {mergeL, 0});
        put(code, Op_BranchConditional, {alive, mergeL, killL});
        emit_label(killL); put(code, Op_Kill, {});
        emit_label(mergeL);
    }

uint32_t SpirvCompute::emit_phi_2way(uint32_t type, uint32_t va, uint32_t la, uint32_t vb, uint32_t lb) {
        uint32_t r = id();
        put(code, Op_Phi, {type, r, va, la, vb, lb});
        propagate_fragment_wave_vote(r, va, vb);
        return r;
    }

void SpirvCompute::device_uniform_release_barrier() {
        put(code, Op_MemoryBarrier,
            {uconst(Scope_Device), uconst(MemSem_UniformRelease)});
    }

void SpirvCompute::barrier() {
        if (ngg_private_lds) return; // exact one-lane wrapper has no peer requiring synchronization
        uses_barrier = true;
        put(code, Op_ControlBarrier, {uconst(Scope_Workgroup), uconst(Scope_Workgroup), uconst(MemSem_WGAcqRel)});
    }

uint32_t SpirvCompute::land(uint32_t a, uint32_t b_) {
        uint32_t r = id();
        put(code, Op_LogicalAnd, {t_bool, r, a, b_});
        propagate_fragment_wave_vote(r, a, b_);
        return r;
    }

uint32_t SpirvCompute::lor(uint32_t a, uint32_t b_) {
        uint32_t r = id();
        put(code, Op_LogicalOr, {t_bool, r, a, b_});
        propagate_fragment_wave_vote(r, a, b_);
        return r;
    }

std::vector<uint32_t> SpirvCompute::finish() {
        if (invalid_cbuf_array_access) return {};
        if (invocation_guard_merge) {
            emit_branch(invocation_guard_merge);
            emit_label(invocation_guard_merge);
        }
        put(code, Op_Return, {}); put(code, Op_FunctionEnd, {});
        // EntryPoint is emitted here (not in begin_*) so lazily-declared Input/Output varyings — added
        // to `iface` as v_interp / EXP PARAM are encountered — appear in the interface list (SPIR-V 1.3).
        { std::vector<uint32_t> o{exec_model, f_main}; pstr(o, "main");
          for (uint32_t v : iface) o.push_back(v); putv(entry, Op_EntryPoint, o); }
        if (is_compute && compute_min_subgroup_size) {
            char marker[64];
            std::snprintf(marker, sizeof marker, "Prosper.ComputeSubgroupMin=%u",
                          compute_min_subgroup_size);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        if (is_fragment && fragment_required_subgroup_size) {
            char marker[64];
            std::snprintf(marker, sizeof marker, "Prosper.FragmentSubgroupSize=%u",
                          fragment_required_subgroup_size);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
            // A SEPARATE marker, not a wider one: a module cached or captured before #2147
            // carries the size and not this, and a reader must be able to tell 'reasons
            // unknown' from 'reasons none'. Absent and zero are the same number and opposite
            // facts -- a zero would assert that nothing required a width this module demands.
            std::snprintf(marker, sizeof marker, "Prosper.FragmentSubgroupWhy=%u",
                          fragment_wave_reasons);
            words.clear();
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        for (const auto& [binding, semantic] : cbuf_zero_pad_candidates) {
            if (cbuf_ordinary_accesses.count(binding)) continue;
            char marker[96];
            const char* token = semantic == StorageBufferTailSemantic::Uint16 ? "u16" :
                                semantic == StorageBufferTailSemantic::Float16 ? "f16" : nullptr;
            if (!token) continue;
            std::snprintf(marker, sizeof marker, "Prosper.StorageBufferZeroPad=%u,%u,2,4,%s",
                          desc_set, binding, token);
            std::vector<uint32_t> words;
            pstr(words, marker);
            putv(debug, Op_ModuleProcessed, words);
        }
        std::vector<uint32_t> m{0x07230203u, 0x00010300u, 0u, next_id, 0u};
        for (auto* s : {&caps, &exts, &extimp, &mem, &entry, &exec, &debug, &deco, &types, &code})
            m.insert(m.end(), s->begin(), s->end());
        return m;
    }

}  // namespace prosper::gpu
