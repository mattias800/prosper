// Storage-buffer declarations moved from the inline declare_cbufs method.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <set>

namespace prosper::gpu {
void SpirvCompute::validate_cbuf_table_contracts(const ShaderResourceTable*& rt) {
    if (rt)
        for (const ShaderResource& resource : rt->resources)
            if (resource.table_index_count != 0u &&
                (resource.cls == ResourceClass::ConstantBuffer ||
                 resource.cls == ResourceClass::VertexBuffer) &&
                !valid_shader_buffer_table_contract(resource))
                invalid_cbuf_array_access = true;
}

void SpirvCompute::declare_additional_cbufs(const ShaderResourceTable*& rt, uint32_t& t_struct_u) {
    if (rt) {
        std::set<uint32_t> seen{2, 3};
        for (const auto& r : rt->resources) {
            if (r.cls != ResourceClass::ConstantBuffer && r.cls != ResourceClass::VertexBuffer)
                continue;
            if (!seen.insert(r.binding).second) continue;
            uint32_t v = id();
            put(deco, Op_Decorate, {v, Dec_DescriptorSet, desc_set});
            put(deco, Op_Decorate, {v, Dec_Binding, r.binding});
            if (r.table_index_count != 0) {
                // TABLE-INDEXED binding (#2412 stage 4b): the descriptor is one OF an array,
                // selected by an index only the GPU knows. The pointee becomes an array of the same
                // Block type rather than the Block itself, so this binding occupies
                // `table_index_count` descriptors and an access chain needs a leading index.
                //
                // A count of 0 never reaches here -- it means "not table-indexed".
                //
                // An IMPLAUSIBLE length emits a runtime array rather than a fixed array of that
                // length. This is the defensive path for a length that reflection could not read:
                // #2463 gives such a length its own sentinel (UINT32_MAX), and emitting
                // `OpTypeArray %Block 4294967295` would ask the driver for a 4-billion-descriptor
                // binding. The numeric guard catches that sentinel without this branch having to
                // depend on #2463 being merged first, and catches any other absurd count on the way.
                declare_descriptor_indexing();
                const uint32_t arr = id();
                put(types, Op_TypeArray,
                    {arr, t_struct_u,
                     uconst(r.table_index_count <= 4096u ? r.table_index_count : 1u)});
                const uint32_t arr_ptr = id();
                put(types, Op_TypePointer, {arr_ptr, SC_StorageBuffer, arr});
                declare_external_storage_buffer(arr_ptr, v);
                cbuf_var[r.binding] = v;
                cbuf_table_arity[r.binding] = r.table_index_count;
                if (r.table_index_sgpr != 0xFFFFFFFFu)
                    cbuf_table_index_sgpr[r.binding] = r.table_index_sgpr;
                continue;
            }
            declare_external_storage_buffer(t_ptr_sb_struct_u, v);
            cbuf_var[r.binding] = v;
        }
    }
}

void SpirvCompute::declare_cbuf_block_types(uint32_t& t_rta_u, uint32_t& t_struct_u) {
    put(deco, Op_Decorate, {t_rta_u, Dec_ArrayStride, 4});
    put(deco, Op_MemberDecorate, {t_struct_u, 0, Dec_Offset, 0});
    put(deco, Op_Decorate, {t_struct_u, Dec_Block});
    put(deco, Op_Decorate, {v_cbuf, Dec_DescriptorSet, desc_set});
    put(deco, Op_Decorate, {v_cbuf, Dec_Binding, 2});
    put(deco, Op_Decorate, {v_cbuf1, Dec_DescriptorSet, desc_set});
    put(deco, Op_Decorate, {v_cbuf1, Dec_Binding, 3});
    put(types, Op_TypeRuntimeArray, {t_rta_u, t_u32});
    put(types, Op_TypeStruct, {t_struct_u, t_rta_u});
    put(types, Op_TypePointer, {t_ptr_sb_struct_u, SC_StorageBuffer, t_struct_u});
}

void SpirvCompute::declare_cbufs(const ShaderResourceTable* rt) {
    validate_cbuf_table_contracts(rt);
    uint32_t t_rta_u = id(), t_struct_u = id();
    t_ptr_sb_struct_u = id();
    v_cbuf = id();
    v_cbuf1 = id();
    t_ptr_sb_u32 = id();
    declare_cbuf_block_types(t_rta_u, t_struct_u);
    // Bindings 2 and 3 are declared here rather than in the loop below, so a TABLE-INDEXED resource
    // on either of them has to be handled here too (#2472). Before this, both were declared
    // unconditionally as scalar buffers and then seeded into the loop's `seen` set, so an array at
    // binding 2 or 3 was skipped and silently emitted as a single descriptor -- and since Vulkan
    // permits a shader declaring one descriptor against a layout declaring N, that produced no
    // error: the shader read element 0 for every index. Titles put their constant buffers on
    // exactly these two bindings, so it was the common case that failed quietly.
    // Both helpers below must select the SAME resource, so the predicate lives in one place. Two
    // resources may share a binding (`declare_cbufs` does not prevent it), so a class filter on one
    // helper and not the other can take the arity from one and the index SGPR from the other —
    // wrong descriptor index, and quiet, because both values are individually plausible.
    auto table_indexed_here = [](const ShaderResource& r, uint32_t binding) {
        return r.binding == binding && r.table_index_count != 0 &&
               (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer);
    };
    auto table_arity_for = [&](uint32_t binding) -> uint32_t {
        if (!rt) return 0;
        for (const auto& r : rt->resources)
            if (table_indexed_here(r, binding)) return r.table_index_count;
        return 0;
    };
    auto index_sgpr_for = [&](uint32_t binding) -> uint32_t {
        if (!rt) return 0xFFFFFFFFu;
        for (const auto& r : rt->resources)
            if (table_indexed_here(r, binding)) return r.table_index_sgpr;
        return 0xFFFFFFFFu;
    };
    // Declare `binding`'s variable as an array of the Block type, reusing the id already decorated
    // for it, so its set/binding decorations above stay correct.
    auto declare_as_array = [&](uint32_t binding, uint32_t var, uint32_t arity) {
        declare_descriptor_indexing();
        const uint32_t arr = id();
        put(types, Op_TypeArray, {arr, t_struct_u, uconst(arity <= 4096u ? arity : 1u)});
        const uint32_t arr_ptr = id();
        put(types, Op_TypePointer, {arr_ptr, SC_StorageBuffer, arr});
        declare_external_storage_buffer(arr_ptr, var);
        cbuf_table_arity[binding] = arity;
        const uint32_t sgpr = index_sgpr_for(binding);
        if (sgpr != 0xFFFFFFFFu) cbuf_table_index_sgpr[binding] = sgpr;
    };
    const uint32_t arity2 = table_arity_for(2), arity3 = table_arity_for(3);
    // NOTE on the fallback: `buf_for_binding` returns `v_cbuf` for a binding it does not know. If
    // binding 2 is an array, that fallback yields an array-typed variable, and an access chain built
    // for a scalar pointee against it is malformed SPIR-V. It is NOT caught at emission: `put`/`putv`
    // (:383, :386) are a word assembler — they push a length word and the raw ids and type-check
    // nothing — so the module is emitted happily. Nor is it caught by `spirv-val`, which gates one
    // representative module per emitter path and has none for this case, since the case has never
    // been constructed. What actually happens is that pipeline creation rejects the module and the
    // draw is dropped. Both branches are hard errors rather than wrong pixels -- index 0 selects a
    // Block, so index 1 indexes into a structure, and SPIR-V requires a struct member index to be an
    // OpConstant, which a computed index is not; in the only constant case it yields a
    // pointer-to-runtime-array against a declared `t_ptr_sb_u32`, a type mismatch. So the direction
    // is still fail-visible, not silently wrong -- but "loud" here means a dropped draw, not a
    // validator, and nothing reports it as a type error at the point the mistake was made.
    if (arity2)
        declare_as_array(2, v_cbuf, arity2);
    else
        declare_external_storage_buffer(t_ptr_sb_struct_u, v_cbuf);
    if (arity3)
        declare_as_array(3, v_cbuf1, arity3);
    else
        declare_external_storage_buffer(t_ptr_sb_struct_u, v_cbuf1);
    put(types, Op_TypePointer, {t_ptr_sb_u32, SC_StorageBuffer, t_u32});
    cbuf_var[2] = v_cbuf;
    cbuf_var[3] = v_cbuf1;
    // N-buffer model: declare an additional storage buffer for each distinct constant/vertex buffer
    // binding the resource table uses beyond 2/3, so the shader's several buffers stay distinct. All
    // share the same Block struct type (a runtime array of u32); only the binding decoration differs.
    declare_additional_cbufs(rt, t_struct_u);
}
}   // namespace prosper::gpu
