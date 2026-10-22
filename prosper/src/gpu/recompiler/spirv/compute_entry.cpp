// Compute entry setup moved from SpirvCompute's inline begin method.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {
void SpirvCompute::begin_compute_layout(uint32_t& local_x, uint32_t& local_y, uint32_t& local_z,
                                        uint32_t& t_rta, uint32_t& t_struct,
                                        uint32_t& push_constant_dwords) {
    exec_model = Exec_GLCompute;
    iface = {v_gid, v_groupid, v_localid};   // EntryPoint deferred to finish()
    // REQUIRE_FULL_SUBGROUPS constrains LocalSize X, not merely the total workgroup size. The
    // native shell never consumes Vulkan's LocalInvocationId: it reconstructs the original guest
    // x/y/z coordinates below from the exact subgroup lane order. Declare the equivalent Vulkan
    // workgroup flattened on X so 8x8 and other multidimensional guest waves satisfy that rule.
    const uint32_t vulkan_local_x = native_subgroup_size ? local_count : local_x;
    const uint32_t vulkan_local_y = native_subgroup_size ? 1u : local_y;
    const uint32_t vulkan_local_z = native_subgroup_size ? 1u : local_z;
    put(exec, Op_ExecutionMode,
        {f_main, EM_LocalSize, vulkan_local_x, vulkan_local_y, vulkan_local_z});
    put(deco, Op_Decorate, {v_gid, Dec_BuiltIn, BI_GlobalInvocationId});
    put(deco, Op_Decorate, {v_groupid, Dec_BuiltIn, BI_WorkgroupId});
    put(deco, Op_Decorate,
        {v_localid, Dec_BuiltIn, BI_LocalInvocationId});   // wave lane index (.x)
    put(deco, Op_Decorate, {t_rta, Dec_ArrayStride, 4});
    put(deco, Op_MemberDecorate, {t_struct, 0, Dec_Offset, 0});
    put(deco, Op_Decorate, {t_struct, Dec_Block});
    put(deco, Op_Decorate, {v_in, Dec_DescriptorSet, shell_io_descriptor_set});
    put(deco, Op_Decorate, {v_in, Dec_Binding, 0});
    put(deco, Op_Decorate, {v_out, Dec_DescriptorSet, shell_io_descriptor_set});
    put(deco, Op_Decorate, {v_out, Dec_Binding, 1});
    put(types, Op_TypeVoid, {t_void});
    put(types, Op_TypeFunction, {t_fn, t_void});
    put(types, Op_TypeFloat, {t_f32, 32});
    put(types, Op_TypeInt, {t_u32, 32, 0});
    put(types, Op_TypeInt, {t_i32, 32, 1});
    put(types, Op_TypeVector, {t_v3u, t_u32, 3});
    put(types, Op_TypeVector, {t_v4f, t_f32, 4});
    put(types, Op_TypeBool, {t_bool});
    if (push_constant_dwords) {
        const uint32_t count = id();
        put(types, Op_Constant, {t_u32, count, push_constant_dwords});
        const uint32_t array = id();
        put(types, Op_TypeArray, {array, t_u32, count});
        put(deco, Op_Decorate, {array, Dec_ArrayStride, 4});
        const uint32_t block = id();
        put(types, Op_TypeStruct, {block, array});
        put(deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
        put(deco, Op_Decorate, {block, Dec_Block});
        const uint32_t block_pointer = id();
        put(types, Op_TypePointer, {block_pointer, SC_PushConstant, block});
        t_ptr_push_u32 = id();
        put(types, Op_TypePointer, {t_ptr_push_u32, SC_PushConstant, t_u32});
        v_push_constants = id();
        put(types, Op_Variable, {block_pointer, v_push_constants, SC_PushConstant});
    }
}

void SpirvCompute::begin_compute_invocations(uint32_t& lbl, uint32_t& local_x, uint32_t& local_y,
                                             uint32_t& local_z) {
    put(code, Op_Function, {t_void, f_main, FC_None, t_fn});
    put(code, Op_Label, {lbl});
    cur_block = lbl;
    function_var_insert = code.size();
    uint32_t ldg = id();
    put(code, Op_Load, {t_v3u, ldg, v_groupid});
    for (uint32_t c = 0; c < 3; c++) {
        groupid[c] = id();
        put(code, Op_CompositeExtract, {t_u32, groupid[c], ldg, c});
    }
    if (native_subgroup_size) {
        // Vulkan does not promise that LocalInvocationIndex follows subgroup lane order.  A
        // full, exact-size subgroup does promise that SubgroupId/SubgroupLocalInvocationId
        // identify every workgroup invocation once, so assign guest coordinates in that order.
        // This makes one Vulkan subgroup exactly one consecutive RDNA wave without depending on
        // the implementation's otherwise-unspecified local-invocation mapping.
        linear_localid = ibin(Op_IAdd, ibin(Op_IMul, subgroup_id(), uconst(native_subgroup_size)),
                              subgroup_local_id());
        localid_comp[0] = ibin(Op_UMod, linear_localid, uconst(local_x));
        const uint32_t linear_yz = ibin(Op_UDiv, linear_localid, uconst(local_x));
        localid_comp[1] = ibin(Op_UMod, linear_yz, uconst(local_y));
        localid_comp[2] = ibin(Op_UDiv, linear_yz, uconst(local_y));
        const uint32_t local_size[3] = {local_x, local_y, local_z};
        for (uint32_t c = 0; c < 3; ++c)
            globalid_comp[c] =
                ibin(Op_IAdd, ibin(Op_IMul, groupid[c], uconst(local_size[c])), localid_comp[c]);
    } else {
        uint32_t ld = id();
        put(code, Op_Load, {t_v3u, ld, v_gid});
        for (uint32_t c = 0; c < 3; c++) {
            globalid_comp[c] = id();
            put(code, Op_CompositeExtract, {t_u32, globalid_comp[c], ld, c});
        }
        uint32_t ldl = id();
        put(code, Op_Load, {t_v3u, ldl, v_localid});
        for (uint32_t c = 0; c < 3; c++) {
            localid_comp[c] = id();
            put(code, Op_CompositeExtract, {t_u32, localid_comp[c], ldl, c});
        }
        // Vulkan and RDNA both linearize X fastest, then Y, then Z.
        uint32_t yz = ibin(Op_IMul, localid_comp[2], uconst(local_y));
        yz = ibin(Op_IAdd, yz, localid_comp[1]);
        linear_localid = ibin(Op_IAdd, ibin(Op_IMul, yz, uconst(local_x)), localid_comp[0]);
    }
    gidx = globalid_comp[0];
    localid = localid_comp[0];
}

void SpirvCompute::begin(uint32_t input_stride, const ShaderResourceTable* rt, uint32_t local_x,
                         uint32_t local_y, uint32_t local_z, uint32_t hardware_wave_size,
                         uint32_t push_constant_dwords, bool raw_word_output, bool raw_word_input) {
    stride = input_stride;
    raw_output_words = raw_word_output;
    raw_input_words = raw_word_input;
    push_constant_dword_count = push_constant_dwords;
    local_count = local_x * local_y * local_z;
    guest_local_size = {local_x, local_y, local_z};
    wave_size = hardware_wave_size;
    t_void = id();
    t_fn = id();
    t_f32 = id();
    t_u32 = id();
    t_i32 = id();
    t_v3u = id();
    t_bool = id();
    t_v4f =
        id();   // vec4<float>: needed by the sampled-texture path (image_sample) in a compute shader
    uint32_t t_ptr_in_v3u = id();
    v_gid = id();
    v_groupid = id();
    v_localid = id();
    uint32_t t_rta = id(), t_struct = id(), t_ptr_sb_struct = id();
    v_in = id();
    v_out = id();
    t_ptr_sb_f32 = id();
    f_main = id();
    uint32_t lbl = id();
    glsl = id();

    put(caps, Op_Capability, {Cap_Shader});
    declare_float_controls(f_main);
    {
        std::vector<uint32_t> o{glsl};
        pstr(o, "GLSL.std.450");
        putv(extimp, Op_ExtInstImport, o);
    }
    put(mem, Op_MemoryModel, {Addr_Logical, Mem_GLSL450});
    is_compute = true;
    guest_stage = GuestShaderStage::Compute;
    physical_domain = PhysicalExecutionDomain::Workgroup;
    begin_compute_layout(local_x, local_y, local_z, t_rta, t_struct, push_constant_dwords);
    put(types, Op_TypePointer, {t_ptr_in_v3u, SC_Input, t_v3u});
    put(types, Op_Variable, {t_ptr_in_v3u, v_gid, SC_Input});
    put(types, Op_Variable, {t_ptr_in_v3u, v_groupid, SC_Input});
    put(types, Op_Variable, {t_ptr_in_v3u, v_localid, SC_Input});
    put(types, Op_TypeRuntimeArray, {t_rta, raw_input_words ? t_u32 : t_f32});
    put(types, Op_TypeStruct, {t_struct, t_rta});
    put(types, Op_TypePointer, {t_ptr_sb_struct, SC_StorageBuffer, t_struct});
    declare_external_storage_buffer(t_ptr_sb_struct, v_in);
    put(types, Op_TypePointer, {t_ptr_sb_f32, SC_StorageBuffer, raw_input_words ? t_u32 : t_f32});
    if (raw_output_words) {
        declare_cbufs(rt);   // supplies the uint32 Block type used by the raw export sink
        declare_external_storage_buffer(t_ptr_sb_struct_u, v_out);
    } else {
        // Preserve the declaration order (and byte identity) of every existing compute module.
        declare_external_storage_buffer(t_ptr_sb_struct, v_out);
        declare_cbufs(rt);   // scalar-memory buffers, including table-assigned bindings beyond 2/3
    }
    begin_compute_invocations(lbl, local_x, local_y, local_z);
}
}   // namespace prosper::gpu
